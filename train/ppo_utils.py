"""Helpers shared by the PPO training entry points."""

from __future__ import annotations

import os

import torch

from gpu_poker import constants as c
from gpu_poker.policy import NUM_RAISE_BUCKETS


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def debug_numeric_enabled() -> bool:
    return os.environ.get("POKERGPU_DEBUG_NUMERIC", "0") == "1"


def assert_finite(name: str, x: torch.Tensor) -> None:
    if not debug_numeric_enabled():
        return
    if torch.isfinite(x).all():
        return
    bad = (~torch.isfinite(x)).sum().item()
    raise RuntimeError(f"Non-finite tensor: {name} (bad={bad}, shape={tuple(x.shape)})")


def bucket_stats(raise_bucket_1d: torch.Tensor) -> dict[str, float]:
    """Per-bucket fraction of raise actions (raise_bucket: [K] int64 in [0, B))."""
    out: dict[str, float] = {}
    if raise_bucket_1d.numel() == 0:
        for b in range(NUM_RAISE_BUCKETS):
            out[f"bucket_{b}_pct"] = 0.0
        return out
    total = float(raise_bucket_1d.numel())
    for b in range(NUM_RAISE_BUCKETS):
        out[f"bucket_{b}_pct"] = float(raise_bucket_1d.eq(b).sum().item()) / total
    return out


def action_stats(action_type: torch.Tensor, raise_bucket: torch.Tensor) -> dict[str, float]:
    # action_type: [T,N] int64, raise_bucket: [T,N] int64
    t, n = action_type.shape
    total = float(t * n)
    counts = {}
    for a in (c.ACTION_FOLD, c.ACTION_CHECK, c.ACTION_CALL, c.ACTION_RAISE):
        counts[a] = float(action_type.eq(a).sum().item())
    raise_mask = action_type.eq(c.ACTION_RAISE)
    rb = raise_bucket.squeeze(-1) if raise_bucket.ndim > 2 else raise_bucket
    rb_at_raise = rb[raise_mask]
    out = {
        "pct_fold": counts[c.ACTION_FOLD] / total,
        "pct_check": counts[c.ACTION_CHECK] / total,
        "pct_call": counts[c.ACTION_CALL] / total,
        "pct_raise": counts[c.ACTION_RAISE] / total,
    }
    out.update(bucket_stats(rb_at_raise))
    return out


def action_stats_masked(
    action_type: torch.Tensor, raise_bucket: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    # action_type: [T,N] int64, raise_bucket: [T,N] int64, mask: [T,N] bool
    mask = mask.to(dtype=torch.bool)
    total = float(mask.sum().item())
    if total <= 0:
        out = {
            "pct_fold": 0.0,
            "pct_check": 0.0,
            "pct_call": 0.0,
            "pct_raise": 0.0,
        }
        out.update(bucket_stats(torch.empty(0, dtype=torch.int64)))
        return out
    at = action_type[mask]
    rb = raise_bucket.squeeze(-1) if raise_bucket.ndim > 2 else raise_bucket
    rb_all = rb[mask]
    counts = {}
    for a in (c.ACTION_FOLD, c.ACTION_CHECK, c.ACTION_CALL, c.ACTION_RAISE):
        counts[a] = float(at.eq(a).sum().item())
    raise_mask = at.eq(c.ACTION_RAISE)
    rb_at_raise = rb_all[raise_mask]
    out = {
        "pct_fold": counts[c.ACTION_FOLD] / total,
        "pct_check": counts[c.ACTION_CHECK] / total,
        "pct_call": counts[c.ACTION_CALL] / total,
        "pct_raise": counts[c.ACTION_RAISE] / total,
    }
    out.update(bucket_stats(rb_at_raise))
    return out


def action_stats_by_street(
    *,
    action_type: torch.Tensor,  # [T,N]
    raise_bucket: torch.Tensor,  # [T,N]
    scalars: torch.Tensor,  # [T,N,S]
    base_mask: torch.Tensor,  # [T,N] bool
) -> dict[str, dict[str, float]]:
    # scalars[...,7] is stage normalized by 5.0; map back to int stage.
    stage = (scalars[..., 7] * 5.0).round().to(dtype=torch.int64).clamp_(0, 5)
    out: dict[str, dict[str, float]] = {}
    for name, st in [
        ("preflop", c.STAGE_PREFLOP),
        ("flop", c.STAGE_FLOP),
        ("turn", c.STAGE_TURN),
        ("river", c.STAGE_RIVER),
    ]:
        m = base_mask & stage.eq(int(st))
        out[name] = action_stats_masked(action_type, raise_bucket, m)
    return out


def compute_gae(
    *,
    rewards: torch.Tensor,  # [T,B]
    dones: torch.Tensor,  # [T,B] bool
    values: torch.Tensor,  # [T,B]
    last_value: torch.Tensor,  # [B]
    player_id: torch.Tensor,  # [T,B] int, value perspective for each timestep
    last_player_id: torch.Tensor,  # [B] int, value perspective for last_value
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_steps, batch = rewards.shape
    advantages = torch.zeros((t_steps, batch), device=rewards.device, dtype=torch.float32)
    gae = torch.zeros((batch,), device=rewards.device, dtype=torch.float32)
    next_value = last_value
    for t in range(t_steps - 1, -1, -1):
        not_done = (~dones[t]).to(dtype=torch.float32)
        next_player = last_player_id if t == t_steps - 1 else player_id[t + 1]
        perspective_sign = torch.where(
            next_player.eq(player_id[t]),
            torch.ones((batch,), device=rewards.device, dtype=torch.float32),
            -torch.ones((batch,), device=rewards.device, dtype=torch.float32),
        )
        delta = rewards[t] + gamma * not_done * perspective_sign * next_value - values[t]
        gae = delta + gamma * gae_lambda * not_done * perspective_sign * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def compute_mc_returns_p0(
    *,
    reward_p0: torch.Tensor,  # [T,N] float32 terminal-only (0 otherwise)
    dones: torch.Tensor,  # [T,N] bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte-Carlo episode returns from the P0 perspective.

    Used for league training where the opponent may be a different policy. Since env
    observations are active-player (include the acting player's private hole cards),
    we avoid requiring value targets on opponent-to-act states to prevent information leaks.

    Returns:
      returns_p0: [T,N] terminal return propagated backward within the rollout
      valid:      [T,N] True if the episode terminates within the rollout window
    """
    t_steps, n = reward_p0.shape
    returns = torch.zeros_like(reward_p0)
    valid = torch.zeros_like(dones)
    ret = torch.zeros((n,), device=reward_p0.device, dtype=torch.float32)
    has = torch.zeros((n,), device=reward_p0.device, dtype=torch.bool)
    for t in range(t_steps - 1, -1, -1):
        term = dones[t]
        ret = torch.where(term, reward_p0[t], ret)
        has = term | has
        returns[t] = ret
        valid[t] = has
    return returns, valid


def select_player_state(
    *,
    player_id: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    h1: torch.Tensor,
    c1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # player_id: [N] in {0,1}; each h/c: [1,N,H]
    sel = player_id.view(1, -1, 1).to(dtype=torch.bool)
    h = torch.where(sel, h1, h0)
    c_ = torch.where(sel, c1, c0)
    return h, c_


def scatter_player_state(
    *,
    player_id: torch.Tensor,
    h_new: torch.Tensor,
    c_new: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    h1: torch.Tensor,
    c1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Update only the acting player's slot for each env.
    is_p1 = player_id.to(dtype=torch.bool).view(1, -1, 1)
    h0 = torch.where(is_p1, h0, h_new)
    c0 = torch.where(is_p1, c0, c_new)
    h1 = torch.where(is_p1, h_new, h1)
    c1 = torch.where(is_p1, c_new, c1)
    return h0, c0, h1, c1
