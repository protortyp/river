"""AlphaStar-style triad controller for HUNL training.

Owns three concurrent agents -- Main (M), Main Exploiter (ME), League
Exploiter (LE) -- each with its own network, optimizer, and rollout
buffer, training on a partitioned slice of `num_envs`. See the
implementation plan ("G -- AlphaStar Triad") for the architectural
decisions baked into this module.

Key invariants:
- The three networks are independent ``nn.Module`` instances. There is
  no parameter sharing -- ME and LE's whole purpose is to discover
  policies M wouldn't explore on its own, which requires distinct
  parameter trajectories.
- Each role's PPO update goes through its own ``loss.backward()`` and
  its own ``optimizer.step()``. Gradient isolation is asserted by
  ``test_gradient_isolation_me_backward_doesnt_touch_main``.
- The env partition is contiguous and stable for the lifetime of a run:
  ``[0..main_n) [main_n..main_n+me_n) [main_n+me_n..num_envs)``. We do
  not reshuffle slices mid-run; doing so would invalidate the buffered
  rollouts of in-flight hands.
- Snapshot lifecycle is centralized here. M snapshots every K updates;
  ME snapshots on win-rate trigger then ALWAYS resets to M's params;
  LE snapshots on a stricter "beat everyone >=70%" trigger and
  resets with probability ``triad_le_reset_prob``.

ASCII data flow (one rollout step):

       ┌────────────────────────────────────────────────┐
       │  env (shared, num_envs=N total)                 │
       └────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    M slice           ME slice           LE slice
    envs [0, m)       envs [m, m+e)      envs [m+e, N)
    net_main          net_me             net_le
    buf_main          buf_me             buf_le

PPO update (per epoch):
    loss_main.backward(); opt_main.step()    # isolated to net_main
    loss_me.backward();   opt_me.step()      # isolated to net_me
    loss_le.backward();   opt_le.step()      # isolated to net_le
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from train.league import OpponentMeta, pfsp_weights_from_bb_per_hand

if TYPE_CHECKING:
    # Avoid runtime circular import. train.train imports this module.
    from gpu_poker.policy import PokerPolicyNet
    from gpu_poker.policy_transformer import PokerTransformerPolicyNet
    from train.train import RolloutBuffer, TrainConfig

    # Either policy backbone. Both expose the same forward_step / evaluate_step
    # / lstm_hidden contract (see train._make_policy), so the controller treats
    # them interchangeably.
    PolicyNet = PokerPolicyNet | PokerTransformerPolicyNet

# Role identifiers used in every per-role mapping. Stable strings (no
# enums) so config / logs / TB namespaces all read identically.
ROLE_MAIN: str = "main"
ROLE_ME: str = "me"
ROLE_LE: str = "le"
ROLES: tuple[str, ...] = (ROLE_MAIN, ROLE_ME, ROLE_LE)

# Opponent-kind encoding for sample_opponents (extends the single-net
# encoding used in train.py with OPP_LIVE_MAIN for the triad).
#   0 = self-play (opponent is this role's OWN live net)
#   1 = scripted bot         (opp_id -> index into opponent_metas)
#   2 = snapshot from pool   (opp_id -> index into opponent_metas)
#   3 = live Main net        (opp_id ignored; resolved at forward-time
#                             to the live M params -- used by ME)
OPP_SELF: int = 0
OPP_BOT: int = 1
OPP_SNAPSHOT: int = 2
OPP_LIVE_MAIN: int = 3

# agent_role tags used in the unified opponent pool. We centralize them
# here so the role-aware filters used in sample_opponents stay in lockstep
# with the snapshot writer in T5.
ROLE_TAG_MAIN_CURRENT: str = "main"
ROLE_TAG_MAIN_HISTORICAL: str = "main_historical"
ROLE_TAG_LE_HISTORICAL: str = "league_exploiter"
ROLE_TAG_ME_HISTORICAL: str = "main_exploiter"


@dataclass(frozen=True)
class RoleSpec:
    """Immutable per-role configuration extracted from TrainConfig.

    One ``RoleSpec`` per role. Crystallizes the bits of TrainConfig that
    are role-specific into a single object the controller can reason
    about uniformly.

    Fields:
      name: role identifier (one of ROLES).
      env_frac: fraction of num_envs allocated to this role.
      lr / entropy_coef: per-role optimizer hyperparameters (resolved
        from TrainConfig with global fallback).
      snapshot_every: M-style periodic snapshot cadence in updates,
        or ``None`` for WR-gated roles.
      snapshot_wr_threshold: ME/LE-style WR gate, or ``None`` for M.
      snapshot_pool_cap: per-role FIFO cap on retained snapshots.
      mutate_kind: behaviour after snapshot. One of:
        - "never"           (M)
        - "reset_to_main"   (ME)
        - "random_with_prob"(LE, gated by mutate_prob)
      mutate_prob: probability of reset for "random_with_prob".
    """

    name: str
    env_frac: float
    lr: float
    entropy_coef: float
    snapshot_every: int | None
    snapshot_wr_threshold: float | None
    snapshot_pool_cap: int
    mutate_kind: str
    mutate_prob: float


@dataclass
class RoleState:
    """Mutable per-role state held by the controller.

    Mirrors what train.py's main loop tracks today for a single net
    (net, optimizer, buffer, h0/c0/h1/c1, snapshot bookkeeping) but
    scoped to one role's env slice.
    """

    spec: RoleSpec
    net: PolicyNet
    opt: torch.optim.Optimizer
    buf: RolloutBuffer

    # Env-slice bookkeeping. `env_slice` is a contiguous range into the
    # shared env's [0, num_envs) index space; `slice_n = stop - start`.
    env_start: int
    env_stop: int

    # Per-seat LSTM/transformer state for *this role's* env slice only.
    # Shape: [1, slice_n, hidden] fp32.
    h0: torch.Tensor
    c0: torch.Tensor
    h1: torch.Tensor
    c1: torch.Tensor

    # Snapshot bookkeeping.
    snapshot_count: int = 0
    last_snapshot_update: int = -1
    # The id of the M snapshot this ME/LE was forked from (for filename
    # provenance per D10). None for M.
    parent_main_id: str | None = None
    # Most recent eval-WR per target (e.g., {"main_current": 0.62,
    # "main_historical_5": 0.71, ...}). Refreshed by
    # ``schedule_async_eval``; read by ``maybe_snapshot``.
    eval_wr_vs_targets: dict[str, float] = field(default_factory=dict)

    @property
    def slice_n(self) -> int:
        return self.env_stop - self.env_start


@dataclass(frozen=True)
class SnapshotDecision:
    """One role's snapshot decision for a given update.

    Returned by ``TriadController.maybe_snapshot``; consumed by train.py
    (T6) to actually write the .pt file and call
    ``mutate_after_snapshot``. The controller does not own disk I/O.

    Fields:
      role: one of ROLES.
      should_snapshot: True iff the trigger fired this update.
      reason: human-readable explanation -- "periodic", "wr_threshold",
        "below_wr_threshold", "no_eval_data", "noop_M_only_periodic".
      agent_role_tag: the ``OpponentMeta.agent_role`` to tag the new
        snapshot with so the per-role samplers in T3 see it correctly.
        Empty when ``should_snapshot=False``.
      mutate_kind: post-snapshot mutation to apply (see RoleSpec
        docstring). Caller MUST call ``mutate_after_snapshot(role)``
        after writing the .pt file iff ``should_snapshot`` is True.
    """

    role: str
    should_snapshot: bool
    reason: str
    agent_role_tag: str
    mutate_kind: str


def _role_specs_from_cfg(cfg: TrainConfig) -> dict[str, RoleSpec]:
    """Crystallize the triad section of TrainConfig into one RoleSpec
    per role. Resolves per-role hparam overrides with global fallback.
    """

    def _resolve(role_val: float | None, global_val: float) -> float:
        return float(global_val if role_val is None else role_val)

    return {
        ROLE_MAIN: RoleSpec(
            name=ROLE_MAIN,
            env_frac=cfg.triad_main_frac,
            lr=_resolve(cfg.triad_lr_main, cfg.lr),
            entropy_coef=cfg.entropy_coef,  # M never overrides entropy
            snapshot_every=int(cfg.triad_main_snapshot_every),
            snapshot_wr_threshold=None,
            snapshot_pool_cap=int(cfg.triad_main_pool_cap),
            mutate_kind="never",
            mutate_prob=0.0,
        ),
        ROLE_ME: RoleSpec(
            name=ROLE_ME,
            env_frac=cfg.triad_me_frac,
            lr=_resolve(cfg.triad_lr_me, cfg.lr),
            entropy_coef=_resolve(cfg.triad_entropy_me, cfg.entropy_coef),
            snapshot_every=None,
            snapshot_wr_threshold=float(cfg.triad_me_promote_wr),
            snapshot_pool_cap=int(cfg.triad_me_pool_cap),
            mutate_kind="reset_to_main",
            mutate_prob=1.0,  # always
        ),
        ROLE_LE: RoleSpec(
            name=ROLE_LE,
            env_frac=cfg.triad_le_frac,
            lr=_resolve(cfg.triad_lr_le, cfg.lr),
            entropy_coef=_resolve(cfg.triad_entropy_le, cfg.entropy_coef),
            snapshot_every=None,
            snapshot_wr_threshold=float(cfg.triad_le_promote_wr),
            snapshot_pool_cap=int(cfg.triad_le_pool_cap),
            mutate_kind="random_with_prob",
            mutate_prob=float(cfg.triad_le_reset_prob),
        ),
    }


def _compute_env_partition(num_envs: int, specs: dict[str, RoleSpec]) -> dict[str, tuple[int, int]]:
    """Split [0, num_envs) into three contiguous slices per env_frac.

    Returns ``{role_name: (start, stop)}``. The last role (LE) absorbs
    any int-truncation slack so the slices partition num_envs exactly.
    """
    main_n = int(num_envs * specs[ROLE_MAIN].env_frac)
    me_n = int(num_envs * specs[ROLE_ME].env_frac)
    le_n = num_envs - main_n - me_n  # absorb slack
    return {
        ROLE_MAIN: (0, main_n),
        ROLE_ME: (main_n, main_n + me_n),
        ROLE_LE: (main_n + me_n, main_n + me_n + le_n),
    }


# --------------------------------------------------------------------------- #
# Opponent-pool filters and PFSP helpers (used by sample_opponents)
# --------------------------------------------------------------------------- #


def _snapshot_indices_by_role(metas: list[OpponentMeta], allowed_roles: set[str]) -> list[int]:
    """Return indices into ``metas`` whose snapshot agent_role matches.

    Scripted bots are never returned here. Each role's sampling rules
    decide separately whether to merge bots in.
    """
    return [
        i for i, m in enumerate(metas) if m.kind == "snapshot" and m.agent_role in allowed_roles
    ]


def _forgotten_main_historical(
    metas: list[OpponentMeta], indices: list[int], fraction: float = 0.5
) -> list[int]:
    """Older-than-median half of ``indices`` by ``env_steps``.

    "Forgotten" is approximated by age: the snapshots M has been training
    away from for the longest are the ones the current policy is most
    likely to have catastrophically forgotten how to beat. Refine into a
    "last-sampled-update" metric only if/when this proxy underperforms.
    """
    if not indices:
        return []
    sorted_idx = sorted(indices, key=lambda i: metas[i].env_steps or 0)
    cutoff = max(1, int(len(sorted_idx) * fraction))
    return sorted_idx[:cutoff]


class _PFSPKwargs(TypedDict):
    """Strongly-typed PFSP kwargs to keep ``**_pfsp_kwargs(...)`` type-safe."""

    mode: str
    temperature: float
    epsilon: float
    q: float


def _pfsp_sample_indices(
    pool_indices: list[int],
    scores: torch.Tensor,
    *,
    n_samples: int,
    mode: str,
    temperature: float,
    epsilon: float,
    q: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Draw ``n_samples`` opponent indices from ``pool_indices`` via PFSP.

    Returns an int64 tensor of length ``n_samples`` on ``device``, or
    ``None`` if the pool is empty / ``n_samples`` is 0. Callers must
    handle the ``None`` fallback explicitly (typically to OPP_SELF).
    """
    if not pool_indices or n_samples <= 0:
        return None
    idx_t = torch.tensor(pool_indices, device=device, dtype=torch.int64)
    weights = pfsp_weights_from_bb_per_hand(
        scores[idx_t],
        temperature=temperature,
        epsilon=epsilon,
        mode=mode,
        q=q,
    )
    weights = weights / weights.sum().clamp_min(1e-12)
    picks = torch.multinomial(weights, num_samples=n_samples, replacement=True)
    return idx_t[picks]


# --------------------------------------------------------------------------- #
# T4: per-role PPO update (gradient-isolation primitive)
# --------------------------------------------------------------------------- #


def _ppo_update_one_role(
    *,
    rs: RoleState,
    cfg: TrainConfig,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    train_mask: torch.Tensor | None,
    device: torch.device,
    scalar_dim: int,
) -> dict[str, float]:
    """Inner PPO loop for one role's RolloutBuffer.

    Mirrors the league-path PPO update in train.py (MC returns,
    train-mask-weighted loss, clipped surrogate + value + entropy) but
    scoped to one net/opt with no AMP/scaler tangle. T6 may wrap calls in
    ``torch.autocast`` externally; the role's parameters and the loss
    accumulators stay fp32 either way.

    The forward / backward / step here only touch ``rs.net`` and
    ``rs.opt``; gradient isolation is therefore inherent.
    """
    buf = rs.buf
    t_steps = buf.t
    slice_n = buf.n
    total = t_steps * slice_n
    expected = (t_steps, slice_n)
    if tuple(returns.shape) != expected:
        raise ValueError(
            f"returns shape {tuple(returns.shape)} != expected {expected} for role {rs.spec.name}"
        )
    if tuple(advantages.shape) != expected:
        raise ValueError(
            f"advantages shape {tuple(advantages.shape)} != expected {expected} "
            f"for role {rs.spec.name}"
        )

    # Flatten for minibatching.
    cards_f = buf.cards.reshape(-1, 7)
    scalars_f = buf.scalars.reshape(-1, scalar_dim)
    mask_f = buf.action_mask.reshape(-1, buf.action_mask.shape[-1])
    action_type_f = buf.action_type.reshape(-1)
    raise_bucket_f = buf.raise_bucket.reshape(-1)
    old_logprob_f = buf.logprob.reshape(-1)
    h_f = buf.h.reshape(-1, buf.hidden)
    c_f = buf.c.reshape(-1, buf.hidden)
    returns_f = returns.reshape(-1).to(device=device, dtype=torch.float32)
    adv_f = advantages.reshape(-1).to(device=device, dtype=torch.float32)
    mask_f_flat: torch.Tensor | None = (
        train_mask.reshape(-1).to(device=device, dtype=torch.bool)
        if train_mask is not None
        else None
    )

    mb_size = int(cfg.minibatch_size)
    num_mbs = (total + mb_size - 1) // max(1, mb_size)

    # GPU tensor accumulators -- crucial perf fix. The previous version
    # called .item() on 7 tensors per minibatch, each forcing a full CUDA
    # synchronize (CPU waits for all pending kernels). At 256 minibatches
    # per role x 3 roles = 5,376 forced syncs per PPO update on the triad
    # path, which dominated wall-clock. The non-triad path (train.py:1032,
    # 1670+) already uses GPU accumulators; aligning here.
    loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    policy_loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    value_loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    entropy_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    kl_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    kl_abs_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    clipfrac_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    # Pre-clip gradient norm accumulator (audit gap #2 fix). clip_grad_norm_
    # returns this tensor, so logging it is essentially free.
    grad_norm_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    # Explained variance: 1 - Var(returns - value) / Var(returns). Crucial
    # value-function quality signal -- if EV collapses the critic isn't
    # learning. Non-triad path computes this at train.py:1494/1544/1737;
    # triad path was missing it entirely (audit gap #2).
    ev_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    # Raw variance + residual variance so we can distinguish
    # "EV low because critic bad" from "EV low because target variance huge".
    var_returns_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    var_residual_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    value_mean_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    returns_mean_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    mb_count = 0

    for _epoch in range(int(cfg.num_epochs)):
        offset_mb = (_epoch * 104729) % max(1, num_mbs)
        for mb_i in range(num_mbs):
            j = (mb_i + offset_mb) % num_mbs
            start = j * mb_size
            end = min(total, start + mb_size)
            if end <= start:
                continue
            sl = slice(start, end)

            eval_out = rs.net.evaluate_step(
                cards=cards_f[sl],
                scalars=scalars_f[sl],
                action_mask=mask_f[sl],
                action_type=action_type_f[sl],
                raise_bucket=raise_bucket_f[sl],
                h=h_f[sl].unsqueeze(0),
                c=c_f[sl].unsqueeze(0),
                terminated=None,
            )

            log_ratio = (eval_out.logprob - old_logprob_f[sl]).clamp(min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            unclipped = ratio * adv_f[sl]
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_f[sl]

            if mask_f_flat is not None:
                w = mask_f_flat[sl].to(dtype=torch.float32)
                denom = w.sum().clamp_min(1.0)
                policy_loss = -(w * torch.minimum(unclipped, clipped)).sum() / denom
                value_loss = 0.5 * (w * (eval_out.value - returns_f[sl]).pow(2)).sum() / denom
                entropy_bonus = (w * eval_out.entropy).sum() / denom
                clipfrac = ((torch.abs(ratio - 1.0) > cfg.clip_eps) & w.bool()).to(
                    dtype=torch.float32
                ).sum() / denom
                kl_diff = (-log_ratio) * w
                approx_kl = kl_diff.sum() / denom
                approx_kl_abs = kl_diff.abs().sum() / denom
            else:
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (eval_out.value - returns_f[sl]).pow(2).mean()
                entropy_bonus = eval_out.entropy.mean()
                clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_eps).to(dtype=torch.float32).mean()
                approx_kl = (-log_ratio).mean()
                approx_kl_abs = (-log_ratio).abs().mean()

            loss = (
                policy_loss
                + float(cfg.value_coef) * value_loss
                - float(rs.spec.entropy_coef) * entropy_bonus
            )

            # Explained variance over this minibatch -- computed BEFORE backward
            # so we use the same forward-pass values. Done with the same
            # train_mask weighting as the losses so EV reflects only valid
            # samples.
            with torch.no_grad():
                v = eval_out.value.detach().to(dtype=torch.float32)
                r = returns_f[sl]
                if mask_f_flat is not None:
                    w = mask_f_flat[sl].to(dtype=torch.float32)
                    n_w = w.sum().clamp_min(1.0)
                    r_mean = (w * r).sum() / n_w
                    var_r = (w * (r - r_mean).pow(2)).sum() / n_w
                    res = r - v
                    res_mean = (w * res).sum() / n_w
                    var_res = (w * (res - res_mean).pow(2)).sum() / n_w
                    v_mean = (w * v).sum() / n_w
                else:
                    r_mean = r.mean()
                    var_r = r.var(unbiased=False)
                    var_res = (r - v).var(unbiased=False)
                    v_mean = v.mean()
                ev_mb = 1.0 - var_res / var_r.clamp_min(1e-8)

            rs.opt.zero_grad(set_to_none=True)
            loss.backward()
            # clip_grad_norm_ returns the pre-clip norm as a 0-d tensor; capture
            # it for free instead of dropping it on the floor.
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                rs.net.parameters(), max_norm=cfg.max_grad_norm
            )
            rs.opt.step()

            # GPU accumulators -- no .item() inside the inner loop. Each
            # .item() previously stalled the GPU pipeline waiting for all
            # pending kernels to finish; with 256 minibatches and 7 such
            # syncs per minibatch this was the dominant cost of PPO.
            loss_sum_t = loss_sum_t + loss.detach()
            policy_loss_sum_t = policy_loss_sum_t + policy_loss.detach()
            value_loss_sum_t = value_loss_sum_t + value_loss.detach()
            entropy_sum_t = entropy_sum_t + entropy_bonus.detach()
            kl_sum_t = kl_sum_t + approx_kl.detach()
            kl_abs_sum_t = kl_abs_sum_t + approx_kl_abs.detach()
            clipfrac_sum_t = clipfrac_sum_t + clipfrac.detach()
            grad_norm_sum_t = grad_norm_sum_t + grad_norm_t.detach()
            ev_sum_t = ev_sum_t + ev_mb.detach()
            var_returns_sum_t = var_returns_sum_t + var_r.detach()
            var_residual_sum_t = var_residual_sum_t + var_res.detach()
            value_mean_sum_t = value_mean_sum_t + v_mean.detach()
            returns_mean_sum_t = returns_mean_sum_t + r_mean.detach()
            mb_count += 1

    # ONE sync at the end of all minibatches: pack the means into a single
    # tensor, transfer with one .tolist() call (a single H2D wait).
    den_t = torch.tensor(float(max(1, mb_count)), device=device, dtype=torch.float32)
    means_t = torch.stack(
        [
            loss_sum_t / den_t,
            policy_loss_sum_t / den_t,
            value_loss_sum_t / den_t,
            entropy_sum_t / den_t,
            kl_sum_t / den_t,
            kl_abs_sum_t / den_t,
            clipfrac_sum_t / den_t,
            grad_norm_sum_t / den_t,
            ev_sum_t / den_t,
            var_returns_sum_t / den_t,
            var_residual_sum_t / den_t,
            value_mean_sum_t / den_t,
            returns_mean_sum_t / den_t,
        ]
    )
    means = means_t.detach().cpu().tolist()
    return {
        "loss": means[0],
        "policy_loss": means[1],
        "value_loss": means[2],
        "entropy": means[3],
        "approx_kl": means[4],
        "approx_kl_abs": means[5],
        "clipfrac": means[6],
        "grad_norm": means[7],
        "explained_variance": means[8],
        "var_returns": means[9],
        "var_residual": means[10],
        "value_mean": means[11],
        "returns_mean": means[12],
        "mb_count": float(mb_count),
    }


class TriadController:
    """Owns three role networks + optimizers + rollout buffers + state.

    The controller is *the* source of truth for per-role training state
    when ``cfg.triad_enabled=true``. train.py's main loop holds one
    reference and routes per-role calls (sample_opponents, forward_step,
    ppo_update, maybe_snapshot) through it.

    Construction takes a ``make_net`` callable so that this module does
    not import the policy factory directly (which lives in train.py).
    The callable receives no arguments and returns a fresh net on the
    target device.
    """

    def __init__(
        self,
        *,
        cfg: TrainConfig,
        scalar_dim: int,
        device: torch.device,
        make_net: Callable[[], PolicyNet],
        state_dtype: torch.dtype = torch.float32,
    ) -> None:
        # Deferred import to break circular import (train.train imports
        # this module). RolloutBuffer is the buffer class we already use
        # for the single-net path; we instantiate one per role.
        from train.train import RolloutBuffer

        self.cfg = cfg
        self.scalar_dim = int(scalar_dim)
        self.device = device
        # Retained for the LE "random_with_prob" mutate path and for T9
        # per-role resume (re-instantiate a fresh net, then load).
        self._make_net: Callable[[], PolicyNet] = make_net

        specs = _role_specs_from_cfg(cfg)
        partition = _compute_env_partition(int(cfg.num_envs), specs)

        self.roles: dict[str, RoleState] = {}
        for role in ROLES:
            spec = specs[role]
            env_start, env_stop = partition[role]
            slice_n = env_stop - env_start

            net = make_net()
            # torch.compile each role's net when cfg.compile_policy is set.
            # The historical concern was that mutate_after_snapshot's
            # load_state_dict would invalidate the compiled graph. In
            # practice load_state_dict uses .data.copy_() which preserves
            # the parameter tensor identities the graph captures, so the
            # swap is transparent. Snapshot opponents loaded ad-hoc via
            # _load_snapshot remain uncompiled (they only run forward and
            # change per-step, so compile-and-cache wouldn't pay off).
            if bool(getattr(cfg, "compile_policy", False)):
                # mode="default": kernel fusion via Inductor without CUDA
                # graphs. Less aggressive than "reduce-overhead" but more
                # tolerant of the data-dependent control flow in
                # evaluate_step (reset masks, terminated branch, etc.).
                # Note: empirically had no measurable PPO speedup on
                # transformer triad (tested 2026-05-17 on RTX A5000 +
                # RTX 3090: 109s baseline vs 109s with compile, both
                # modes). Left in for completeness / future PyTorch
                # versions that may improve Inductor fusion on this
                # workload.
                # torch.compile is typed as returning a bare Callable; the
                # OptimizedModule it actually returns proxies every attribute
                # (evaluate_step / lstm_hidden / state_dict / ...) to the
                # wrapped net, so treating it as a PolicyNet is sound.
                net = cast("PolicyNet", torch.compile(net, mode="default", dynamic=False))
            # fused=True: single CUDA kernel for the optimizer step. With 256
            # minibatches per role per update on the triad path, the per-step
            # overhead of unfused Adam (one launch per parameter tensor) was a
            # major chunk of PPO wall time. Non-triad path already uses
            # fused=True (train.py:793); align here.
            opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, fused=True)

            buf = RolloutBuffer(
                t=int(cfg.rollout_steps),
                n=slice_n,
                scalar_dim=self.scalar_dim,
                hidden=int(net.lstm_hidden),
                device=device,
            )

            h0 = torch.zeros((1, slice_n, int(net.lstm_hidden)), device=device, dtype=state_dtype)
            c0 = torch.zeros((1, slice_n, int(net.lstm_hidden)), device=device, dtype=state_dtype)
            h1 = torch.zeros((1, slice_n, int(net.lstm_hidden)), device=device, dtype=state_dtype)
            c1 = torch.zeros((1, slice_n, int(net.lstm_hidden)), device=device, dtype=state_dtype)

            self.roles[role] = RoleState(
                spec=spec,
                net=net,
                opt=opt,
                buf=buf,
                env_start=env_start,
                env_stop=env_stop,
                h0=h0,
                c0=c0,
                h1=h1,
                c1=c1,
            )

    # ----------------------------------------------------------- accessors

    def role(self, name: str) -> RoleState:
        if name not in self.roles:
            raise KeyError(f"Unknown role: {name!r}. Valid: {ROLES}")
        return self.roles[name]

    def env_slice(self, role: str) -> slice:
        """Contiguous Python slice into the [0, num_envs) index space."""
        rs = self.role(role)
        return slice(rs.env_start, rs.env_stop)

    # ------------------------------------------------------- resume / init

    def init_from_resume(self, resume_path: str | None) -> None:
        """Load weights from a single checkpoint into all three nets (D8).

        If ``resume_path`` is empty/None, leaves nets at their fresh
        random init from ``make_net()``. If it's a single .pt file, all
        three nets load the same state_dict (they then diverge through
        per-role gradients). Three-file resume (one .pt per role) is
        deferred to T9.
        """
        if not resume_path:
            return
        ckpt = torch.load(resume_path, map_location=self.device)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        for role in ROLES:
            self.roles[role].net.load_state_dict(state_dict)

    # --------------------------------------------------------- T3: sampling

    def sample_opponents(
        self,
        role: str,
        slice_n: int,
        *,
        opponent_metas: list[OpponentMeta],
        opponent_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Per-env opponent assignment for one role's env slice.

        Returns a dict with two int64 tensors of length ``slice_n`` on
        ``self.device``:
          - ``opp_kind``: one of OPP_SELF / OPP_BOT / OPP_SNAPSHOT /
            OPP_LIVE_MAIN (see module-level constants).
          - ``opp_id``: index into ``opponent_metas`` when kind is BOT or
            SNAPSHOT; meaningless (zeroed) otherwise.

        Per-role policy:
          * MAIN: mix self / PFSP-hard over the whole league / PFSP-hard
            over forgotten main_historical, with weights from
            ``cfg.triad_main_mix_{self,pfsp,forgotten}``. Any bucket that
            has no valid opponents falls back to OPP_SELF.
          * ME: 100% live-Main when this role's last evaluated WR vs
            current Main is >= ``cfg.triad_me_curriculum_threshold``;
            otherwise PFSP-VAR (f=p(1-p)) over main_historical
            snapshots (curriculum fallback). Empty historical pool ->
            live-Main fallback.
          * LE: 100% PFSP-HARD over the whole league EXCLUDING
            league_exploiter historicals (LE never trains against itself
            in this role's slice). Empty pool -> OPP_SELF fallback.

        Args:
            role: one of ROLES.
            slice_n: number of envs in this role's slice (== role.slice_n
                in normal use; passed explicitly so this method is pure
                wrt the controller's mutable state).
            opponent_metas: snapshot of the unified opponent pool index.
            opponent_scores: [len(opponent_metas)] float32 bb/hand
                estimates (per learner's view) used for PFSP weighting.
        """
        if role not in self.roles:
            raise KeyError(f"Unknown role: {role!r}. Valid: {ROLES}")
        if slice_n <= 0:
            return {
                "opp_kind": torch.zeros((0,), device=self.device, dtype=torch.int64),
                "opp_id": torch.zeros((0,), device=self.device, dtype=torch.int64),
            }

        if role == ROLE_MAIN:
            return self._sample_opponents_main(slice_n, opponent_metas, opponent_scores)
        if role == ROLE_ME:
            return self._sample_opponents_me(slice_n, opponent_metas, opponent_scores)
        if role == ROLE_LE:
            return self._sample_opponents_le(slice_n, opponent_metas, opponent_scores)
        raise KeyError(f"Unknown role: {role!r}")  # pragma: no cover - guarded above

    # --- per-role sampling implementations ---------------------------------

    def _empty_assignment(self, slice_n: int) -> dict[str, torch.Tensor]:
        return {
            "opp_kind": torch.zeros((slice_n,), device=self.device, dtype=torch.int64),
            "opp_id": torch.zeros((slice_n,), device=self.device, dtype=torch.int64),
        }

    def _pfsp_kwargs(self, mode: str) -> _PFSPKwargs:
        return {
            "mode": mode,
            "temperature": float(self.cfg.league_pfsp_temperature),
            "epsilon": float(self.cfg.league_pfsp_epsilon),
            "q": float(self.cfg.league_pfsp_q),
        }

    def _sample_opponents_main(
        self,
        slice_n: int,
        metas: list[OpponentMeta],
        scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self._empty_assignment(slice_n)
        # Bucket draw: 0=self, 1=pfsp_whole_league, 2=pfsp_forgotten.
        # We sample buckets first, then within each non-self bucket draw
        # the actual snapshot index via PFSP. This keeps the marginal
        # bucket distribution exactly on target even when one of the
        # league filters is empty (those envs fall back to OPP_SELF).
        w_self = float(self.cfg.triad_main_mix_self)
        w_pfsp = float(self.cfg.triad_main_mix_pfsp)
        w_forg = float(self.cfg.triad_main_mix_forgotten)
        bucket_weights = torch.tensor(
            [w_self, w_pfsp, w_forg], device=self.device, dtype=torch.float32
        )
        buckets = torch.multinomial(
            bucket_weights, num_samples=slice_n, replacement=True
        )  # [slice_n]

        # Snapshot pool eligible for M training (excludes ME historicals
        # and bots-unless-cfg-allows).
        allowed_roles = {ROLE_TAG_MAIN_CURRENT, ROLE_TAG_MAIN_HISTORICAL, ROLE_TAG_LE_HISTORICAL}
        whole_league_idx = _snapshot_indices_by_role(metas, allowed_roles)
        forgotten_idx = _forgotten_main_historical(
            metas, _snapshot_indices_by_role(metas, {ROLE_TAG_MAIN_HISTORICAL})
        )

        # Self-play bucket already zero; only need to overwrite buckets 1/2.
        for bucket_id, pool_indices in ((1, whole_league_idx), (2, forgotten_idx)):
            mask = buckets.eq(bucket_id)
            n = int(mask.sum().item())
            if n == 0:
                continue
            picks = _pfsp_sample_indices(
                pool_indices,
                scores,
                n_samples=n,
                device=self.device,
                **self._pfsp_kwargs("hard"),
            )
            if picks is None:
                # Empty pool: leave as OPP_SELF (already 0).
                continue
            out["opp_kind"][mask] = OPP_SNAPSHOT
            out["opp_id"][mask] = picks
        return out

    def _sample_opponents_me(
        self,
        slice_n: int,
        metas: list[OpponentMeta],
        scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self._empty_assignment(slice_n)
        wr = float(self.roles[ROLE_ME].eval_wr_vs_targets.get("main_current", 0.0))
        threshold = float(self.cfg.triad_me_curriculum_threshold)
        if wr >= threshold:
            # Default mode: all envs vs the live Main net.
            out["opp_kind"].fill_(OPP_LIVE_MAIN)
            return out
        # Curriculum fallback: PFSP-VAR over historical mains.
        historical = _snapshot_indices_by_role(metas, {ROLE_TAG_MAIN_HISTORICAL})
        picks = _pfsp_sample_indices(
            historical,
            scores,
            n_samples=slice_n,
            device=self.device,
            **self._pfsp_kwargs("var"),
        )
        if picks is None:
            # No historicals yet -> degenerate to live Main so ME still
            # trains rather than blocking on an empty pool.
            out["opp_kind"].fill_(OPP_LIVE_MAIN)
            return out
        out["opp_kind"].fill_(OPP_SNAPSHOT)
        out["opp_id"] = picks
        return out

    def _sample_opponents_le(
        self,
        slice_n: int,
        metas: list[OpponentMeta],
        scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self._empty_assignment(slice_n)
        # Whole league EXCLUDING LE historicals (LE never trains against
        # its own prior snapshots in its env slice).
        allowed_roles = {
            ROLE_TAG_MAIN_CURRENT,
            ROLE_TAG_MAIN_HISTORICAL,
            ROLE_TAG_ME_HISTORICAL,
        }
        pool_indices = _snapshot_indices_by_role(metas, allowed_roles)
        picks = _pfsp_sample_indices(
            pool_indices,
            scores,
            n_samples=slice_n,
            device=self.device,
            **self._pfsp_kwargs("hard"),
        )
        if picks is None:
            return out  # all OPP_SELF (empty pool)
        out["opp_kind"].fill_(OPP_SNAPSHOT)
        out["opp_id"] = picks
        return out

    def forward_step(self, role: str, **kwargs):  # pragma: no cover - T6
        """Run the role's net on its env slice. Implemented in T6."""
        raise NotImplementedError("forward_step wiring lands in T6")

    # --------------------------------------------------------- T4: PPO update

    def ppo_update_role(
        self,
        role: str,
        *,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        train_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one PPO update over a single role's RolloutBuffer.

        This is the gradient-isolation primitive: the forward, backward,
        and optimizer.step are all scoped to ``role.net`` / ``role.opt``.
        Other roles' parameters are mathematically guaranteed to stay
        bit-identical (no shared params, no shared optimizer state).
        Asserted by ``test_ppo_update_role_changes_only_target_net_params``.

        Args:
            role: one of ROLES.
            returns: [T, slice_n] float32 -- MC or GAE return targets from
                the acting player's perspective.
            advantages: [T, slice_n] float32 -- already-normalized
                advantages (mean 0, std 1) under the same mask the caller
                will pass as ``train_mask``.
            train_mask: optional [T, slice_n] bool -- True where the step
                contributes to the loss (terminating-hand steps in the
                league path). Defaults to all-True.

        Returns: dict of per-update means:
            loss, policy_loss, value_loss, entropy, approx_kl,
            approx_kl_abs, clipfrac, mb_count.
        """
        rs = self.role(role)
        return _ppo_update_one_role(
            rs=rs,
            cfg=self.cfg,
            returns=returns,
            advantages=advantages,
            train_mask=train_mask,
            device=self.device,
            scalar_dim=self.scalar_dim,
        )

    def ppo_update(
        self,
        *,
        returns_by_role: dict[str, torch.Tensor],
        advantages_by_role: dict[str, torch.Tensor],
        train_mask_by_role: dict[str, torch.Tensor | None] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Convenience: call ``ppo_update_role`` sequentially for all 3 roles.

        Each role's update runs to completion before the next starts.
        Gradient isolation is inherent (separate nets/opts), so order
        does not affect correctness; we keep it deterministic (M, ME, LE)
        for reproducible logging.

        Returns ``{role: per-role-metrics-dict}``.
        """
        masks = train_mask_by_role or {}
        out: dict[str, dict[str, float]] = {}
        for role in ROLES:
            out[role] = self.ppo_update_role(
                role,
                returns=returns_by_role[role],
                advantages=advantages_by_role[role],
                train_mask=masks.get(role),
            )
        return out

    # --------------------------------------------------------- T5: snapshots

    def maybe_snapshot(self, update: int, env_steps: int) -> dict[str, SnapshotDecision]:
        """Return per-role snapshot decisions for this update.

        Does NOT write to disk and does NOT mutate any role's net. The
        caller (train.py T6) is responsible for:
          1. Writing the .pt for each role with ``should_snapshot=True``,
             tagging with ``decision.agent_role_tag``.
          2. Calling ``mutate_after_snapshot(role)`` immediately after,
             so the snapshot on disk reflects the PRE-mutation weights.
          3. (Optional) Updating ``last_snapshot_update`` /
             ``snapshot_count`` on the role -- the controller updates
             these inside ``mutate_after_snapshot`` for atomicity.

        Triggers:
          MAIN: every ``triad_main_snapshot_every`` updates
                (``update > 0 and update % K == 0``).
          ME:   ``eval_wr_vs_targets["main_current"] >= triad_me_promote_wr``.
                ``no_eval_data`` if the key is missing.
          LE:   min WR across non-LE targets in ``eval_wr_vs_targets``
                >= ``triad_le_promote_wr``. ``no_eval_data`` if no
                non-LE target evals have been recorded yet.
        """
        env_steps = int(env_steps)  # accept any int-like, normalize
        out: dict[str, SnapshotDecision] = {}
        for role in ROLES:
            out[role] = self._role_snapshot_decision(role, int(update))
        return out

    def mutate_after_snapshot(self, role: str) -> str:
        """Apply the role's post-snapshot mutation in-place.

        Returns the mutation actually applied:
          - "noop"            -- M (mutate_kind="never"), or LE rolled
                                 the dice and stayed (random_with_prob
                                 failed).
          - "reset_to_main"   -- ME copied M's current params.
          - "random_reinit"   -- LE rolled below ``triad_le_reset_prob``
                                 and got a fresh random init.

        Also bumps ``role.snapshot_count`` and updates
        ``last_snapshot_update`` (the caller passes ``update`` separately
        via maybe_snapshot but the *bookkeeping* is centralized here so
        train.py can't forget to do it).
        """
        rs = self.role(role)
        kind = rs.spec.mutate_kind
        applied = "noop"
        if kind == "reset_to_main":
            main_state = self.role(ROLE_MAIN).net.state_dict()
            rs.net.load_state_dict(main_state)
            applied = "reset_to_main"
        elif kind == "random_with_prob":
            roll = float(torch.rand((), device=self.device).item())
            if roll < float(rs.spec.mutate_prob):
                fresh = self._make_net()
                rs.net.load_state_dict(fresh.state_dict())
                applied = "random_reinit"
            else:
                applied = "noop"
        # M's "never" path falls through with applied="noop".
        rs.snapshot_count += 1
        return applied

    def maybe_force_reset_me(self, update: int) -> dict[str, int | str] | None:
        """If ME has been stuck (no snapshot/reset) longer than the timeout,
        copy current Main's weights into ME's net.

        Counters the [[me-exploiter-stalling]] failure mode: once Main
        catches up to ME, ME's WR collapses below ``triad_me_promote_wr``
        and the WR-gated snapshot+mutate path never fires again. ME then
        sits frozen, training but never re-initializing -- burning GPU
        cycles producing nothing new. Per AlphaStar (Vinyals et al. 2019
        §3 / Extended Data Fig. 3), main exploiters reinitialize from the
        main agent every 50-100 PPO updates regardless of WR, to escape
        local optima.

        Disabled when ``cfg.triad_me_force_reset_every <= 0`` (default).

        Does NOT write a snapshot to disk: the post-reset ME is byte-identical
        to current Main, so a .pt of it would just be a redundant copy of
        whatever Main saved most recently. We do bump ``rs.last_snapshot_update``
        so the next ``maybe_force_reset_me`` call measures the cadence
        from this reset (not from the previous WR-gated snapshot).

        Args:
            update: current PPO update index (same value passed to
                ``maybe_snapshot``).

        Returns:
            None when the timeout hasn't elapsed or the feature is disabled.
            Otherwise a small dict with diagnostic fields:
              - ``updates_since``: updates elapsed since ME's last
                snapshot/reset (== threshold at the firing tick).
              - ``kind``: always ``"reset_to_main_forced"`` -- mirrors the
                ``mutate_after_snapshot`` "applied" strings so logging
                downstream can use the same telemetry sink.

        LE intentionally has no equivalent knob: LE's
        ``mutate_kind="random_with_prob"`` is structurally different from
        ME's deterministic reset, and the report notes LE rarely fires
        its gate in practice (a separate fix, not this one).
        """
        every = int(getattr(self.cfg, "triad_me_force_reset_every", 0))
        if every <= 0:
            return None
        rs = self.role(ROLE_ME)
        updates_since = int(update) - int(rs.last_snapshot_update)
        if updates_since < every:
            return None
        main_state = self.role(ROLE_MAIN).net.state_dict()
        rs.net.load_state_dict(main_state)
        rs.last_snapshot_update = int(update)
        return {"updates_since": updates_since, "kind": "reset_to_main_forced"}

    # --- snapshot trigger helpers ------------------------------------------

    def _role_snapshot_decision(self, role: str, update: int) -> SnapshotDecision:
        rs = self.role(role)
        if role == ROLE_MAIN:
            return self._snapshot_decision_main(rs, update)
        if role == ROLE_ME:
            return self._snapshot_decision_me(rs, update)
        if role == ROLE_LE:
            return self._snapshot_decision_le(rs, update)
        raise KeyError(f"Unknown role: {role!r}")  # pragma: no cover - guarded

    def _snapshot_decision_main(self, rs: RoleState, update: int) -> SnapshotDecision:
        k = int(rs.spec.snapshot_every or 0)
        fire = k > 0 and update > 0 and update % k == 0
        return SnapshotDecision(
            role=ROLE_MAIN,
            should_snapshot=bool(fire),
            reason="periodic" if fire else "not_due",
            agent_role_tag=ROLE_TAG_MAIN_HISTORICAL if fire else "",
            mutate_kind=rs.spec.mutate_kind,
        )

    def _snapshot_decision_me(self, rs: RoleState, update: int) -> SnapshotDecision:
        threshold = float(rs.spec.snapshot_wr_threshold or 0.0)
        wr = rs.eval_wr_vs_targets.get("main_current")
        if wr is None:
            return SnapshotDecision(
                role=ROLE_ME,
                should_snapshot=False,
                reason="no_eval_data",
                agent_role_tag="",
                mutate_kind=rs.spec.mutate_kind,
            )
        fire = float(wr) >= threshold
        return SnapshotDecision(
            role=ROLE_ME,
            should_snapshot=bool(fire),
            reason="wr_threshold" if fire else "below_wr_threshold",
            agent_role_tag=ROLE_TAG_ME_HISTORICAL if fire else "",
            mutate_kind=rs.spec.mutate_kind,
        )

    def _snapshot_decision_le(self, rs: RoleState, update: int) -> SnapshotDecision:
        threshold = float(rs.spec.snapshot_wr_threshold or 0.0)
        # LE's snapshot trigger: "beat EVERY non-LE target by >= threshold".
        # Filter out LE-historical targets (those keys start with "le_" by
        # convention written by T7 / async eval). If no non-LE targets have
        # been evaluated yet, defer.
        targets = {k: float(v) for k, v in rs.eval_wr_vs_targets.items() if not k.startswith("le_")}
        if not targets:
            return SnapshotDecision(
                role=ROLE_LE,
                should_snapshot=False,
                reason="no_eval_data",
                agent_role_tag="",
                mutate_kind=rs.spec.mutate_kind,
            )
        worst = min(targets.values())
        fire = worst >= threshold
        return SnapshotDecision(
            role=ROLE_LE,
            should_snapshot=bool(fire),
            reason="wr_threshold" if fire else "below_wr_threshold",
            agent_role_tag=ROLE_TAG_LE_HISTORICAL if fire else "",
            mutate_kind=rs.spec.mutate_kind,
        )

    def schedule_async_eval(self, current_update: int):  # pragma: no cover - T7
        """Async eval cadence for ME and LE. Implemented in T7."""
        raise NotImplementedError("schedule_async_eval lands in T7")
