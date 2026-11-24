"""Single-net self-play and NFSP-lite league PPO runner."""

from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import NUM_RAISE_BUCKETS, raise_bucket_to_amount
from train import bots as train_bots
from train.buffer import RolloutBuffer
from train.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from train.config import TrainConfig, TrainResult
from train.eval import (
    eval_vs_bot,
    eval_vs_bot_seat_swap,
    eval_vs_policy,
    eval_vs_snapshot_mix_seat_swap,
)
from train.league import OpponentMeta, UnifiedOpponentPool, pfsp_weights_from_bb_per_hand
from train.policy_factory import (
    PolicyNet,
    load_snapshot_model,
)
from train.policy_factory import (
    make_policy as _make_policy,
)
from train.policy_factory import (
    push_transformer_token as _push_transformer_token,
)
from train.policy_factory import (
    stage_from_scalars as _stage_from_scalars,
)
from train.ppo_utils import (
    action_stats,
    action_stats_by_street,
    action_stats_masked,
    assert_finite,
    compute_gae,
    compute_mc_returns_p0,
    scatter_player_state,
    select_player_state,
    sync,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ModuleNotFoundError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _HAS_TB = False

try:
    import pynvml

    _HAS_PYNVML = True
except ModuleNotFoundError:
    pynvml = None
    _HAS_PYNVML = False

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_impl(cfg: TrainConfig) -> TrainResult:
    # Hydra changes CWD into the run directory, so repo-relative paths must be
    # normalized before any checkpoint IO.
    cfg_dict = None
    if not os.path.isabs(cfg.checkpoint_dir):
        cfg_dict = asdict(cfg)
        cfg_dict["checkpoint_dir"] = str(_REPO_ROOT / cfg.checkpoint_dir)
    if cfg.resume_from and not os.path.isabs(cfg.resume_from):
        if cfg_dict is None:
            cfg_dict = asdict(cfg)
        cfg_dict["resume_from"] = str(_REPO_ROOT / cfg.resume_from)
    if cfg_dict is not None:
        cfg = TrainConfig(**cfg_dict)

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    env = WarpPokerEnv(
        num_envs=cfg.num_envs,
        starting_stack=int(cfg.starting_stack),
        small_blind=int(cfg.small_blind),
        big_blind=int(cfg.big_blind),
        device=cfg.device,
    )
    # Randomized per-env starting stack: teaches the policy stack-depth-aware
    # play (40 BB push/fold all the way up to 200 BB 3-bet wars) instead of
    # overfitting to one specific depth. Fixed stack if min == max.
    if int(cfg.starting_stack_min) < int(cfg.starting_stack_max):
        per_env_stack = torch.randint(
            int(cfg.starting_stack_min),
            int(cfg.starting_stack_max) + 1,
            (cfg.num_envs,),
            device=device,
            dtype=torch.int32,
        )
        env.reset_with_config(starting_stack=per_env_stack)
        print(
            f"[env] randomized starting_stack in "
            f"[{int(cfg.starting_stack_min)}, {int(cfg.starting_stack_max)}] chips "
            f"({int(cfg.starting_stack_min) // int(cfg.big_blind)}-"
            f"{int(cfg.starting_stack_max) // int(cfg.big_blind)} BB)"
        )
    obs = env.reset()

    scalar_dim = int(obs["scalars"].shape[1])
    net = _make_policy(cfg, scalar_dim, device)

    if cfg.compile_policy:
        net = torch.compile(net, mode="reduce-overhead")  # type: ignore[assignment]

    # Use fused optimizer for better performance (single kernel for entire update).
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, fused=True)

    # Automatic mixed precision (AMP) setup.
    # BF16 is preferred (same exponent range as FP32, no loss scaling) but is only
    # natively supported on Ampere+ (sm_80+). "auto" picks bf16 if hardware supports
    # it, otherwise fp16 — important for V100 (sm_70) where bf16 has no hw acceleration.
    amp_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if cfg.amp_dtype == "auto":
        # Native bf16 requires compute capability >= 8.0 (Ampere). PyTorch's
        # is_bf16_supported() returns True even on V100 (sm_70) because it can be
        # emulated via fp32, but there's no perf benefit — and it can be slower
        # than fp16 since fp16 has Tensor Core acceleration on V100.
        cap = torch.cuda.get_device_capability() if cfg.device.startswith("cuda") else (0, 0)
        if cap >= (8, 0):
            amp_dtype = torch.bfloat16
            print(f"[amp] auto -> bfloat16 (sm_{cap[0]}{cap[1]}, native hardware support)")
        else:
            amp_dtype = torch.float16
            print(f"[amp] auto -> float16 (sm_{cap[0]}{cap[1]}, bf16 not hw-accelerated)")
    else:
        amp_dtype = amp_dtype_map.get(cfg.amp_dtype, torch.bfloat16)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")

    # GradScaler is REQUIRED for fp16 to prevent gradient underflow (Micikevicius
    # et al., "Mixed Precision Training", 2017). bf16 has the same exponent range
    # as fp32 so no scaling is needed — enabling the scaler for bf16 would be a
    # no-op pass-through but we explicitly disable it for clarity.
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=scaler_enabled)
    if scaler_enabled:
        print("[amp] GradScaler enabled (fp16 needs loss scaling to avoid underflow)")

    # League (NFSP-lite): sample opponents per-env at hand boundaries.
    # Encoding:
    #   opp_kind: 0=self-play (opponent==learner), 1=bot, 2=snapshot
    #   opp_id:   unified-pool index for bot/snapshot opponents
    bot_names = train_bots.available_bot_names()
    opp_kind = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    opp_id = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    learner_seat = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)

    pool = None
    opponent_metas: list[OpponentMeta] = []
    opponent_scores = torch.empty((0,), device=device, dtype=torch.float32)
    opponent_kinds = torch.empty((0,), device=device, dtype=torch.int64)
    opponent_filled = 0
    snapshot_models: dict[str, PolicyNet] = {}

    if cfg.league_enabled:
        pool_dir = Path(cfg.checkpoint_dir) / "league_pool"
        pool = UnifiedOpponentPool(
            pool_dir=pool_dir,
            bot_names=bot_names,
        )

        # In-memory opponent tracking
        max_opponents = len(bot_names) + cfg.league_snapshot_pool
        opponent_metas = pool.load_index()
        opponent_scores = torch.zeros(max_opponents, device=device)
        opponent_kinds = torch.zeros(
            max_opponents, device=device, dtype=torch.int64
        )  # 1=bot, 2=snapshot

        for i, meta in enumerate(opponent_metas[:max_opponents]):
            opponent_scores[i] = (
                float(meta.bb_per_hand)
                if meta.bb_per_hand is not None
                else float(cfg.league_bot_initial_score)
            )
            opponent_kinds[i] = 1 if meta.kind == "bot" else 2

        opponent_filled = min(len(opponent_metas), max_opponents)
    snapshot_adds = 0

    def resample_roles(mask: torch.Tensor) -> None:
        idx = torch.nonzero(mask.to(dtype=torch.bool), as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        num = int(idx.numel())

        learner_seat[idx] = torch.randint(0, 2, (num,), device=device, dtype=torch.int64)
        u = torch.rand((num,), device=device, dtype=torch.float32)

        is_selfplay = u < float(cfg.league_eta_selfplay)
        want_opponent = ~is_selfplay

        opp_kind[idx[is_selfplay]] = 0
        opp_id[idx[is_selfplay]] = 0

        # Opponent from unified pool (PFSP-weighted)
        if want_opponent.any():
            if opponent_filled > 0:
                # Include main historical agents and exploiters. Scripted bots
                # are only included if `league_train_on_bots=true` (false by
                # default — bots are eval-only per the poker-AI SOTA recipes).
                valid_indices = [
                    i
                    for i, meta in enumerate(opponent_metas[:opponent_filled])
                    if meta.agent_role in {"main", "main_historical", "league_exploiter"}
                    or (meta.kind == "bot" and cfg.league_train_on_bots)
                ]
                snapshot_indices = [
                    i for i in valid_indices if opponent_metas[i].kind == "snapshot"
                ]
                max_rollout_snapshots = int(cfg.league_rollout_snapshot_k)
                if max_rollout_snapshots > 0 and len(snapshot_indices) > max_rollout_snapshots:
                    snap_t = torch.tensor(snapshot_indices, device=device, dtype=torch.int64)
                    snap_weights = pfsp_weights_from_bb_per_hand(
                        opponent_scores[snap_t],
                        temperature=cfg.league_pfsp_temperature,
                        epsilon=cfg.league_pfsp_epsilon,
                        mode=cfg.league_pfsp_mode,
                        q=cfg.league_pfsp_q,
                    )
                    snap_weights = snap_weights / snap_weights.sum().clamp_min(1e-12)
                    picked = torch.multinomial(
                        snap_weights,
                        num_samples=max_rollout_snapshots,
                        replacement=False,
                    )
                    picked_snapshots = set(snap_t[picked].tolist())
                    valid_indices = [
                        i
                        for i in valid_indices
                        if opponent_metas[i].kind == "bot" or i in picked_snapshots
                    ]

                if valid_indices:
                    indices_t = torch.tensor(valid_indices, device=device, dtype=torch.int64)
                    weights = pfsp_weights_from_bb_per_hand(
                        opponent_scores[indices_t],
                        temperature=cfg.league_pfsp_temperature,
                        epsilon=cfg.league_pfsp_epsilon,
                        mode=cfg.league_pfsp_mode,
                        q=cfg.league_pfsp_q,
                    )
                    weights = weights / weights.sum().clamp_min(1e-12)

                    n_samples = int(want_opponent.sum().item())
                    choice_idx = torch.multinomial(weights, num_samples=n_samples, replacement=True)
                    choice = indices_t[choice_idx]

                    opp_id[idx[want_opponent]] = choice
                    opp_kind[idx[want_opponent]] = opponent_kinds[choice]
                else:
                    # Fallback to bots/self-play if no specific roles found
                    opp_kind[idx[want_opponent]] = 0
                    opp_id[idx[want_opponent]] = 0
            else:
                # Fallback to self-play if pool is empty (shouldn't happen if bots exist)
                opp_kind[idx[want_opponent]] = 0
                opp_id[idx[want_opponent]] = 0

        # opp_kind/opp_id were updated in place via slicing above.

    def _pick_snapshot_eval_ids(*, subset: int, k_recent: int = 4) -> list[int]:
        # Filter for snapshots only
        snap_indices = [
            i for i, m in enumerate(opponent_metas[:opponent_filled]) if m.kind == "snapshot"
        ]

        if not snap_indices:
            return []

        subset = int(subset)
        if subset <= 0 or subset >= len(snap_indices):
            return snap_indices

        sorted_snaps = sorted(
            snap_indices,
            key=lambda i: (opponent_metas[i].env_steps or 0, opponent_metas[i].update or 0),
        )

        k_recent = int(min(k_recent, subset, len(snap_indices)))
        recent = sorted_snaps[-k_recent:]
        recent_set = set(recent)

        remaining = subset - k_recent
        if remaining <= 0:
            return recent

        cand = torch.tensor(
            [i for i in sorted_snaps if i not in recent_set],
            device=device,
            dtype=torch.int64,
        )
        if cand.numel() == 0:
            return recent

        r = min(remaining, int(cand.numel()))
        pick = cand[torch.randperm(int(cand.numel()), device=device)[:r]].tolist()
        return recent + pick

    if cfg.league_enabled:
        resample_roles(torch.ones((cfg.num_envs,), device=device, dtype=torch.bool))

    # Optional resume: loads model/optimizer and restores counters/RNG.
    update = 0
    ckpt_state = None
    if cfg.resume_from:
        ckpt_state = load_checkpoint(
            path=cfg.resume_from,
            model=net,
            optimizer=opt,
            device=device,
        )
        update = ckpt_state.update

    writer = None
    if _HAS_TB:
        tb_dir = os.environ.get("POKERGPU_TRAIN_TB_DIR", "train/tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)

    buf = RolloutBuffer(
        t=cfg.rollout_steps,
        n=cfg.num_envs,
        scalar_dim=scalar_dim,
        hidden=net.lstm_hidden,
        device=device,
        dtype=amp_dtype if amp_enabled else torch.float32,
    )

    # Per-env per-player state: [1, N, H] - use buffer dtype for efficiency
    state_dtype = buf.dtype
    h0 = torch.zeros((1, cfg.num_envs, net.lstm_hidden), device=device, dtype=state_dtype)
    c0 = torch.zeros((1, cfg.num_envs, net.lstm_hidden), device=device, dtype=state_dtype)
    h1 = torch.zeros((1, cfg.num_envs, net.lstm_hidden), device=device, dtype=state_dtype)
    c1 = torch.zeros((1, cfg.num_envs, net.lstm_hidden), device=device, dtype=state_dtype)
    # Snapshot opponents are evaluated as memoryless during rollouts; feed constant zero state
    # to avoid per-step allocations.
    snap_h0 = torch.zeros_like(h0)
    snap_c0 = torch.zeros_like(c0)

    invalid_accum = torch.zeros((), device=device, dtype=torch.int64)

    t0 = time.perf_counter()
    # Accumulate on-device to avoid sync-heavy `.item()` inside the PPO loop.
    loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    kl_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    clipfrac_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    value_loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    policy_loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    entropy_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    opt_steps = 0
    env_steps = 0
    eval_metrics: dict[str, float] = {}
    eval_env: WarpPokerEnv | None = None

    def get_eval_env() -> WarpPokerEnv:
        nonlocal eval_env
        if eval_env is None:
            eval_env = WarpPokerEnv(num_envs=min(1024, cfg.num_envs), device=cfg.device)
        return eval_env

    # If we resumed, start counters from the checkpoint so:
    # - TensorBoard steps keep increasing (we use env_steps as the TB step axis)
    # - tqdm percent/progress starts where we left off
    if ckpt_state is not None:
        env_steps = int(ckpt_state.env_steps)
        opt_steps = int(ckpt_state.opt_steps)

    disable_pbar = os.environ.get("POKERGPU_TRAIN_TQDM", "1") == "0" or not sys.stderr.isatty()
    env_steps_per_rollout = int(cfg.num_envs) * int(cfg.rollout_steps)
    max_updates = cfg.num_updates if cfg.total_env_steps <= 0 else 0
    target_env_steps = cfg.total_env_steps if cfg.total_env_steps > 0 else 0
    pbar_total = (
        target_env_steps
        if target_env_steps > 0
        else (max_updates * env_steps_per_rollout if max_updates > 0 else 0)
    )
    pbar = tqdm(
        total=pbar_total if pbar_total > 0 else None,
        initial=env_steps if pbar_total > 0 else 0,
        disable=disable_pbar,
        desc="train",
        unit="env_step",
        unit_scale=True,
        unit_divisor=1000,
        mininterval=max(0.1, float(cfg.log_every_seconds)),
    )

    log_stride = cfg.log_every_env_steps if cfg.log_every_env_steps > 0 else env_steps_per_rollout
    next_log_env_steps = env_steps + log_stride
    next_log_time = time.perf_counter() + max(0.1, float(cfg.log_every_seconds))
    last_loss_u = float("nan")
    last_kl_u = float("nan")
    last_clip_u = float("nan")
    last_ent_u = float("nan")
    last_vloss_u = float("nan")
    last_ev = float("nan")

    def maybe_refresh(phase: str) -> None:
        nonlocal next_log_time
        if disable_pbar:
            return
        now = time.perf_counter()
        if now < next_log_time:
            return
        overall_sps_now = float(env_steps) / max(1e-9, now - t0)
        pbar.set_postfix(
            phase=phase,
            upd=update,
            all=f"{overall_sps_now / 1000:.0f}k",
            loss=f"{last_loss_u:.2f}" if last_loss_u == last_loss_u else "-",
            kl=f"{last_kl_u:.4f}" if last_kl_u == last_kl_u else "-",
            clip=f"{last_clip_u:.3f}" if last_clip_u == last_clip_u else "-",
            ent=f"{last_ent_u:.3f}" if last_ent_u == last_ent_u else "-",
            vloss=f"{last_vloss_u:.3f}" if last_vloss_u == last_vloss_u else "-",
            ev=f"{last_ev:.3f}" if last_ev == last_ev else "-",
            eval=f"{eval_metrics.get('bb_per_hand', 0.0):.3f}",
        )
        pbar.refresh()
        next_log_time = now + max(0.1, float(cfg.log_every_seconds))

    next_save_env_steps = (
        env_steps + cfg.save_every_env_steps if cfg.save_every_env_steps > 0 else 0
    )

    while True:
        if max_updates > 0 and update >= max_updates:
            break
        if target_env_steps > 0 and env_steps >= target_env_steps:
            break

        update_t0 = time.perf_counter()
        t_collect_s = 0.0
        t_opt_s = 0.0
        t_eval_s = 0.0
        t_league_eval_s = 0.0
        t_selfplay_eval_s = 0.0
        t_misc_s = 0.0
        mb_count = 0
        loss_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        kl_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        kl_abs_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        clipfrac_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        value_loss_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        policy_loss_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        entropy_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        ev_rollout = 0.0
        # Diagnostics (counted over epoch 0 only, once per sample).
        diag_total_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_old_nonfinite_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_new_nonfinite_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_log_ratio_nonfinite_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_log_ratio_abs_sum_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_log_ratio_abs_max_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_old_lp_min_t = torch.full((), float("inf"), device=device, dtype=torch.float32)
        diag_old_lp_max_t = torch.full((), float("-inf"), device=device, dtype=torch.float32)
        diag_new_lp_min_t = torch.full((), float("inf"), device=device, dtype=torch.float32)
        diag_new_lp_max_t = torch.full((), float("-inf"), device=device, dtype=torch.float32)
        diag_log_ratio_min_t = torch.full((), float("inf"), device=device, dtype=torch.float32)
        diag_log_ratio_max_t = torch.full((), float("-inf"), device=device, dtype=torch.float32)

        if cfg.timing_sync:
            sync(cfg.device)
        t_collect0 = time.perf_counter()
        with torch.no_grad():
            for t in range(cfg.rollout_steps):
                # If env just terminated on the previous step, reset both players.
                reset = obs["terminated"].to(dtype=torch.bool)
                if reset.any():
                    z = torch.zeros_like(h0)
                    h0 = torch.where(reset.view(1, -1, 1), z, h0)
                    c0 = torch.where(reset.view(1, -1, 1), z, c0)
                    h1 = torch.where(reset.view(1, -1, 1), z, h1)
                    c1 = torch.where(reset.view(1, -1, 1), z, c1)
                    if cfg.league_enabled:
                        resample_roles(reset)

                pid = obs["player_id"].to(dtype=torch.int64)
                ep_id = obs["episode_id"].to(dtype=torch.int32)

                h_batch, c_batch = select_player_state(player_id=pid, h0=h0, c0=c0, h1=h1, c1=c1)

                # IMPORTANT: `obs` tensors are zero-copy views into env-owned buffers that
                # get overwritten in-place by `env.step()`. Snapshot the current observation
                # into the rollout buffer BEFORE stepping to keep (s_t, a_t, logp_t, mask_t)
                # aligned for PPO.
                buf.cards[t].copy_(obs["cards"])
                buf.scalars[t].copy_(obs["scalars"])
                buf.action_mask[t].copy_(obs["action_mask"])
                buf.player_id[t].copy_(pid)
                buf.episode_id[t].copy_(ep_id)
                buf.reset_mask[t].copy_(reset)
                buf.h[t].copy_(h_batch.squeeze(0))
                buf.c[t].copy_(c_batch.squeeze(0))

                if not cfg.league_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out = net.forward_step(
                            cards=obs["cards"],
                            scalars=obs["scalars"],
                            action_mask=obs["action_mask"],
                            h=h_batch,
                            c=c_batch,
                            terminated=None,
                            deterministic=False,
                        )

                    # Update the acting player's hidden state.
                    h0, c0, h1, c1 = scatter_player_state(
                        player_id=pid,
                        h_new=out.h,
                        c_new=out.c,
                        h0=h0,
                        c0=c0,
                        h1=h1,
                        c1=c1,
                    )

                    amounts = raise_bucket_to_amount(
                        raise_bucket=out.raise_bucket,
                        pot_total=obs["pot_total"],
                        min_raise=obs["min_raise"],
                        max_raise=obs["max_raise"],
                    )
                    amounts = torch.where(
                        out.action_type.eq(c.ACTION_RAISE),
                        amounts,
                        torch.zeros_like(amounts),
                    )
                    next_obs = env.step(out.action_type.to(dtype=torch.int32), amounts)

                    if cfg.policy_backbone == "transformer":
                        stage_at_action = _stage_from_scalars(buf.scalars[t])
                        h0, h1 = _push_transformer_token(
                            h0,
                            h1,
                            player_id=pid,
                            action_type=out.action_type.to(dtype=torch.int32),
                            raise_bucket=out.raise_bucket,
                            stage=stage_at_action,
                        )

                    bb = next_obs["big_blind"].to(dtype=torch.float32).clamp_min(1.0)
                    # Reward is always returned as P0 reward; map to acting player perspective.
                    reward_act = torch.where(
                        pid.eq(0), next_obs["rewards"], -next_obs["rewards"]
                    ).to(dtype=torch.float32)
                    reward_act = reward_act / (bb * float(c.DEFAULT_STACK_BB))

                    buf.learner_mask[t].fill_(True)
                    buf.action_type[t].copy_(out.action_type)
                    buf.raise_bucket[t].copy_(out.raise_bucket)
                    buf.logprob[t].copy_(out.logprob)
                    buf.value[t].copy_(out.value)
                    buf.reward[t].copy_(reward_act)
                    buf.reward_p0[t].copy_(
                        torch.where(
                            next_obs["terminated"].to(dtype=torch.bool),
                            next_obs["rewards"].to(dtype=torch.float32)
                            / (bb * float(c.DEFAULT_STACK_BB)),
                            torch.zeros_like(reward_act),
                        )
                    )
                    buf.done[t].copy_(next_obs["terminated"])

                    invalid_accum = invalid_accum + next_obs["invalid_action"].sum().to(
                        dtype=torch.int64
                    )

                    obs = next_obs
                    maybe_refresh("collect")
                    continue

                # League-enabled: action selection depends on per-env opponent assignment.
                is_selfplay = opp_kind.eq(0)
                is_learner_actor = is_selfplay | pid.eq(learner_seat)
                buf.learner_mask[t].copy_(is_learner_actor)

                action_type = torch.full(
                    (cfg.num_envs,), c.ACTION_FOLD, device=device, dtype=torch.int32
                )
                # Per-step staging tensors for the chosen action + ratio-sensitive
                # log-prob / value (the rollout buffer keeps these in fp32 to
                # avoid mixed-precision quantization breaking the PPO ratio).
                raise_bucket = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
                logprob = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)
                value = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)

                if is_learner_actor.any():
                    idx_l = torch.nonzero(is_learner_actor, as_tuple=False).squeeze(-1)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out_l = net.forward_step(
                            cards=obs["cards"][idx_l],
                            scalars=obs["scalars"][idx_l],
                            action_mask=obs["action_mask"][idx_l],
                            h=h_batch[:, idx_l, :],
                            c=c_batch[:, idx_l, :],
                            terminated=None,
                            deterministic=False,
                        )
                    action_type[idx_l] = out_l.action_type.to(dtype=torch.int32)
                    raise_bucket[idx_l] = out_l.raise_bucket.to(dtype=raise_bucket.dtype)
                    logprob[idx_l] = out_l.logprob.to(dtype=logprob.dtype)
                    value[idx_l] = out_l.value.to(dtype=value.dtype)

                    pid_l = pid[idx_l].to(dtype=torch.bool)
                    if (~pid_l).any():
                        idx0 = idx_l[~pid_l]
                        h0[:, idx0, :] = out_l.h[:, ~pid_l, :]
                        c0[:, idx0, :] = out_l.c[:, ~pid_l, :]
                    if pid_l.any():
                        idx1 = idx_l[pid_l]
                        h1[:, idx1, :] = out_l.h[:, pid_l, :]
                        c1[:, idx1, :] = out_l.c[:, pid_l, :]

                is_opponent_actor = (~is_learner_actor) & (~is_selfplay)
                bot_mask = is_opponent_actor & opp_kind.eq(1)
                if bot_mask.any():
                    for pool_idx in torch.unique(opp_id[bot_mask]).tolist():
                        pool_idx = int(pool_idx)
                        meta = opponent_metas[pool_idx]
                        if meta.kind != "bot":  # Should be guaranteed by opp_kind logic
                            continue
                        bot_name = meta.id

                        m = bot_mask & opp_id.eq(pool_idx)
                        if not m.any():
                            continue
                        idx_b = torch.nonzero(m, as_tuple=False).squeeze(-1)
                        bot_fn = train_bots.get_bot_fn(bot_name)
                        a_b, rf_b = bot_fn(
                            {
                                "action_mask": obs["action_mask"][idx_b],
                                "min_raise": obs["min_raise"][idx_b],
                                "max_raise": obs["max_raise"][idx_b],
                            }
                        )
                        action_type[idx_b] = a_b
                        # Bots return a continuous fraction in [0, 1]; coarse-map
                        # to a bucket index for the discrete action space. (This
                        # path is normally inactive — bots are eval-only when
                        # cfg.league_train_on_bots=false.)
                        bucket_b = (
                            (rf_b.to(dtype=torch.float32) * (NUM_RAISE_BUCKETS - 1))
                            .round()
                            .clamp_(0, NUM_RAISE_BUCKETS - 1)
                            .to(dtype=torch.int64)
                        )
                        raise_bucket[idx_b] = bucket_b

                snap_mask = is_opponent_actor & opp_kind.eq(2)
                if snap_mask.any():
                    for pool_idx in torch.unique(opp_id[snap_mask]).tolist():
                        pool_idx = int(pool_idx)
                        meta = opponent_metas[pool_idx]
                        if meta.kind != "snapshot":
                            continue
                        snap_path = os.path.join(cfg.checkpoint_dir, meta.id)

                        if snap_path not in snapshot_models:
                            snapshot_models[snap_path] = load_snapshot_model(
                                snap_path, device, cfg, scalar_dim
                            )

                        snap_net = snapshot_models[snap_path]

                        m = snap_mask & opp_id.eq(pool_idx)
                        if not m.any():
                            continue
                        idx_s = torch.nonzero(m, as_tuple=False).squeeze(-1)

                        with torch.autocast(
                            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
                        ):
                            out_s = snap_net.forward_step(
                                cards=obs["cards"][idx_s],
                                scalars=obs["scalars"][idx_s],
                                action_mask=obs["action_mask"][idx_s],
                                h=snap_h0[:, idx_s, :],
                                c=snap_c0[:, idx_s, :],
                                terminated=None,
                                deterministic=True,
                            )
                        action_type[idx_s] = out_s.action_type.to(dtype=torch.int32)
                        raise_bucket[idx_s] = out_s.raise_bucket.to(dtype=raise_bucket.dtype)

                amounts = raise_bucket_to_amount(
                    raise_bucket=raise_bucket,
                    pot_total=obs["pot_total"],
                    min_raise=obs["min_raise"],
                    max_raise=obs["max_raise"],
                )
                amounts = torch.where(
                    action_type.eq(c.ACTION_RAISE),
                    amounts,
                    torch.zeros_like(amounts),
                )
                next_obs = env.step(action_type, amounts)

                if cfg.policy_backbone == "transformer":
                    stage_at_action = _stage_from_scalars(buf.scalars[t])
                    h0, h1 = _push_transformer_token(
                        h0,
                        h1,
                        player_id=pid,
                        action_type=action_type,
                        raise_bucket=raise_bucket,
                        stage=stage_at_action,
                    )

                bb = next_obs["big_blind"].to(dtype=torch.float32).clamp_min(1.0)
                reward_p0 = next_obs["rewards"].to(dtype=torch.float32) / (
                    bb * float(c.DEFAULT_STACK_BB)
                )
                reward_p0 = torch.where(
                    next_obs["terminated"].to(dtype=torch.bool),
                    reward_p0,
                    torch.zeros_like(reward_p0),
                )

                buf.action_type[t].copy_(action_type.to(dtype=torch.int64))
                buf.raise_bucket[t].copy_(raise_bucket)
                buf.logprob[t].copy_(logprob)
                buf.value[t].copy_(value)
                buf.reward[t].zero_()
                buf.reward_p0[t].copy_(reward_p0)
                buf.done[t].copy_(next_obs["terminated"])

                invalid_accum = invalid_accum + next_obs["invalid_action"].sum().to(
                    dtype=torch.int64
                )

                obs = next_obs
                maybe_refresh("collect")

            # Bootstrap value at the end of rollout from the current acting player.
            if not cfg.league_enabled:
                pid_last = obs["player_id"].to(dtype=torch.int64)
                h_last, c_last = select_player_state(player_id=pid_last, h0=h0, c0=c0, h1=h1, c1=c1)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out_last = net.forward_step(
                        cards=obs["cards"],
                        scalars=obs["scalars"],
                        action_mask=obs["action_mask"],
                        h=h_last,
                        c=c_last,
                        terminated=None,
                        deterministic=True,
                    )
                last_value = out_last.value
                last_player_id = pid_last
            else:
                # Bootstrap value path mirrors buf.value dtype (fp32) so the
                # advantage computation has no implicit downcast.
                last_value = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)
                last_player_id = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)

        # Advance progress as soon as we have collected the rollout, before PPO updates.
        env_steps += env_steps_per_rollout
        pbar.update(env_steps_per_rollout)
        if cfg.timing_sync:
            sync(cfg.device)
        t_collect_s = time.perf_counter() - t_collect0

        # Determine whether we will log this update BEFORE running PPO.
        do_log = env_steps >= next_log_env_steps

        if cfg.timing_sync:
            sync(cfg.device)
        t_opt0 = time.perf_counter()

        if not cfg.league_enabled:
            advantages, returns = compute_gae(
                rewards=buf.reward,
                dones=buf.done,
                values=buf.value,
                last_value=last_value,
                player_id=buf.player_id,
                last_player_id=last_player_id,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            adv = advantages.reshape(-1)
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)

            # Flatten rollout for minibatching.
            cards_f = buf.cards.reshape(-1, 7)
            scalars_f = buf.scalars.reshape(-1, scalar_dim)
            mask_f = buf.action_mask.reshape(-1, c.NUM_ACTIONS)
            action_type_f = buf.action_type.reshape(-1)
            raise_bucket_f = buf.raise_bucket.reshape(-1)
            old_logprob_f = buf.logprob.reshape(-1)
            returns_f = returns.reshape(-1)
            values_f = buf.value.reshape(-1)
            h_f = buf.h.reshape(-1, net.lstm_hidden)
            c_f = buf.c.reshape(-1, net.lstm_hidden)

            # Explained variance (critic health signal), computed on rollout-time values.
            var_v = returns_f.var(unbiased=False)
            if float(var_v.item()) < 1e-12:
                ev = torch.zeros((), device=device, dtype=torch.float32)
            else:
                ev = 1.0 - (returns_f - values_f).var(unbiased=False) / var_v
            ev_rollout = float(ev.detach().item())
            act_stats = action_stats(buf.action_type, buf.raise_bucket)
            street_stats = action_stats_by_street(
                action_type=buf.action_type,
                raise_bucket=buf.raise_bucket,
                scalars=buf.scalars,
                base_mask=torch.ones_like(buf.done, dtype=torch.bool),
            )

            total = int(cards_f.shape[0])
            train_mask_f = None
            total_effective = float(total)
        else:
            returns_p0, valid_p0 = compute_mc_returns_p0(reward_p0=buf.reward_p0, dones=buf.done)
            returns_act = torch.where(buf.player_id.eq(0), returns_p0, -returns_p0)
            train_mask_f = (valid_p0 & buf.learner_mask).reshape(-1)
            if not bool(train_mask_f.any().item()):
                maybe_refresh("idle")
                update += 1
                continue

            # League training: keep the full rollout flattened and use `train_mask_f`
            # to zero-weight opponent steps and episodes that don't terminate within
            # the rollout window (no huge gathered tensors).
            cards_f = buf.cards.reshape(-1, 7)
            scalars_f = buf.scalars.reshape(-1, scalar_dim)
            mask_f = buf.action_mask.reshape(-1, c.NUM_ACTIONS)
            action_type_f = buf.action_type.reshape(-1)
            raise_bucket_f = buf.raise_bucket.reshape(-1)
            old_logprob_f = buf.logprob.reshape(-1)
            values_f = buf.value.reshape(-1)
            h_f = buf.h.reshape(-1, net.lstm_hidden)
            c_f = buf.c.reshape(-1, net.lstm_hidden)
            returns_f = returns_act.reshape(-1)

            w_all = train_mask_f.to(dtype=torch.float32)
            denom_all = w_all.sum().clamp_min(1.0)
            adv_raw = (returns_f - values_f).detach()
            adv_mean = (adv_raw * w_all).sum() / denom_all
            adv_var = ((adv_raw - adv_mean).pow(2) * w_all).sum() / denom_all
            adv_std = adv_var.sqrt().clamp_min(1e-6)
            adv = (adv_raw - adv_mean) / adv_std

            v_mean = (returns_f * w_all).sum() / denom_all
            var_v = ((returns_f - v_mean).pow(2) * w_all).sum() / denom_all
            if float(var_v.item()) < 1e-12:
                ev = torch.zeros((), device=device, dtype=torch.float32)
            else:
                resid_var = ((returns_f - values_f).pow(2) * w_all).sum() / denom_all
                ev = 1.0 - resid_var / var_v
            ev_rollout = float(ev.detach().item())
            act_stats = action_stats_masked(buf.action_type, buf.raise_bucket, buf.learner_mask)
            street_stats = action_stats_by_street(
                action_type=buf.action_type,
                raise_bucket=buf.raise_bucket,
                scalars=buf.scalars,
                base_mask=buf.learner_mask,
            )

            total = int(cards_f.shape[0])
            total_effective = float(denom_all.detach().item())

        # NOTE: `torch.randperm(total, device=cuda)` becomes very expensive at large totals
        # (e.g. millions of samples) and can look like the run is "stuck".
        # Instead of a full permutation, we use a cheap per-epoch offset and iterate
        # contiguous minibatches. This is sufficient for PPO diagnostics and avoids
        # multi-minute stalls on giant batches.

        # Iterate contiguous minibatches (views) to avoid per-minibatch index tensor
        # allocations and expensive advanced indexing gathers.
        mb_size = int(cfg.minibatch_size)
        num_mbs = (total + mb_size - 1) // max(1, mb_size)
        for _epoch in range(cfg.num_epochs):
            offset_mb = (_epoch * 104729) % max(1, num_mbs)  # 104729 is a prime
            for mb_i in range(num_mbs):
                j = (mb_i + offset_mb) % num_mbs
                start = j * mb_size
                end = min(total, start + mb_size)
                if end <= start:
                    continue
                sl = slice(start, end)
                mb_count += 1

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    eval_out = net.evaluate_step(
                        cards=cards_f[sl],
                        scalars=scalars_f[sl],
                        action_mask=mask_f[sl],
                        action_type=action_type_f[sl],
                        raise_bucket=raise_bucket_f[sl],
                        h=h_f[sl].unsqueeze(0),
                        c=c_f[sl].unsqueeze(0),
                        terminated=None,
                    )

                assert_finite("old_logprob", old_logprob_f[sl])
                assert_finite("new_logprob", eval_out.logprob)

                # Guard against exp overflow (which can otherwise turn PPO into NaNs/inf).
                log_ratio_raw = eval_out.logprob - old_logprob_f[sl]
                log_ratio = log_ratio_raw.clamp(min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                # Diagnostics: only compute when we are going to log, and keep everything
                # on-device to avoid per-minibatch `.item()` sync storms.
                if do_log and _epoch == 0:
                    diag_total_t = diag_total_t + torch.tensor(
                        float(end - start), device=device, dtype=torch.float32
                    )
                    old_lp = old_logprob_f[sl]
                    new_lp = eval_out.logprob
                    diag_old_nonfinite_t = diag_old_nonfinite_t + (
                        (~torch.isfinite(old_lp)).to(dtype=torch.float32).sum()
                    )
                    diag_new_nonfinite_t = diag_new_nonfinite_t + (
                        (~torch.isfinite(new_lp)).to(dtype=torch.float32).sum()
                    )
                    diag_log_ratio_nonfinite_t = diag_log_ratio_nonfinite_t + (
                        (~torch.isfinite(log_ratio_raw)).to(dtype=torch.float32).sum()
                    )
                    diag_old_lp_min_t = torch.minimum(diag_old_lp_min_t, old_lp.min())
                    diag_old_lp_max_t = torch.maximum(diag_old_lp_max_t, old_lp.max())
                    diag_new_lp_min_t = torch.minimum(diag_new_lp_min_t, new_lp.min())
                    diag_new_lp_max_t = torch.maximum(diag_new_lp_max_t, new_lp.max())
                    diag_log_ratio_min_t = torch.minimum(diag_log_ratio_min_t, log_ratio_raw.min())
                    diag_log_ratio_max_t = torch.maximum(diag_log_ratio_max_t, log_ratio_raw.max())
                    diag_log_ratio_abs_sum_t = diag_log_ratio_abs_sum_t + log_ratio_raw.abs().sum()
                    diag_log_ratio_abs_max_t = torch.maximum(
                        diag_log_ratio_abs_max_t, log_ratio_raw.abs().max()
                    )
                unclipped = ratio * adv[sl]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv[sl]
                if cfg.league_enabled:
                    w = train_mask_f[sl].to(dtype=torch.float32)
                    denom = w.sum().clamp_min(1.0)
                    policy_loss = -(w * torch.minimum(unclipped, clipped)).sum() / denom
                    value_loss = 0.5 * (w * (eval_out.value - returns_f[sl]).pow(2)).sum() / denom
                    entropy_bonus = (w * eval_out.entropy).sum() / denom
                else:
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (eval_out.value - returns_f[sl]).pow(2).mean()
                    entropy_bonus = eval_out.entropy.mean()

                # Note: this common PPO "approx KL" can be near-zero due to sign cancellation.
                # We also log an absolute variant for a more reliable magnitude signal.
                logp_diff = -log_ratio_raw  # old - new
                finite = torch.isfinite(logp_diff)
                if cfg.league_enabled:
                    m = train_mask_f[sl] & finite
                    count = m.to(dtype=torch.float32).sum().clamp_min(1.0)
                    diff = torch.where(m, logp_diff, torch.zeros_like(logp_diff))
                    approx_kl = diff.sum() / count
                    approx_kl_abs = diff.abs().sum() / count
                    clipped_m = (torch.abs(ratio - 1.0) > cfg.clip_eps) & train_mask_f[sl]
                    clipfrac = clipped_m.to(dtype=torch.float32).sum() / denom
                else:
                    count = finite.to(dtype=torch.float32).sum().clamp_min(1.0)
                    diff = torch.where(finite, logp_diff, torch.zeros_like(logp_diff))
                    approx_kl = diff.sum() / count
                    approx_kl_abs = diff.abs().sum() / count
                    clipfrac = (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps).to(dtype=torch.float32).mean()
                    )

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # Unscale gradients before clipping so max_grad_norm is in the
                # original (unscaled) gradient space.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt_steps += 1

                loss_sum_t = loss_sum_t + loss.detach()
                kl_sum_t = kl_sum_t + approx_kl.detach()
                clipfrac_sum_t = clipfrac_sum_t + clipfrac.detach()
                value_loss_sum_t = value_loss_sum_t + value_loss.detach()
                policy_loss_sum_t = policy_loss_sum_t + policy_loss.detach()
                entropy_sum_t = entropy_sum_t + entropy_bonus.detach()

                loss_sum_update_t = loss_sum_update_t + loss.detach()
                kl_sum_update_t = kl_sum_update_t + approx_kl.detach()
                kl_abs_sum_update_t = kl_abs_sum_update_t + approx_kl_abs.detach()
                clipfrac_sum_update_t = clipfrac_sum_update_t + clipfrac.detach()
                value_loss_sum_update_t = value_loss_sum_update_t + value_loss.detach()
                policy_loss_sum_update_t = policy_loss_sum_update_t + policy_loss.detach()
                entropy_sum_update_t = entropy_sum_update_t + entropy_bonus.detach()
                maybe_refresh("update")

        if cfg.timing_sync:
            sync(cfg.device)
        t_opt_s = time.perf_counter() - t_opt0

        do_log = env_steps >= next_log_env_steps
        # Per-update means (avoid div-by-zero in small test configs).
        mb_den = max(1, mb_count)
        denom_t = torch.tensor(float(mb_den), device=device, dtype=torch.float32)
        loss_u = float((loss_sum_update_t / denom_t).item())
        kl_u = float((kl_sum_update_t / denom_t).item())
        kl_abs_u = float((kl_abs_sum_update_t / denom_t).item())
        clip_u = float((clipfrac_sum_update_t / denom_t).item())
        vloss_u = float((value_loss_sum_update_t / denom_t).item())
        ploss_u = float((policy_loss_sum_update_t / denom_t).item())
        ent_u = float((entropy_sum_update_t / denom_t).item())
        last_loss_u = loss_u
        last_kl_u = kl_u
        last_clip_u = clip_u
        last_ent_u = ent_u
        last_vloss_u = vloss_u
        last_ev = ev_rollout

        if do_log:
            next_log_env_steps += (
                cfg.log_every_env_steps if cfg.log_every_env_steps > 0 else env_steps_per_rollout
            )

        if writer is not None and do_log:
            if _HAS_PYNVML and cfg.device.startswith("cuda"):
                try:
                    dev_idx = int(cfg.device.split(":")[-1])
                    handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    writer.add_scalar("system/gpu_util_pct", float(util.gpu), env_steps)
                    writer.add_scalar(
                        "system/gpu_mem_pct", float(100.0 * mem.used / mem.total), env_steps
                    )
                except Exception:
                    # Non-fatal; driver issues, etc.
                    pass

            overall_sps_now = float(env_steps) / max(1e-9, time.perf_counter() - t0)
            writer.add_scalar("train/overall_sps", overall_sps_now, env_steps)
            writer.add_scalar("train/loss", loss_u, env_steps)
            writer.add_scalar("train/approx_kl", kl_u, env_steps)
            writer.add_scalar("train/approx_kl_abs", kl_abs_u, env_steps)
            writer.add_scalar("train/clipfrac", clip_u, env_steps)
            writer.add_scalar("train/value_loss", vloss_u, env_steps)
            writer.add_scalar("train/policy_loss", ploss_u, env_steps)
            writer.add_scalar("train/entropy", ent_u, env_steps)
            writer.add_scalar("train/explained_variance", ev_rollout, env_steps)
            # PPO loop diagnostics (helps explain sudden t_opt spikes).
            writer.add_scalar("debug/ppo_total_samples", float(total_effective), env_steps)
            writer.add_scalar("debug/ppo_mb_count", float(mb_count), env_steps)
            writer.add_scalar("debug/ppo_minibatch_size", float(cfg.minibatch_size), env_steps)
            writer.add_scalar(
                "debug/ppo_samples_per_rollout",
                float(env_steps_per_rollout),
                env_steps,
            )
            writer.add_scalar(
                "debug/ppo_total_frac_of_rollout",
                float(total_effective) / float(max(1, env_steps_per_rollout)),
                env_steps,
            )
            if cfg.league_enabled and opponent_filled > 0:
                w = pfsp_weights_from_bb_per_hand(
                    opponent_scores[:opponent_filled],
                    temperature=cfg.league_pfsp_temperature,
                    epsilon=cfg.league_pfsp_epsilon,
                    mode=cfg.league_pfsp_mode,
                    q=cfg.league_pfsp_q,
                )
                w = w / w.sum().clamp_min(1e-12)
                writer.add_scalar("league/pfsp_weight_min", float(w.min().item()), env_steps)
                writer.add_scalar("league/pfsp_weight_max", float(w.max().item()), env_steps)
                ent = -(w * (w.clamp_min(1e-12).log())).sum()
                writer.add_scalar("league/pfsp_weight_entropy", float(ent.item()), env_steps)
                ess = 1.0 / float((w * w).sum().clamp_min(1e-12).item())
                writer.add_scalar("league/pfsp_weight_ess", float(ess), env_steps)

                # Log individual opponent weights
                for i, meta in enumerate(opponent_metas[:opponent_filled]):
                    weight = float(w[i].item())
                    if meta.kind == "bot":
                        name = f"bot_{meta.id}"
                    else:
                        # Sanitize snapshot path for a clean tag name
                        p = Path(meta.id)
                        name = f"snap_{p.stem}"
                    writer.add_scalar(f"league_weights/{name}", weight, env_steps)

                # Log per-opponent-type breakdown
                bot_mask = opponent_kinds[:opponent_filled] == 1
                snap_mask = opponent_kinds[:opponent_filled] == 2

                if bot_mask.any():
                    bot_weight_sum = w[bot_mask].sum()
                    writer.add_scalar("league/bot_weight_total", float(bot_weight_sum), env_steps)

                if snap_mask.any():
                    snap_weight_sum = w[snap_mask].sum()
                    writer.add_scalar(
                        "league/snapshot_weight_total", float(snap_weight_sum), env_steps
                    )

                # Rollout mixture composition (cheap diagnostic for throughput changes).
                writer.add_scalar(
                    "league/rollout_snap_frac",
                    float(opp_kind.eq(2).to(dtype=torch.float32).mean().item()),
                    env_steps,
                )
                writer.add_scalar(
                    "league/rollout_bot_frac",
                    float(opp_kind.eq(1).to(dtype=torch.float32).mean().item()),
                    env_steps,
                )
                writer.add_scalar(
                    "league/rollout_selfplay_frac",
                    float(opp_kind.eq(0).to(dtype=torch.float32).mean().item()),
                    env_steps,
                )
            writer.add_scalar("actions/pct_fold", act_stats["pct_fold"], env_steps)
            writer.add_scalar("actions/pct_check", act_stats["pct_check"], env_steps)
            writer.add_scalar("actions/pct_call", act_stats["pct_call"], env_steps)
            writer.add_scalar("actions/pct_raise", act_stats["pct_raise"], env_steps)
            for b in range(NUM_RAISE_BUCKETS):
                key = f"bucket_{b}_pct"
                writer.add_scalar(f"actions/raise_{key}", act_stats[key], env_steps)
            for street, st in street_stats.items():
                writer.add_scalar(f"actions/{street}/pct_fold", st["pct_fold"], env_steps)
                writer.add_scalar(f"actions/{street}/pct_check", st["pct_check"], env_steps)
                writer.add_scalar(f"actions/{street}/pct_call", st["pct_call"], env_steps)
                writer.add_scalar(f"actions/{street}/pct_raise", st["pct_raise"], env_steps)
                for b in range(NUM_RAISE_BUCKETS):
                    key = f"bucket_{b}_pct"
                    writer.add_scalar(f"actions/{street}/raise_{key}", st[key], env_steps)
            writer.add_scalar(
                "train/invalid_rate",
                float(invalid_accum.item()) / max(1, env_steps),
                env_steps,
            )
            diag_total = float(diag_total_t.item())
            if diag_total > 0:
                writer.add_scalar(
                    "debug/logprob_old_nonfinite_rate",
                    float((diag_old_nonfinite_t / diag_total_t.clamp_min(1.0)).item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/logprob_new_nonfinite_rate",
                    float((diag_new_nonfinite_t / diag_total_t.clamp_min(1.0)).item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/log_ratio_nonfinite_rate",
                    float((diag_log_ratio_nonfinite_t / diag_total_t.clamp_min(1.0)).item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/log_ratio_abs_mean",
                    float((diag_log_ratio_abs_sum_t / diag_total_t.clamp_min(1.0)).item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/log_ratio_abs_max",
                    float(diag_log_ratio_abs_max_t.item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/logprob_old_min", float(diag_old_lp_min_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/logprob_old_max", float(diag_old_lp_max_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/logprob_new_min", float(diag_new_lp_min_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/logprob_new_max", float(diag_new_lp_max_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/log_ratio_min", float(diag_log_ratio_min_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/log_ratio_max", float(diag_log_ratio_max_t.item()), env_steps
                )

        # Periodic checkpointing by env steps.
        if next_save_env_steps > 0 and env_steps >= next_save_env_steps:
            if cfg.timing_sync:
                sync(cfg.device)
            t_misc0 = time.perf_counter()
            ckpt_path = Path(cfg.checkpoint_dir) / f"ckpt_env{env_steps}_upd{update}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=net,
                optimizer=opt,
                state=CheckpointState(env_steps=env_steps, opt_steps=opt_steps, update=update),
            )
            next_save_env_steps += cfg.save_every_env_steps
            if cfg.timing_sync:
                sync(cfg.device)
            t_misc_s += time.perf_counter() - t_misc0

        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            if cfg.timing_sync:
                sync(cfg.device)
            t_eval0 = time.perf_counter()
            eval_env = get_eval_env()
            all_eval: dict[str, dict[str, float]] = {}
            for bot_name in train_bots.available_bot_names():
                # For a couple of key opponents, also collect per-street action diagnostics.
                track = bot_name in {"nit", "random_aggressive"}
                if track:
                    m0 = eval_vs_bot(
                        env=eval_env,
                        policy=net,
                        bot_name=bot_name,
                        target_hands=cfg.eval_hands,
                        max_steps=cfg.eval_max_steps,
                        learner_seat=0,
                        deterministic=False,
                        track_actions=True,
                    )
                    m1 = eval_vs_bot(
                        env=eval_env,
                        policy=net,
                        bot_name=bot_name,
                        target_hands=cfg.eval_hands,
                        max_steps=cfg.eval_max_steps,
                        learner_seat=1,
                        deterministic=False,
                        track_actions=True,
                    )
                    bb_avg = 0.5 * (float(m0["bb_per_hand"]) + float(m1["bb_per_hand"]))
                    hands_total = int(m0["hands"]) + int(m1["hands"])
                    m = {
                        "bb_per_hand": bb_avg,
                        "bb_per_100": bb_avg * 100.0,
                        "hands": float(hands_total),
                        "bb_per_hand_p0": float(m0["bb_per_hand"]),
                        "bb_per_hand_p1": float(m1["bb_per_hand"]),
                    }
                    # Prefer seat-0 action stats as a stable diagnostic view.
                    for k, v in m0.items():
                        if k.startswith("actions/"):
                            m[k] = float(v)
                else:
                    m = eval_vs_bot_seat_swap(
                        env=eval_env,
                        policy=net,
                        bot_name=bot_name,
                        target_hands=cfg.eval_hands,
                        max_steps=cfg.eval_max_steps,
                        deterministic=False,
                    )
                all_eval[bot_name] = m
                if writer is not None:
                    writer.add_scalar(f"eval/{bot_name}_bb_per_hand", m["bb_per_hand"], env_steps)
                    writer.add_scalar(f"eval/{bot_name}_bb_per_100", m["bb_per_100"], env_steps)
                    writer.add_scalar(f"eval/{bot_name}_hands", m["hands"], env_steps)
                    writer.add_scalar(
                        f"eval/{bot_name}_bb_per_hand_p0", m["bb_per_hand_p0"], env_steps
                    )
                    writer.add_scalar(
                        f"eval/{bot_name}_bb_per_hand_p1", m["bb_per_hand_p1"], env_steps
                    )
                    if track:
                        for street in ("preflop", "flop", "turn", "river"):
                            for stat in (
                                "pct_fold",
                                "pct_check",
                                "pct_call",
                                "pct_raise",
                                "raise_frac_mean",
                                "raise_frac_min",
                                "raise_frac_max",
                            ):
                                tag = f"actions/{street}/{stat}"
                                key = f"actions/{street}/{stat}"
                                if key in m:
                                    writer.add_scalar(
                                        f"eval_actions/{bot_name}/{tag}", m[key], env_steps
                                    )

            # Keep the trainer's "headline" eval metric stable for progress bars and
            # `results.json` by using calling_station when present.
            eval_metrics = all_eval.get("calling_station", next(iter(all_eval.values()), {}))

            # Stronger eval: current policy vs a snapshot-mixture opponent.
            snap_indices = [
                i for i, m in enumerate(opponent_metas[:opponent_filled]) if m.kind == "snapshot"
            ]
            if cfg.league_enabled and snap_indices and writer is not None:
                s_scores = opponent_scores[snap_indices]
                w = pfsp_weights_from_bb_per_hand(
                    s_scores,
                    temperature=cfg.league_pfsp_temperature,
                    epsilon=cfg.league_pfsp_epsilon,
                    mode=cfg.league_pfsp_mode,
                    q=cfg.league_pfsp_q,
                )

                snap_policies = []
                for i in snap_indices:
                    path = os.path.join(cfg.checkpoint_dir, opponent_metas[i].id)
                    if path not in snapshot_models:
                        snapshot_models[path] = load_snapshot_model(path, device, cfg, scalar_dim)
                    snap_policies.append(snapshot_models[path])

                mix = eval_vs_snapshot_mix_seat_swap(
                    env=eval_env,
                    policy=net,
                    snapshots=snap_policies,
                    snap_weights=w,
                    target_hands=cfg.eval_hands,
                    max_steps=cfg.eval_max_steps,
                    deterministic=False,
                )
                writer.add_scalar("eval/pool_mix_bb_per_hand", mix["bb_per_hand"], env_steps)
                writer.add_scalar("eval/pool_mix_bb_per_100", mix["bb_per_100"], env_steps)
                writer.add_scalar("eval/pool_mix_hands", mix["hands"], env_steps)
                writer.add_scalar("eval/pool_mix_bb_per_hand_p0", mix["bb_per_hand_p0"], env_steps)
                writer.add_scalar("eval/pool_mix_bb_per_hand_p1", mix["bb_per_hand_p1"], env_steps)
            if cfg.timing_sync:
                sync(cfg.device)
            t_eval_s = time.perf_counter() - t_eval0

        # League: snapshot creation (disk-backed + in-memory ring).
        if (
            cfg.league_enabled
            and pool is not None
            and cfg.league_snapshot_every_updates > 0
            and (update + 1) % int(cfg.league_snapshot_every_updates) == 0
        ):
            if cfg.timing_sync:
                sync(cfg.device)
            # Demote previous main agent to historical
            current_main = next((o for o in opponent_metas if o.agent_role == "main"), None)
            if current_main is not None:
                pool.update_role(opponent_id=current_main.id, new_role="main_historical")

            meta = pool.add_snapshot(
                model=net, env_steps=env_steps, update=update, agent_role="main"
            )

            # Update in-memory pool using ring buffer logic for snapshots
            num_bots = len(bot_names)
            snap_capacity = int(cfg.league_snapshot_pool)

            if snap_capacity > 0:
                slot = snapshot_adds % snap_capacity
                idx = num_bots + slot

                # If we are replacing an existing snapshot, remove its model from cache
                if idx < len(opponent_metas):
                    old_id = opponent_metas[idx].id
                    if old_id in snapshot_models:
                        del snapshot_models[old_id]

                if idx < len(opponent_metas):
                    opponent_metas[idx] = meta
                else:
                    opponent_metas.append(meta)

                if idx < max_opponents:
                    opponent_scores[idx] = 0.0  # Neutral start
                    opponent_kinds[idx] = 2
                    opponent_filled = max(opponent_filled, idx + 1)

                snapshot_models[meta.id] = copy.deepcopy(net)
                snapshot_models[meta.id].eval()

            snapshot_adds += 1

        # League: periodic eval vs unified pool (for PFSP sampling weights).
        if (
            cfg.league_enabled
            and opponent_filled > 0
            and cfg.league_eval_every_updates > 0
            and (update + 1) % int(cfg.league_eval_every_updates) == 0
        ):
            was_training = net.training
            net.eval()
            try:
                if cfg.timing_sync:
                    sync(cfg.device)
                t_league0 = time.perf_counter()
                eval_env = get_eval_env()

                # 1. Evaluate Bots
                if cfg.league_eval_bots and (update + 1) % int(cfg.league_eval_bots_every) == 0:
                    for i, meta in enumerate(opponent_metas[:opponent_filled]):
                        if meta.kind != "bot":
                            continue

                        m = eval_vs_bot(
                            env=eval_env,
                            policy=net,
                            bot_name=meta.id,
                            target_hands=cfg.league_eval_hands,
                            max_steps=cfg.eval_max_steps,
                            deterministic=bool(cfg.league_eval_deterministic),
                        )
                        bb = float(m["bb_per_hand"])
                        opponent_scores[i] = bb

                        if pool is not None:
                            pool.update_eval(
                                opponent_id=meta.id,
                                bb_per_hand=bb,
                                hands=int(m["hands"]),
                            )
                        if writer is not None:
                            writer.add_scalar(f"league/bot_{meta.id}_bb_per_hand", bb, env_steps)

                # 2. Evaluate Snapshots
                # Filter for snapshots only
                snap_indices = [
                    i
                    for i, m in enumerate(opponent_metas[:opponent_filled])
                    if m.kind == "snapshot"
                ]

                if snap_indices:
                    subset = int(cfg.league_eval_subset)
                    eval_ids: list[int] = []

                    if subset <= 0 or subset >= len(snap_indices):
                        eval_ids = snap_indices
                    else:
                        # Pick most recent snapshots by env_steps.
                        # We need to sort snap_indices by the env_steps of the meta.
                        sorted_snaps = sorted(
                            snap_indices,
                            key=lambda i: (
                                opponent_metas[i].env_steps or 0,
                                opponent_metas[i].update or 0,
                            ),
                        )

                        k_recent = min(4, subset)
                        recent = sorted_snaps[-k_recent:]
                        recent_set = set(recent)

                        remaining = subset - k_recent
                        if remaining > 0:
                            cand = [i for i in sorted_snaps if i not in recent_set]
                            if cand:
                                r = min(remaining, len(cand))
                                perm = torch.randperm(len(cand), device=device)[:r].tolist()
                                pick = [cand[p] for p in perm]
                                eval_ids = recent + pick
                            else:
                                eval_ids = recent
                        else:
                            eval_ids = recent

                    det = bool(cfg.league_eval_deterministic)
                    for i in eval_ids:
                        meta = opponent_metas[i]
                        snap_path = os.path.join(cfg.checkpoint_dir, meta.id)

                        if snap_path not in snapshot_models:
                            snapshot_models[snap_path] = load_snapshot_model(
                                snap_path, device, cfg, scalar_dim
                            )

                        snap_net = snapshot_models[snap_path]

                        m = eval_vs_policy(
                            env=eval_env,
                            policy_p0=net,
                            policy_p1=snap_net,
                            target_hands=cfg.league_eval_hands,
                            max_steps=cfg.eval_max_steps,
                            deterministic=det,
                        )
                        bb_a = float(m["bb_per_hand"])  # learner as P0
                        hands_a = int(m["hands"])

                        bb_b_learn = 0.0
                        hands_b = 0
                        if bool(cfg.league_eval_seat_swap):
                            m2 = eval_vs_policy(
                                env=eval_env,
                                policy_p0=snap_net,
                                policy_p1=net,
                                target_hands=cfg.league_eval_hands,
                                max_steps=cfg.eval_max_steps,
                                deterministic=det,
                            )
                            # m2 is snapshot as P0; learner is P1, so learner bb/hand is -P0.
                            bb_b_learn = -float(m2["bb_per_hand"])
                            hands_b = int(m2["hands"])

                        score = (
                            0.5 * (bb_a + bb_b_learn) if bool(cfg.league_eval_seat_swap) else bb_a
                        )
                        opponent_scores[i] = float(score)

                        if pool is not None:
                            if bool(cfg.league_eval_seat_swap):
                                total_hands = hands_a + hands_b
                            else:
                                total_hands = hands_a
                            pool.update_eval(
                                opponent_id=meta.id,
                                bb_per_hand=float(score),
                                hands=int(total_hands),
                                bb_per_hand_p0=bb_a,
                                bb_per_hand_p1=(
                                    bb_b_learn if bool(cfg.league_eval_seat_swap) else None
                                ),
                            )

                if writer is not None and snap_indices:
                    # Log stats for snapshots specifically
                    s_vals = opponent_scores[snap_indices]
                    writer.add_scalar("league/snapshots_count", len(snap_indices), env_steps)
                    writer.add_scalar(
                        "league/snapshot_bb_per_hand_mean", float(s_vals.mean().item()), env_steps
                    )
                    writer.add_scalar(
                        "league/snapshot_bb_per_hand_min", float(s_vals.min().item()), env_steps
                    )
                    writer.add_scalar(
                        "league/snapshot_bb_per_hand_max", float(s_vals.max().item()), env_steps
                    )

                if cfg.timing_sync:
                    sync(cfg.device)
                t_league_eval_s = time.perf_counter() - t_league0
            finally:
                if was_training:
                    net.train()

        # Self-play eval vs snapshot pool (stronger than bots).
        has_snapshots = any(m.kind == "snapshot" for m in opponent_metas[:opponent_filled])
        if (
            cfg.selfplay_eval_enabled
            and cfg.league_enabled
            and has_snapshots
            and cfg.selfplay_eval_every_updates > 0
            and (update + 1) % int(cfg.selfplay_eval_every_updates) == 0
        ):
            was_training = net.training
            try:
                if cfg.timing_sync:
                    sync(cfg.device)
                t_sp0 = time.perf_counter()
                net.eval()

                eval_env = get_eval_env()
                det = bool(cfg.selfplay_eval_deterministic)
                seat_swap = bool(cfg.selfplay_eval_seat_swap)
                eval_ids = _pick_snapshot_eval_ids(subset=int(cfg.selfplay_eval_subset))

                scores: list[float] = []
                hands_total: list[int] = []
                for i in eval_ids:
                    meta = opponent_metas[i]
                    snap_path = os.path.join(cfg.checkpoint_dir, meta.id)

                    if snap_path not in snapshot_models:
                        snapshot_models[snap_path] = load_snapshot_model(
                            snap_path, device, cfg, scalar_dim
                        )

                    snap_net = snapshot_models[snap_path]
                    snap_net.eval()

                    m = eval_vs_policy(
                        env=eval_env,
                        policy_p0=net,
                        policy_p1=snap_net,
                        target_hands=int(cfg.selfplay_eval_hands),
                        max_steps=cfg.eval_max_steps,
                        deterministic=det,
                    )
                    bb_a = float(m["bb_per_hand"])
                    hands_a = int(m["hands"])

                    bb_b_learn = 0.0
                    hands_b = 0
                    if seat_swap:
                        m2 = eval_vs_policy(
                            env=eval_env,
                            policy_p0=snap_net,
                            policy_p1=net,
                            target_hands=int(cfg.selfplay_eval_hands),
                            max_steps=cfg.eval_max_steps,
                            deterministic=det,
                        )
                        bb_b_learn = -float(m2["bb_per_hand"])
                        hands_b = int(m2["hands"])

                    score = 0.5 * (bb_a + bb_b_learn) if seat_swap else bb_a
                    scores.append(float(score))
                    hands_total.append(hands_a + hands_b if seat_swap else hands_a)

                if writer is not None and scores:
                    s = torch.tensor(scores, device=device, dtype=torch.float32)
                    writer.add_scalar("selfplay/snapshots_filled", len(eval_ids), env_steps)
                    writer.add_scalar(
                        "selfplay/bb_per_hand_mean", float(s.mean().item()), env_steps
                    )
                    writer.add_scalar("selfplay/bb_per_hand_min", float(s.min().item()), env_steps)
                    writer.add_scalar("selfplay/bb_per_hand_max", float(s.max().item()), env_steps)
                    writer.add_scalar("selfplay/eval_opponents", float(len(scores)), env_steps)
                    writer.add_scalar(
                        "selfplay/eval_hands_total", float(sum(hands_total)), env_steps
                    )

                    # Explicit "vs latest snapshot" headline.
                    snap_indices = [
                        i
                        for i, m in enumerate(opponent_metas[:opponent_filled])
                        if m.kind == "snapshot"
                    ]
                    if snap_indices:
                        latest_i = max(
                            snap_indices,
                            key=lambda i: (
                                opponent_metas[i].env_steps or 0,
                                opponent_metas[i].update or 0,
                            ),
                        )

                        meta = opponent_metas[latest_i]
                        snap_path = os.path.join(cfg.checkpoint_dir, meta.id)
                        if snap_path not in snapshot_models:
                            snapshot_models[snap_path] = load_snapshot_model(
                                snap_path, device, cfg, scalar_dim
                            )
                        snap_net = snapshot_models[snap_path]

                        m_latest = eval_vs_policy(
                            env=eval_env,
                            policy_p0=net,
                            policy_p1=snap_net,
                            target_hands=int(cfg.selfplay_eval_hands),
                            max_steps=cfg.eval_max_steps,
                            deterministic=det,
                        )
                        bb0 = float(m_latest["bb_per_hand"])
                        bb1 = 0.0
                        if seat_swap:
                            m_latest2 = eval_vs_policy(
                                env=eval_env,
                                policy_p0=snap_net,
                                policy_p1=net,
                                target_hands=int(cfg.selfplay_eval_hands),
                                max_steps=cfg.eval_max_steps,
                                deterministic=det,
                            )
                            bb1 = -float(m_latest2["bb_per_hand"])
                        bb_latest = 0.5 * (bb0 + bb1) if seat_swap else bb0
                        writer.add_scalar("selfplay/vs_latest_bb_per_hand", bb_latest, env_steps)
                if cfg.timing_sync:
                    sync(cfg.device)
                t_selfplay_eval_s = time.perf_counter() - t_sp0
            finally:
                if was_training:
                    net.train()

        if writer is not None and do_log:
            writer.add_scalar("time/collect_s", t_collect_s, env_steps)
            writer.add_scalar("time/opt_s", t_opt_s, env_steps)
            writer.add_scalar("time/eval_s", t_eval_s, env_steps)
            writer.add_scalar("time/league_eval_s", t_league_eval_s, env_steps)
            writer.add_scalar("time/selfplay_eval_s", t_selfplay_eval_s, env_steps)
            writer.add_scalar("time/misc_s", t_misc_s, env_steps)
            writer.add_scalar(
                "time/update_total_s",
                t_collect_s + t_opt_s + t_eval_s + t_league_eval_s + t_selfplay_eval_s + t_misc_s,
                env_steps,
            )

        # tqdm progress bar postfix (fast to compute; avoids per-step noise).
        if do_log:
            update_elapsed = time.perf_counter() - update_t0
            update_sps = (cfg.num_envs * cfg.rollout_steps) / max(1e-9, update_elapsed)
            overall_sps_now = float(env_steps) / max(1e-9, time.perf_counter() - t0)
            # `diag_*` are only computed when `do_log` and `_epoch==0`; keep postfix robust.
            diag_total_f = float(diag_total_t.item()) if do_log else 0.0
            if diag_total_f > 0:
                nf = (diag_old_nonfinite_t + diag_new_nonfinite_t) / diag_total_t.clamp_min(1.0)
                nf_rate = float(nf.item())
            else:
                nf_rate = 0.0
            pbar.set_postfix(
                upd=update,
                sps=f"{update_sps / 1000:.0f}k",
                all=f"{overall_sps_now / 1000:.0f}k",
                tcol=f"{t_collect_s:.2f}",
                topt=f"{t_opt_s:.2f}",
                teval=f"{(t_eval_s + t_league_eval_s + t_selfplay_eval_s):.2f}",
                loss=f"{loss_u:.2f}",
                kl=f"{kl_u:.4f}",
                klabs=f"{kl_abs_u:.4f}",
                clip=f"{clip_u:.3f}",
                ent=f"{ent_u:.3f}",
                ev=f"{ev_rollout:.3f}",
                nf=f"{nf_rate:.2e}",
                eval=f"{eval_metrics.get('bb_per_hand', 0.0):.3f}",
            )
        maybe_refresh("idle")

        update += 1

    pbar.close()

    sync(cfg.device)
    t1 = time.perf_counter()

    elapsed_s = t1 - t0
    overall_sps = env_steps / elapsed_s if elapsed_s > 0 else float("inf")
    denom = max(1, opt_steps)
    denom_t = torch.tensor(float(denom), device=device, dtype=torch.float32)
    mean_loss = float((loss_sum_t / denom_t).item())
    approx_kl = float((kl_sum_t / denom_t).item())
    clipfrac = float((clipfrac_sum_t / denom_t).item())
    value_loss = float((value_loss_sum_t / denom_t).item())
    policy_loss = float((policy_loss_sum_t / denom_t).item())
    entropy = float((entropy_sum_t / denom_t).item())
    invalid_rate = float(invalid_accum.item()) / max(1, env_steps)

    if writer is not None:  # avoid TB flush thread warnings on exit
        writer.flush()
        writer.close()

    return TrainResult(
        config=asdict(cfg),
        env_steps=env_steps,
        opt_steps=opt_steps,
        elapsed_s=elapsed_s,
        overall_sps=overall_sps,
        mean_loss=mean_loss,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        value_loss=value_loss,
        policy_loss=policy_loss,
        entropy=entropy,
        invalid_rate=invalid_rate,
        eval=eval_metrics,
    )


# =========================================================================== #
# AlphaStar triad training loop (cfg.triad_enabled == True)
# =========================================================================== #
