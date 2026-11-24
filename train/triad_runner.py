"""AlphaStar-style triad PPO runner (Main, Main Exploiter, League Exploiter)."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import raise_bucket_to_amount
from train.checkpointing import CheckpointState, save_checkpoint
from train.config import TrainConfig, TrainResult, normalize_cfg_paths, resolve_amp
from train.eval import eval_vs_policy_duplicate
from train.league import OpponentMeta, UnifiedOpponentPool
from train.policy_factory import (
    PolicyNet,
    load_snapshot_model,
    make_policy,
    push_transformer_token,
    stage_from_scalars,
)
from train.ppo_utils import compute_gae, select_player_state
from train.triad import (
    OPP_LIVE_MAIN,
    OPP_SELF,
    OPP_SNAPSHOT,
    ROLE_LE,
    ROLE_MAIN,
    ROLE_ME,
    ROLE_TAG_LE_HISTORICAL,
    ROLES,
    TriadController,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ModuleNotFoundError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _HAS_TB = False

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _scatter_subset_state(
    idx: torch.Tensor,
    pid_subset: torch.Tensor,
    h_new: torch.Tensor,
    c_new: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    h1: torch.Tensor,
    c1: torch.Tensor,
) -> None:
    """Write updated per-seat recurrent/packed state for a subset of env indices."""
    is_p1 = pid_subset.to(dtype=torch.bool)
    if (~is_p1).any():
        idx0 = idx[~is_p1]
        h0[:, idx0, :] = h_new[:, ~is_p1, :]
        c0[:, idx0, :] = c_new[:, ~is_p1, :]
    if is_p1.any():
        idx1 = idx[is_p1]
        h1[:, idx1, :] = h_new[:, is_p1, :]
        c1[:, idx1, :] = c_new[:, is_p1, :]


def _eval_seat_swapped(
    *,
    eval_env: WarpPokerEnv,
    policy_a: PolicyNet,
    policy_b: PolicyNet,
    target_hands: int,
    max_steps: int,
) -> tuple[float, float, float, int]:
    """Duplicate-deck seat-swapped stochastic eval returning (bb_avg, bb_p0, bb_p1, total_hands)."""
    m = eval_vs_policy_duplicate(
        env=eval_env,
        policy_a=policy_a,
        policy_b=policy_b,
        target_hands=target_hands,
        max_steps=max_steps,
        deterministic=False,
    )
    return (
        float(m["bb_per_hand"]),
        float(m["bb_per_hand_p0"]),
        float(m["bb_per_hand_p1"]),
        int(m["hands"]),
    )


def _bb_to_wr(bb_per_hand: float) -> float:
    """Map bb/hand into [0, 1] win-rate proxy; sigmoid(5x) crosses 0.70 at ~+17 bb/100."""
    return float(torch.sigmoid(torch.tensor(bb_per_hand * 5.0)).item())


def _run_impl_triad(cfg: TrainConfig) -> TrainResult:
    """Triad training loop: M / ME / LE on partitioned slices of `num_envs`."""
    cfg = normalize_cfg_paths(cfg, _REPO_ROOT)

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
    if int(cfg.starting_stack_min) < int(cfg.starting_stack_max):
        per_env_stack = torch.randint(
            int(cfg.starting_stack_min),
            int(cfg.starting_stack_max) + 1,
            (cfg.num_envs,),
            device=device,
            dtype=torch.int32,
        )
        env.reset_with_config(starting_stack=per_env_stack)
    obs = env.reset()
    scalar_dim = int(obs["scalars"].shape[1])

    def _factory() -> PolicyNet:
        return make_policy(cfg, scalar_dim, device)

    triad = TriadController(cfg=cfg, scalar_dim=scalar_dim, device=device, make_net=_factory)
    if cfg.resume_from:
        triad.init_from_resume(cfg.resume_from)

    amp_enabled, amp_dtype = resolve_amp(cfg)

    # Shared snapshot pool across roles (triad does not train against scripted bots).
    pool = UnifiedOpponentPool(
        pool_dir=Path(cfg.checkpoint_dir) / "league_pool",
        bot_names=[],
    )
    opponent_metas: list[OpponentMeta] = pool.load_index()
    max_opponents = (
        int(cfg.triad_main_pool_cap) + int(cfg.triad_me_pool_cap) + int(cfg.triad_le_pool_cap) + 16
    )

    def _build_scores_tensor(metas: list[OpponentMeta]) -> torch.Tensor:
        size = max(max_opponents, len(metas) + 8)
        scores = torch.zeros(size, device=device, dtype=torch.float32)
        for i, m in enumerate(metas):
            scores[i] = float(m.bb_per_hand) if m.bb_per_hand is not None else 0.0
        return scores

    opponent_scores = _build_scores_tensor(opponent_metas)
    snapshot_models: dict[str, PolicyNet] = {}

    def _load_snapshot(meta: OpponentMeta) -> PolicyNet:
        if meta.id in snapshot_models:
            return snapshot_models[meta.id]
        snapshot_models[meta.id] = load_snapshot_model(meta.id, device, cfg, scalar_dim)
        return snapshot_models[meta.id]

    # Per-env per-seat recurrent or packed-transformer state shared across both
    # learner and opponent forwards so every seat sees coherent hand history.
    hidden = int(triad.role(ROLE_MAIN).net.lstm_hidden)
    h0 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=torch.float32)
    c0 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=torch.float32)
    h1 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=torch.float32)
    c1 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=torch.float32)

    opp_kind = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    opp_id = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    learner_seat = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)

    def _resample_for_role(role: str, reset_mask_slice: torch.Tensor) -> None:
        sl = triad.env_slice(role)
        idx_local = torch.nonzero(reset_mask_slice, as_tuple=False).squeeze(-1)
        if idx_local.numel() == 0:
            return
        num = int(idx_local.numel())
        idx_global = idx_local + sl.start
        learner_seat[idx_global] = torch.randint(0, 2, (num,), device=device, dtype=torch.int64)
        out = triad.sample_opponents(
            role,
            num,
            opponent_metas=opponent_metas,
            opponent_scores=opponent_scores,
        )
        opp_kind[idx_global] = out["opp_kind"]
        opp_id[idx_global] = out["opp_id"]

    for role in ROLES:
        sl = triad.env_slice(role)
        _resample_for_role(role, torch.ones((sl.stop - sl.start,), device=device, dtype=torch.bool))

    eval_env: WarpPokerEnv | None = None

    writer = None
    if _HAS_TB:
        tb_dir = os.environ.get("POKERGPU_TRAIN_TB_DIR", "train/tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)

    from train import mlflow_logging as mlf

    mlf_run = mlf.start_run(cfg)

    env_steps_per_rollout = int(cfg.num_envs) * int(cfg.rollout_steps)
    max_updates = cfg.num_updates if cfg.total_env_steps <= 0 else 0
    target_env_steps = cfg.total_env_steps if cfg.total_env_steps > 0 else 0
    pbar_total = (
        target_env_steps
        if target_env_steps > 0
        else (max_updates * env_steps_per_rollout if max_updates > 0 else 0)
    )
    env_steps = 0
    update = 0
    pbar = tqdm(
        total=pbar_total if pbar_total > 0 else None,
        initial=0,
        unit="env_step",
        dynamic_ncols=True,
        disable=False,
    )
    log_stride = cfg.log_every_env_steps if cfg.log_every_env_steps > 0 else env_steps_per_rollout
    next_log_env_steps = env_steps + log_stride
    print_interval_s = max(0.5, float(cfg.log_every_seconds))
    next_print_time = time.perf_counter() + print_interval_s
    next_save_env_steps = (
        env_steps + cfg.save_every_env_steps if cfg.save_every_env_steps > 0 else 0
    )

    role_reward_sum: dict[str, float] = dict.fromkeys(ROLES, 0.0)
    role_reward_count: dict[str, float] = dict.fromkeys(ROLES, 0.0)

    t0 = time.perf_counter()
    sync_enabled = bool(cfg.timing_sync) and device.type == "cuda"

    def _t_now() -> float:
        if sync_enabled:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    per_role_metrics: dict[str, dict[str, float]] = {}

    while True:
        if max_updates > 0 and update >= max_updates:
            break
        if target_env_steps > 0 and env_steps >= target_env_steps:
            break

        t_phase_rollout = 0.0
        t_phase_returns = 0.0
        t_phase_ppo = 0.0
        t_phase_eval = 0.0
        t_phase_snapshot = 0.0
        t_phase_save = 0.0
        t_phase_log = 0.0
        t_upd_start = _t_now()

        # ---- Rollout collection -------------------------------------------
        _t_rollout_start = _t_now()
        with torch.no_grad():
            for t_step in range(cfg.rollout_steps):
                reset = obs["terminated"].to(dtype=torch.bool)
                if reset.any():
                    z = torch.zeros_like(h0)
                    h0 = torch.where(reset.view(1, -1, 1), z, h0)
                    c0 = torch.where(reset.view(1, -1, 1), z, c0)
                    h1 = torch.where(reset.view(1, -1, 1), z, h1)
                    c1 = torch.where(reset.view(1, -1, 1), z, c1)
                    for role in ROLES:
                        sl = triad.env_slice(role)
                        _resample_for_role(role, reset[sl])

                pid = obs["player_id"].to(dtype=torch.int64)
                ep_id = obs["episode_id"].to(dtype=torch.int32)
                h_batch, c_batch = select_player_state(player_id=pid, h0=h0, c0=c0, h1=h1, c1=c1)

                for role in ROLES:
                    rs = triad.role(role)
                    sl = triad.env_slice(role)
                    rs.buf.cards[t_step].copy_(obs["cards"][sl])
                    rs.buf.scalars[t_step].copy_(obs["scalars"][sl])
                    rs.buf.action_mask[t_step].copy_(obs["action_mask"][sl])
                    rs.buf.player_id[t_step].copy_(pid[sl])
                    rs.buf.episode_id[t_step].copy_(ep_id[sl])
                    rs.buf.reset_mask[t_step].copy_(reset[sl])
                    rs.buf.h[t_step].copy_(h_batch[0, sl, :])
                    rs.buf.c[t_step].copy_(c_batch[0, sl, :])

                is_selfplay = opp_kind.eq(OPP_SELF)
                is_learner_actor = is_selfplay | pid.eq(learner_seat)

                action_type = torch.full(
                    (cfg.num_envs,), c.ACTION_FOLD, device=device, dtype=torch.int32
                )
                raise_bucket = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
                logprob = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)
                value = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)

                # Learner forwards per role.
                for role in ROLES:
                    rs = triad.role(role)
                    sl = triad.env_slice(role)
                    learner_mask_slice = is_learner_actor[sl]
                    rs.buf.learner_mask[t_step].copy_(learner_mask_slice)
                    if not bool(learner_mask_slice.any().item()):
                        continue
                    idx_local = torch.nonzero(learner_mask_slice, as_tuple=False).squeeze(-1)
                    idx_global = idx_local + sl.start
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out_l = rs.net.forward_step(
                            cards=obs["cards"][idx_global],
                            scalars=obs["scalars"][idx_global],
                            action_mask=obs["action_mask"][idx_global],
                            h=h_batch[:, idx_global, :],
                            c=c_batch[:, idx_global, :],
                            terminated=None,
                            deterministic=False,
                        )
                    action_type[idx_global] = out_l.action_type.to(dtype=torch.int32)
                    raise_bucket[idx_global] = out_l.raise_bucket.to(dtype=raise_bucket.dtype)
                    logprob[idx_global] = out_l.logprob.to(dtype=logprob.dtype)
                    value[idx_global] = out_l.value.to(dtype=value.dtype)
                    _scatter_subset_state(
                        idx_global, pid[idx_global], out_l.h, out_l.c, h0, c0, h1, c1
                    )

                # Opponent forwards: always stochastic (poker Nash policies are mixed)
                # and state-coherent (read/write the opponent seat's slot in h0/h1).
                is_opp_actor = (~is_learner_actor) & (~is_selfplay)
                live_main_mask = is_opp_actor & opp_kind.eq(OPP_LIVE_MAIN)
                if live_main_mask.any():
                    idx_lm = torch.nonzero(live_main_mask, as_tuple=False).squeeze(-1)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out_lm = triad.role(ROLE_MAIN).net.forward_step(
                            cards=obs["cards"][idx_lm],
                            scalars=obs["scalars"][idx_lm],
                            action_mask=obs["action_mask"][idx_lm],
                            h=h_batch[:, idx_lm, :],
                            c=c_batch[:, idx_lm, :],
                            terminated=None,
                            deterministic=False,
                        )
                    action_type[idx_lm] = out_lm.action_type.to(dtype=torch.int32)
                    raise_bucket[idx_lm] = out_lm.raise_bucket.to(dtype=raise_bucket.dtype)
                    _scatter_subset_state(idx_lm, pid[idx_lm], out_lm.h, out_lm.c, h0, c0, h1, c1)

                snap_mask = is_opp_actor & opp_kind.eq(OPP_SNAPSHOT)
                if snap_mask.any():
                    for pool_idx in torch.unique(opp_id[snap_mask]).tolist():
                        pool_idx = int(pool_idx)
                        if pool_idx >= len(opponent_metas):
                            continue
                        meta = opponent_metas[pool_idx]
                        if meta.kind != "snapshot":
                            continue
                        m = snap_mask & opp_id.eq(pool_idx)
                        if not m.any():
                            continue
                        idx_s = torch.nonzero(m, as_tuple=False).squeeze(-1)
                        snap_net = _load_snapshot(meta)
                        with torch.autocast(
                            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
                        ):
                            out_s = snap_net.forward_step(
                                cards=obs["cards"][idx_s],
                                scalars=obs["scalars"][idx_s],
                                action_mask=obs["action_mask"][idx_s],
                                h=h_batch[:, idx_s, :],
                                c=c_batch[:, idx_s, :],
                                terminated=None,
                                deterministic=False,
                            )
                        action_type[idx_s] = out_s.action_type.to(dtype=torch.int32)
                        raise_bucket[idx_s] = out_s.raise_bucket.to(dtype=raise_bucket.dtype)
                        _scatter_subset_state(idx_s, pid[idx_s], out_s.h, out_s.c, h0, c0, h1, c1)

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
                    stage_at_action = stage_from_scalars(obs["scalars"])
                    h0, h1 = push_transformer_token(
                        h0,
                        h1,
                        player_id=pid,
                        action_type=action_type,
                        raise_bucket=raise_bucket,
                        stage=stage_at_action,
                    )

                bb = next_obs["big_blind"].to(dtype=torch.float32).clamp_min(1.0)
                reward_p0_full = next_obs["rewards"].to(dtype=torch.float32) / (
                    bb * float(c.DEFAULT_STACK_BB)
                )
                reward_p0_full = torch.where(
                    next_obs["terminated"].to(dtype=torch.bool),
                    reward_p0_full,
                    torch.zeros_like(reward_p0_full),
                )

                for role in ROLES:
                    rs = triad.role(role)
                    sl = triad.env_slice(role)
                    rs.buf.action_type[t_step].copy_(action_type[sl].to(dtype=torch.int64))
                    rs.buf.raise_bucket[t_step].copy_(raise_bucket[sl])
                    rs.buf.logprob[t_step].copy_(logprob[sl])
                    rs.buf.value[t_step].copy_(value[sl])
                    rs.buf.reward[t_step].zero_()
                    rs.buf.reward_p0[t_step].copy_(reward_p0_full[sl])
                    rs.buf.done[t_step].copy_(next_obs["terminated"][sl])

                obs = next_obs

        env_steps += env_steps_per_rollout
        pbar.update(env_steps_per_rollout)
        t_phase_rollout = _t_now() - _t_rollout_start

        # ---- Per-role GAE returns + advantages ----------------------------
        _t_returns_start = _t_now()
        backfill_chunk = max(1024, int(cfg.num_envs))
        with torch.no_grad():
            for role in ROLES:
                rs = triad.role(role)
                opp_mask = ~rs.buf.learner_mask
                if not bool(opp_mask.any().item()):
                    continue
                idx_t, idx_n = torch.nonzero(opp_mask, as_tuple=True)
                n_opp = int(idx_t.numel())
                for start in range(0, n_opp, backfill_chunk):
                    end = min(start + backfill_chunk, n_opp)
                    s_t = idx_t[start:end]
                    s_n = idx_n[start:end]
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out_v = rs.net.forward_step(
                            cards=rs.buf.cards[s_t, s_n],
                            scalars=rs.buf.scalars[s_t, s_n],
                            action_mask=rs.buf.action_mask[s_t, s_n],
                            h=rs.buf.h[s_t, s_n].unsqueeze(0),
                            c=rs.buf.c[s_t, s_n].unsqueeze(0),
                            terminated=None,
                            deterministic=True,
                        )
                    rs.buf.value[s_t, s_n] = out_v.value.to(dtype=rs.buf.value.dtype)

            last_value_by_role: dict[str, torch.Tensor] = {}
            last_player_id_by_role: dict[str, torch.Tensor] = {}
            pid_now = obs["player_id"].to(dtype=torch.int64)
            for role in ROLES:
                sl = triad.env_slice(role)
                pid_sl = pid_now[sl]
                h_b = torch.where(pid_sl.view(1, -1, 1).eq(0), h0[:, sl, :], h1[:, sl, :])
                c_b = torch.where(pid_sl.view(1, -1, 1).eq(0), c0[:, sl, :], c1[:, sl, :])
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out_last = triad.role(role).net.forward_step(
                        cards=obs["cards"][sl],
                        scalars=obs["scalars"][sl],
                        action_mask=obs["action_mask"][sl],
                        h=h_b,
                        c=c_b,
                        terminated=None,
                        deterministic=True,
                    )
                last_value_by_role[role] = out_last.value.detach().to(dtype=torch.float32)
                last_player_id_by_role[role] = pid_sl.clone()

        returns_by_role: dict[str, torch.Tensor] = {}
        adv_by_role: dict[str, torch.Tensor] = {}
        mask_by_role: dict[str, torch.Tensor | None] = {}
        for role in ROLES:
            rs = triad.role(role)
            reward_actor = torch.where(rs.buf.player_id.eq(0), rs.buf.reward_p0, -rs.buf.reward_p0)
            advantages, returns = compute_gae(
                rewards=reward_actor,
                dones=rs.buf.done.to(dtype=torch.bool),
                values=rs.buf.value,
                last_value=last_value_by_role[role],
                player_id=rs.buf.player_id.to(dtype=torch.int64),
                last_player_id=last_player_id_by_role[role],
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            train_mask = rs.buf.learner_mask
            if not bool(train_mask.any().item()):
                continue
            w = train_mask.to(dtype=torch.float32)
            denom = w.sum().clamp_min(1.0)
            adv_mean = (advantages * w).sum() / denom
            adv_var = ((advantages - adv_mean).pow(2) * w).sum() / denom
            adv_std = adv_var.sqrt().clamp_min(1e-6)
            adv_norm = (advantages - adv_mean) / adv_std

            returns_by_role[role] = returns
            adv_by_role[role] = adv_norm
            mask_by_role[role] = train_mask

            terminal_rewards = rs.buf.reward_p0[rs.buf.done.to(dtype=torch.bool)]
            if terminal_rewards.numel() > 0:
                role_reward_sum[role] += float(terminal_rewards.sum().item())
                role_reward_count[role] += float(terminal_rewards.numel())

        t_phase_returns = _t_now() - _t_returns_start

        # ---- PPO update ---------------------------------------------------
        _t_ppo_start = _t_now()
        if returns_by_role:
            per_role_metrics = triad.ppo_update(
                returns_by_role=returns_by_role,
                advantages_by_role=adv_by_role,
                train_mask_by_role=mask_by_role,
            )
        t_phase_ppo = _t_now() - _t_ppo_start

        # ---- Periodic evaluation (promotion gates + PFSP score updates) ---
        _t_eval_start = _t_now()
        triad_eval_every = max(1, int(cfg.triad_eval_every_updates))
        if (update + 1) % triad_eval_every == 0:
            if eval_env is None:
                eval_env = WarpPokerEnv(
                    num_envs=min(1024, int(cfg.num_envs)),
                    starting_stack=int(cfg.starting_stack),
                    small_blind=int(cfg.small_blind),
                    big_blind=int(cfg.big_blind),
                    device=cfg.device,
                )
            main_net = triad.role(ROLE_MAIN).net
            main_net.eval()
            eval_hands = max(64, int(cfg.triad_eval_hands))
            eval_max_steps = max(eval_hands * 10, int(cfg.eval_max_steps))

            # 1) ME and LE vs current Main.
            for role in (ROLE_ME, ROLE_LE):
                rs = triad.role(role)
                rs.net.eval()
                try:
                    bb_avg, _, _, _ = _eval_seat_swapped(
                        eval_env=eval_env,
                        policy_a=rs.net,
                        policy_b=main_net,
                        target_hands=eval_hands,
                        max_steps=eval_max_steps,
                    )
                    rs.eval_wr_vs_targets["main_current"] = _bb_to_wr(bb_avg)
                    rs.eval_bb_per_hand_vs_targets["main_current"] = bb_avg
                finally:
                    rs.net.train()

            # 2) Main vs snapshot pool -> updates PFSP opponent_scores.
            snap_indices = [i for i, m in enumerate(opponent_metas) if m.kind == "snapshot"]
            subset_cap = (
                int(cfg.league_eval_subset)
                if int(cfg.league_eval_subset) > 0
                else len(snap_indices)
            )
            # Prioritize unevaluated and newest snapshots so PFSP stays calibrated.
            snap_indices_prioritized = sorted(
                snap_indices,
                key=lambda idx: (
                    opponent_metas[idx].hands is not None,
                    -(opponent_metas[idx].update or 0),
                ),
            )[:subset_cap]
            for idx in snap_indices_prioritized:
                meta = opponent_metas[idx]
                snap_net = _load_snapshot(meta)
                bb_avg, bb_p0, bb_p1, total_h = _eval_seat_swapped(
                    eval_env=eval_env,
                    policy_a=main_net,
                    policy_b=snap_net,
                    target_hands=eval_hands,
                    max_steps=eval_max_steps,
                )
                opponent_scores[idx] = float(bb_avg)
                pool.update_eval(
                    opponent_id=meta.id,
                    bb_per_hand=bb_avg,
                    hands=total_h,
                    bb_per_hand_p0=bb_p0,
                    bb_per_hand_p1=bb_p1,
                )

            # 3) LE vs non-LE historical snapshots so LE promotion checks the league.
            non_le_indices = [
                i
                for i in snap_indices_prioritized
                if opponent_metas[i].agent_role != ROLE_TAG_LE_HISTORICAL
            ][:4]
            if non_le_indices:
                rs_le = triad.role(ROLE_LE)
                rs_le.net.eval()
                try:
                    for idx in non_le_indices:
                        meta = opponent_metas[idx]
                        snap_net = _load_snapshot(meta)
                        bb_avg, _, _, _ = _eval_seat_swapped(
                            eval_env=eval_env,
                            policy_a=rs_le.net,
                            policy_b=snap_net,
                            target_hands=eval_hands,
                            max_steps=eval_max_steps,
                        )
                        target_key = f"snap_{meta.agent_role}_{meta.update}"
                        rs_le.eval_wr_vs_targets[target_key] = _bb_to_wr(bb_avg)
                        rs_le.eval_bb_per_hand_vs_targets[target_key] = bb_avg
                finally:
                    rs_le.net.train()

            if snap_indices_prioritized:
                opponent_metas = pool.load_index()
                opponent_scores = _build_scores_tensor(opponent_metas)

            main_net.train()

        t_phase_eval = _t_now() - _t_eval_start

        # ---- Snapshot triggers + per-role pool pruning --------------------
        _t_snap_start = _t_now()
        decisions = triad.maybe_snapshot(update, env_steps)
        pool_changed = False
        for role, dec in decisions.items():
            if not dec.should_snapshot:
                continue
            rs = triad.role(role)
            meta = pool.add_snapshot(
                model=rs.net,
                env_steps=env_steps,
                update=update,
                agent_role=dec.agent_role_tag,
            )
            pool_changed = True
            if cfg.mlflow_upload_checkpoints == "all":
                mlf.log_checkpoint(
                    mlf_run,
                    meta.id,
                    artifact_path=f"checkpoints/snapshots/{role}",
                )
            removed_ids = pool.prune_role(
                agent_role=dec.agent_role_tag,
                max_keep=rs.spec.snapshot_pool_cap,
            )
            for rid in removed_ids:
                snapshot_models.pop(rid, None)

            applied = triad.mutate_after_snapshot(role, update)
            if writer is not None:
                writer.add_scalar(f"{role}/snapshot_count", float(rs.snapshot_count), env_steps)
                writer.add_scalar(f"{role}/snapshot_applied_{applied}", 1.0, env_steps)
            mlf.log_snapshot_event(
                mlf_run,
                role=role,
                snapshot_count=rs.snapshot_count,
                env_steps=env_steps,
                update=update,
                mutate_applied=applied,
            )

        if pool_changed:
            opponent_metas = pool.load_index()
            opponent_scores = _build_scores_tensor(opponent_metas)
            # Invalidate any in-flight env opp_id that now points out of range.
            invalid_snap = opp_kind.eq(OPP_SNAPSHOT) & (opp_id >= len(opponent_metas))
            if invalid_snap.any():
                opp_kind[invalid_snap] = OPP_SELF
                opp_id[invalid_snap] = 0

        forced = triad.maybe_force_reset_me(update)
        if forced is not None:
            rs_me = triad.role(ROLE_ME)
            if writer is not None:
                writer.add_scalar("me/force_reset_count", 1.0, env_steps)
                writer.add_scalar(
                    "me/force_reset_updates_since", float(forced["updates_since"]), env_steps
                )
            mlf.log_snapshot_event(
                mlf_run,
                role=ROLE_ME,
                snapshot_count=rs_me.snapshot_count,
                env_steps=env_steps,
                update=update,
                mutate_applied=str(forced["kind"]),
            )

        t_phase_snapshot = _t_now() - _t_snap_start

        # ---- Periodic checkpoint -----------------------------------------
        _t_save_start = _t_now()
        if cfg.save_every_env_steps > 0 and env_steps >= next_save_env_steps:
            triad.save_triad_checkpoint(
                cfg.checkpoint_dir,
                env_steps=env_steps,
                update=update,
            )
            ckpt_path = Path(cfg.checkpoint_dir) / f"triad_main_env{env_steps}_upd{update}.pt"
            if cfg.mlflow_upload_checkpoints in ("periodic", "all"):
                mlf.log_checkpoint(
                    mlf_run,
                    ckpt_path,
                    artifact_path="checkpoints/main/periodic",
                )
            next_save_env_steps += cfg.save_every_env_steps
        t_phase_save = _t_now() - _t_save_start

        # ---- Logging ------------------------------------------------------
        _t_log_start = _t_now()
        do_log = env_steps >= next_log_env_steps
        if do_log:
            overall_sps_now = float(env_steps) / max(1e-9, time.perf_counter() - t0)
            mlf_metrics: dict[str, float] = {
                "train/overall_sps": overall_sps_now,
                "train/update": float(update),
            }
            mlf_metrics.update(mlf.system_stats(cfg.device))

            if writer is not None:
                writer.add_scalar("train/overall_sps", overall_sps_now, env_steps)
                writer.add_scalar("train/update", float(update), env_steps)

            for role, m in per_role_metrics.items():
                for k, v in m.items():
                    mlf_metrics[f"{role}/{k}"] = float(v)
                    if writer is not None:
                        writer.add_scalar(f"{role}/{k}", float(v), env_steps)

            for role in ROLES:
                rs = triad.role(role)
                sl = triad.env_slice(role)
                mlf_metrics[f"{role}/snapshot_count"] = float(rs.snapshot_count)
                mlf_metrics[f"{role}/updates_since_last_snapshot"] = float(
                    update - rs.last_snapshot_update if rs.last_snapshot_update >= 0 else update
                )
                if writer is not None:
                    writer.add_scalar(f"{role}/snapshot_count", float(rs.snapshot_count), env_steps)

                for target_id, wr in rs.eval_wr_vs_targets.items():
                    mlf_metrics[f"{role}/eval_wr/{target_id}"] = float(wr)
                    if writer is not None:
                        writer.add_scalar(f"{role}/eval_wr/{target_id}", float(wr), env_steps)
                for target_id, bb in rs.eval_bb_per_hand_vs_targets.items():
                    mlf_metrics[f"{role}/eval_wr/{target_id}_bb_per_hand"] = float(bb)
                    if writer is not None:
                        writer.add_scalar(
                            f"{role}/eval_wr/{target_id}_bb_per_hand", float(bb), env_steps
                        )

                opp_dist = mlf.opp_kind_distribution(opp_kind[sl])
                for k, v in opp_dist.items():
                    mlf_metrics[f"{role}/{k}"] = v
                    if writer is not None:
                        writer.add_scalar(f"{role}/{k}", v, env_steps)

                act_dist = mlf.action_distribution(rs.buf.action_type, rs.buf.learner_mask)
                rb_dist = mlf.raise_bucket_distribution(
                    rs.buf.raise_bucket, rs.buf.action_type, rs.buf.learner_mask
                )
                for k, v in {**act_dist, **rb_dist}.items():
                    mlf_metrics[f"{role}/{k}"] = v
                    if writer is not None:
                        writer.add_scalar(f"{role}/{k}", v, env_steps)

                n = role_reward_count[role]
                if n > 0:
                    mean_r = role_reward_sum[role] / n
                    mlf_metrics[f"{role}/terminal_reward_mean"] = mean_r
                    mlf_metrics[f"{role}/terminal_hands_count"] = float(n)
                    if writer is not None:
                        writer.add_scalar(f"{role}/terminal_reward_mean", mean_r, env_steps)

            pool_sizes = {"total": len(opponent_metas)}
            for tag in ("main_historical", "main_exploiter", "league_exploiter"):
                pool_sizes[tag] = sum(1 for m in opponent_metas if m.agent_role == tag)
            for k, v in pool_sizes.items():
                mlf_metrics[f"league/pool_size_{k}"] = float(v)
                if writer is not None:
                    writer.add_scalar(f"league/pool_size_{k}", float(v), env_steps)

            me_wr = triad.role(ROLE_ME).eval_wr_vs_targets.get("main_current", 0.0)
            mlf_metrics["triad/me_curriculum_active"] = (
                1.0 if float(me_wr) < float(cfg.triad_me_curriculum_threshold) else 0.0
            )

            t_update_total = _t_now() - t_upd_start
            t_phase_log = _t_now() - _t_log_start
            t_phase_misc = max(
                0.0,
                t_update_total
                - (
                    t_phase_rollout
                    + t_phase_returns
                    + t_phase_ppo
                    + t_phase_eval
                    + t_phase_snapshot
                    + t_phase_save
                    + t_phase_log
                ),
            )
            mlf_metrics["time/rollout_s"] = t_phase_rollout
            mlf_metrics["time/returns_s"] = t_phase_returns
            mlf_metrics["time/ppo_s"] = t_phase_ppo
            mlf_metrics["time/eval_s"] = t_phase_eval
            mlf_metrics["time/snapshot_s"] = t_phase_snapshot
            mlf_metrics["time/save_s"] = t_phase_save
            mlf_metrics["time/log_s"] = t_phase_log
            mlf_metrics["time/misc_s"] = t_phase_misc
            mlf_metrics["time/update_total_s"] = t_update_total
            mlf_metrics["throughput/sps_rollout"] = (
                env_steps_per_rollout / t_phase_rollout if t_phase_rollout > 0 else 0.0
            )
            mlf_metrics["throughput/sps_update"] = (
                env_steps_per_rollout / t_update_total if t_update_total > 0 else 0.0
            )
            if writer is not None:
                for k in (
                    "rollout_s",
                    "returns_s",
                    "ppo_s",
                    "eval_s",
                    "snapshot_s",
                    "save_s",
                    "log_s",
                    "misc_s",
                    "update_total_s",
                ):
                    writer.add_scalar(f"time/{k}", mlf_metrics[f"time/{k}"], env_steps)

            mlf.log_metrics(mlf_run, mlf_metrics, step=env_steps)
            next_log_env_steps += log_stride
        else:
            t_phase_log = _t_now() - _t_log_start

        update += 1

        now_t = _t_now()
        if now_t >= next_print_time:
            sps_now = float(env_steps) / max(1e-9, now_t - t0)
            roles_summary = " ".join(
                f"{r.upper()}={per_role_metrics.get(r, {}).get('mb_count', 0):.0f}mb" for r in ROLES
            )
            t_update_total = (
                t_phase_rollout
                + t_phase_returns
                + t_phase_ppo
                + t_phase_eval
                + t_phase_snapshot
                + t_phase_save
                + t_phase_log
            )
            phase_break = (
                f"roll={t_phase_rollout:.2f}s "
                f"ret={t_phase_returns:.2f}s "
                f"ppo={t_phase_ppo:.2f}s "
                f"eval={t_phase_eval:.2f}s "
                f"snap={t_phase_snapshot:.2f}s "
                f"save={t_phase_save:.2f}s "
                f"log={t_phase_log:.2f}s "
                f"tot={t_update_total:.2f}s"
            )
            print(
                f"[upd {update:>5}] env={env_steps:>10}  sps={sps_now:>7.0f}  "
                f"{roles_summary}  |  {phase_break}",
                flush=True,
            )
            next_print_time = now_t + print_interval_s

    pbar.close()
    if writer is not None:
        writer.flush()
        writer.close()

    if cfg.mlflow_upload_checkpoints in ("final", "periodic", "all"):
        final_ckpt = Path(cfg.checkpoint_dir) / f"triad_main_final_env{env_steps}_upd{update}.pt"
        final_ckpt.parent.mkdir(parents=True, exist_ok=True)
        final_opt_steps = int(sum(m.get("mb_count", 0) for m in per_role_metrics.values()))
        save_checkpoint(
            path=str(final_ckpt),
            model=triad.role(ROLE_MAIN).net,
            optimizer=triad.role(ROLE_MAIN).opt,
            state=CheckpointState(
                env_steps=env_steps,
                opt_steps=final_opt_steps,
                update=update,
            ),
        )
        mlf.log_checkpoint(
            mlf_run,
            final_ckpt,
            artifact_path="checkpoints/main/final",
        )

    mlf.end_run(mlf_run, status="completed")

    elapsed_s = time.perf_counter() - t0
    overall_sps = float(env_steps) / max(1e-9, elapsed_s)
    last_main = per_role_metrics.get(ROLE_MAIN, {})
    return TrainResult(
        config=asdict(cfg),
        env_steps=env_steps,
        opt_steps=int(sum(m.get("mb_count", 0) for m in per_role_metrics.values())),
        elapsed_s=elapsed_s,
        overall_sps=overall_sps,
        mean_loss=float(last_main.get("loss", 0.0)),
        approx_kl=float(last_main.get("approx_kl", 0.0)),
        clipfrac=float(last_main.get("clipfrac", 0.0)),
        value_loss=float(last_main.get("value_loss", 0.0)),
        policy_loss=float(last_main.get("policy_loss", 0.0)),
        entropy=float(last_main.get("entropy", 0.0)),
        invalid_rate=0.0,
        eval={},
    )
