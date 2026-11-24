"""Regularized Nash Dynamics (R-NaD) trainer with CTDE omniscient critic.

Implements the zero-sum reward-transformed self-play algorithm from Perolat et al.
(DeepNash, Science 2022) on River's GPU-parallel WarpPokerEnv:

1. Collect full-batch self-play rollouts (zero opponent fragmentation).
2. Transform per-step actor rewards by subtracting `eta * (log pi - log pi_reg)`
   against a periodically updated anchor policy `pi_reg`. Because GAE alternates
   signs across opposing seats, this keeps the transformed game strictly zero-sum
   and pulls the fixed point monotonically toward a Nash equilibrium without cycling.
3. Train the value head with an optional omniscient critic (seeing both players'
   hole cards) to cut advantage variance while the actor stays strictly decentralized.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import raise_bucket_to_amount
from gpu_poker.policy_transformer import PokerTransformerPolicyNet
from train.buffer import RolloutBuffer
from train.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from train.config import TrainConfig, TrainResult, normalize_cfg_paths, resolve_amp
from train.eval import eval_vs_bot_seat_swap, eval_vs_policy_duplicate
from train.policy_factory import (
    make_policy,
    push_transformer_token,
    stage_from_scalars,
)
from train.ppo_utils import compute_gae, scatter_player_state, select_player_state

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ModuleNotFoundError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _HAS_TB = False

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_impl_rnad(cfg: TrainConfig) -> TrainResult:
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

    net = make_policy(cfg, scalar_dim, device)
    is_transformer = isinstance(net, PokerTransformerPolicyNet)
    use_omni = bool(cfg.use_omniscient_critic and is_transformer)

    # Frozen regularization anchor policy (pi_reg) updated every `rnad_reg_update_every` updates.
    net_reg = copy.deepcopy(net).to(device=device)
    net_reg.eval()
    for p in net_reg.parameters():
        p.requires_grad_(False)

    # Keep a frozen baseline copy for progress tracking via duplicate-deck eval.
    net_anchor_eval = copy.deepcopy(net).to(device=device)
    net_anchor_eval.eval()
    for p in net_anchor_eval.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.lr,
        fused=(device.type == "cuda"),
    )

    amp_enabled, amp_dtype = resolve_amp(cfg)
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(amp_enabled and amp_dtype == torch.float16),
    )

    env_steps = 0
    opt_steps = 0
    update = 0
    if cfg.resume_from:
        ckpt_state = load_checkpoint(
            path=cfg.resume_from,
            model=net,
            optimizer=opt,
            device=device,
        )
        net_reg.load_state_dict(net.state_dict())
        env_steps = int(ckpt_state.env_steps)
        opt_steps = int(ckpt_state.opt_steps)
        update = int(ckpt_state.update)

    hidden = int(net.lstm_hidden)
    buf = RolloutBuffer(
        t=cfg.rollout_steps,
        n=cfg.num_envs,
        scalar_dim=scalar_dim,
        hidden=hidden,
        device=device,
    )
    h0 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=torch.float32)
    c0 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=torch.float32)
    h1 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=torch.float32)
    c1 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=torch.float32)

    writer = None
    if _HAS_TB:
        tb_dir = os.environ.get("POKERGPU_TRAIN_TB_DIR", "train/tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)

    from train import mlflow_logging as mlf

    mlf_run = mlf.start_run(cfg)

    eval_env: WarpPokerEnv | None = None
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
        initial=env_steps,
        unit="env_step",
        dynamic_ncols=True,
        disable=False,
    )
    next_save_env_steps = (
        env_steps + cfg.save_every_env_steps if cfg.save_every_env_steps > 0 else 0
    )
    reg_update_every = max(1, int(cfg.rnad_reg_update_every))
    eta = float(cfg.rnad_eta)
    kl_clip = float(cfg.rnad_kl_clip)
    reg_updates = 0

    t0 = time.perf_counter()
    sync_enabled = bool(cfg.timing_sync) and device.type == "cuda"

    def _t_now() -> float:
        if sync_enabled:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    last_metrics: dict[str, float] = {}
    last_eval: dict[str, float] = {}
    invalid_total = 0.0
    total_steps_collected = 0.0

    while True:
        if max_updates > 0 and update >= max_updates:
            break
        if target_env_steps > 0 and env_steps >= target_env_steps:
            break

        t_upd_start = _t_now()

        # ---- 1. Full-batch self-play rollout ------------------------------
        with torch.no_grad():
            for t_step in range(cfg.rollout_steps):
                reset = obs["terminated"].to(dtype=torch.bool)
                if reset.any():
                    z = torch.zeros_like(h0)
                    h0 = torch.where(reset.view(1, -1, 1), z, h0)
                    c0 = torch.where(reset.view(1, -1, 1), z, c0)
                    h1 = torch.where(reset.view(1, -1, 1), z, h1)
                    c1 = torch.where(reset.view(1, -1, 1), z, c1)

                pid = obs["player_id"].to(dtype=torch.int64)
                h_batch, c_batch = select_player_state(player_id=pid, h0=h0, c0=c0, h1=h1, c1=c1)

                buf.cards[t_step].copy_(obs["cards"])
                buf.opp_cards[t_step].copy_(obs["opp_cards"])
                buf.scalars[t_step].copy_(obs["scalars"])
                buf.action_mask[t_step].copy_(obs["action_mask"])
                buf.player_id[t_step].copy_(pid)
                buf.episode_id[t_step].copy_(obs["episode_id"])
                buf.reset_mask[t_step].copy_(reset)
                buf.h[t_step].copy_(h_batch[0])
                buf.c[t_step].copy_(c_batch[0])

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    if use_omni and isinstance(net, PokerTransformerPolicyNet):
                        out = net.forward_step(
                            cards=obs["cards"],
                            scalars=obs["scalars"],
                            action_mask=obs["action_mask"],
                            h=h_batch,
                            c=c_batch,
                            terminated=None,
                            deterministic=False,
                            opp_cards=obs["opp_cards"],
                        )
                    else:
                        out = net.forward_step(
                            cards=obs["cards"],
                            scalars=obs["scalars"],
                            action_mask=obs["action_mask"],
                            h=h_batch,
                            c=c_batch,
                            terminated=None,
                            deterministic=False,
                        )

                action_type = out.action_type.to(dtype=torch.int32)
                raise_bucket = out.raise_bucket
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
                    raise_bucket=raise_bucket,
                    pot_total=obs["pot_total"],
                    min_raise=obs["min_raise"],
                    max_raise=obs["max_raise"],
                )
                amounts = torch.where(
                    action_type.eq(c.ACTION_RAISE), amounts, torch.zeros_like(amounts)
                )
                next_obs = env.step(action_type, amounts)

                if is_transformer:
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
                reward_p0 = next_obs["rewards"].to(dtype=torch.float32) / (
                    bb * float(c.DEFAULT_STACK_BB)
                )
                term = next_obs["terminated"].to(dtype=torch.bool)
                reward_p0 = torch.where(term, reward_p0, torch.zeros_like(reward_p0))
                reward_actor = torch.where(pid.eq(0), reward_p0, -reward_p0)

                buf.action_type[t_step].copy_(out.action_type)
                buf.raise_bucket[t_step].copy_(raise_bucket)
                buf.logprob[t_step].copy_(out.logprob)
                buf.value[t_step].copy_(out.value)
                buf.reward[t_step].copy_(reward_actor)
                buf.reward_p0[t_step].copy_(reward_p0)
                buf.done[t_step].copy_(term)

                invalid_total += float(next_obs["invalid_action"].sum().item())
                total_steps_collected += float(cfg.num_envs)
                obs = next_obs

        env_steps += env_steps_per_rollout
        pbar.update(env_steps_per_rollout)
        t_rollout = _t_now() - t_upd_start

        # ---- 2. R-NaD zero-sum reward transform + CTDE GAE ----------------
        t_ret_start = _t_now()
        total = cfg.rollout_steps * cfg.num_envs
        cards_f = buf.cards.reshape(-1, 7)
        opp_cards_f = buf.opp_cards.reshape(-1, 2)
        scalars_f = buf.scalars.reshape(-1, scalar_dim)
        mask_f = buf.action_mask.reshape(-1, c.NUM_ACTIONS)
        action_type_f = buf.action_type.reshape(-1)
        raise_bucket_f = buf.raise_bucket.reshape(-1)
        old_logprob_f = buf.logprob.reshape(-1)
        h_f = buf.h.reshape(-1, hidden)
        c_f = buf.c.reshape(-1, hidden)

        logprob_reg_f = torch.empty_like(old_logprob_f)
        chunk = max(1024, int(cfg.minibatch_size))

        # Sort flat buffer indices by history length (with random tie-breaking) so each
        # transformer minibatch trims to a short active sequence instead of padding to 32.
        if is_transformer:
            hist_lens_f = h_f[:, -1]
            sort_keys = hist_lens_f + torch.rand_like(hist_lens_f) * 0.5
            bucket_order = torch.argsort(sort_keys)
        else:
            bucket_order = torch.randperm(total, device=device)

        with torch.no_grad():
            for start in range(0, total, chunk):
                end = min(total, start + chunk)
                idx = bucket_order[start:end]
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out_reg = net_reg.evaluate_step(
                        cards=cards_f[idx],
                        scalars=scalars_f[idx],
                        action_mask=mask_f[idx],
                        action_type=action_type_f[idx],
                        raise_bucket=raise_bucket_f[idx],
                        h=h_f[idx].unsqueeze(0),
                        c=c_f[idx].unsqueeze(0),
                        terminated=None,
                    )
                logprob_reg_f[idx] = out_reg.logprob.to(dtype=torch.float32)

            # Bootstrap value at post-rollout state using omniscient critic when enabled.
            pid_now = obs["player_id"].to(dtype=torch.int64)
            h_last, c_last = select_player_state(player_id=pid_now, h0=h0, c0=c0, h1=h1, c1=c1)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                if use_omni and isinstance(net, PokerTransformerPolicyNet):
                    out_last = net.forward_step(
                        cards=obs["cards"],
                        scalars=obs["scalars"],
                        action_mask=obs["action_mask"],
                        h=h_last,
                        c=c_last,
                        terminated=None,
                        deterministic=True,
                        opp_cards=obs["opp_cards"],
                    )
                else:
                    out_last = net.forward_step(
                        cards=obs["cards"],
                        scalars=obs["scalars"],
                        action_mask=obs["action_mask"],
                        h=h_last,
                        c=c_last,
                        terminated=None,
                        deterministic=True,
                    )
            last_value = out_last.value.detach().to(dtype=torch.float32)

        log_ratio_reg = (
            buf.logprob - logprob_reg_f.reshape(cfg.rollout_steps, cfg.num_envs)
        ).clamp(min=-kl_clip, max=kl_clip)
        # Strictly zero-sum regularized reward: subtracting eta * log(pi / pi_reg) from the
        # acting player adds +eta * log(pi / pi_reg) to the opponent via GAE sign alternation.
        reward_reg = buf.reward - eta * log_ratio_reg
        reg_kl_mean = float(log_ratio_reg.mean().item())

        advantages, returns = compute_gae(
            rewards=reward_reg,
            dones=buf.done,
            values=buf.value,
            last_value=last_value,
            player_id=buf.player_id,
            last_player_id=pid_now,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
        adv_f = ((advantages - adv_mean) / adv_std).reshape(-1)
        returns_f = returns.reshape(-1)
        t_ret = _t_now() - t_ret_start

        # ---- 3. Clipped PPO update on regularized advantages --------------
        t_ppo_start = _t_now()
        mb_size = int(cfg.minibatch_size)
        num_mbs = (total + mb_size - 1) // max(1, mb_size)

        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        pi_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        v_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        ent_sum = torch.zeros((), device=device, dtype=torch.float32)
        kl_sum = torch.zeros((), device=device, dtype=torch.float32)
        clip_sum = torch.zeros((), device=device, dtype=torch.float32)
        ev_sum = torch.zeros((), device=device, dtype=torch.float32)
        mb_count = 0

        for _epoch in range(int(cfg.num_epochs)):
            mb_perm = torch.randperm(num_mbs, device=device).tolist()
            for j in mb_perm:
                start = int(j) * mb_size
                end = min(total, start + mb_size)
                if end <= start:
                    continue
                idx = bucket_order[start:end]

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    if use_omni and isinstance(net, PokerTransformerPolicyNet):
                        eval_out = net.evaluate_step(
                            cards=cards_f[idx],
                            scalars=scalars_f[idx],
                            action_mask=mask_f[idx],
                            action_type=action_type_f[idx],
                            raise_bucket=raise_bucket_f[idx],
                            h=h_f[idx].unsqueeze(0),
                            c=c_f[idx].unsqueeze(0),
                            terminated=None,
                            opp_cards=opp_cards_f[idx],
                        )
                    else:
                        eval_out = net.evaluate_step(
                            cards=cards_f[idx],
                            scalars=scalars_f[idx],
                            action_mask=mask_f[idx],
                            action_type=action_type_f[idx],
                            raise_bucket=raise_bucket_f[idx],
                            h=h_f[idx].unsqueeze(0),
                            c=c_f[idx].unsqueeze(0),
                            terminated=None,
                        )

                    log_ratio = (eval_out.logprob - old_logprob_f[idx]).clamp(min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    unclipped = ratio * adv_f[idx]
                    clipped = (
                        torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_f[idx]
                    )
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (eval_out.value - returns_f[idx]).pow(2).mean()
                    # Also keep the decentralized value head calibrated for ONNX inference.
                    if eval_out.decentralized_value is not None:
                        dec_v_loss = (
                            0.5 * (eval_out.decentralized_value - returns_f[idx]).pow(2).mean()
                        )
                        value_loss = value_loss + 0.1 * dec_v_loss
                    entropy_bonus = eval_out.entropy.mean()
                    loss = (
                        policy_loss
                        + float(cfg.value_coef) * value_loss
                        - float(cfg.entropy_coef) * entropy_bonus
                    )

                with torch.no_grad():
                    y = returns_f[idx]
                    var_y = y.var(unbiased=False)
                    var_res = (y - eval_out.value.detach()).var(unbiased=False)
                    ev = torch.where(var_y > 1e-8, 1.0 - var_res / var_y, torch.zeros_like(var_y))
                    clipfrac = (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps).to(dtype=torch.float32).mean()
                    )
                    approx_kl = (-log_ratio).mean()

                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.max_grad_norm)
                    opt.step()

                loss_sum = loss_sum + loss.detach()
                pi_loss_sum = pi_loss_sum + policy_loss.detach()
                v_loss_sum = v_loss_sum + value_loss.detach()
                ent_sum = ent_sum + entropy_bonus.detach()
                kl_sum = kl_sum + approx_kl.detach()
                clip_sum = clip_sum + clipfrac.detach()
                ev_sum = ev_sum + ev.detach()
                mb_count += 1
                opt_steps += 1

        denom = float(max(1, mb_count))
        last_metrics = {
            "loss": float((loss_sum / denom).item()),
            "policy_loss": float((pi_loss_sum / denom).item()),
            "value_loss": float((v_loss_sum / denom).item()),
            "entropy": float((ent_sum / denom).item()),
            "approx_kl": float((kl_sum / denom).item()),
            "clipfrac": float((clip_sum / denom).item()),
            "explained_variance": float((ev_sum / denom).item()),
            "rnad_reg_kl": reg_kl_mean,
        }
        t_ppo = _t_now() - t_ppo_start

        # ---- 4. Outer-loop R-NaD anchor update ----------------------------
        if (update + 1) % reg_update_every == 0:
            net_reg.load_state_dict(net.state_dict())
            reg_updates += 1

        # ---- 5. Periodic duplicate-deck eval ------------------------------
        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            if eval_env is None:
                eval_env = WarpPokerEnv(
                    num_envs=min(1024, int(cfg.num_envs)),
                    starting_stack=int(cfg.starting_stack),
                    small_blind=int(cfg.small_blind),
                    big_blind=int(cfg.big_blind),
                    device=cfg.device,
                )
            net.eval()
            try:
                eval_hands = max(64, int(cfg.eval_hands))
                eval_max_steps = max(eval_hands * 10, int(cfg.eval_max_steps))
                vs_init = eval_vs_policy_duplicate(
                    env=eval_env,
                    policy_a=net,
                    policy_b=net_anchor_eval,
                    target_hands=eval_hands,
                    max_steps=eval_max_steps,
                    deterministic=False,
                    seed=cfg.seed + update,
                )
                last_eval = {"vs_init_bb_per_100": float(vs_init["bb_per_100"])}
                for bot_name in ("calling_station", "nit", "loose_aggressive"):
                    bm = eval_vs_bot_seat_swap(
                        env=eval_env,
                        policy=net,
                        bot_name=bot_name,
                        target_hands=eval_hands,
                        max_steps=eval_max_steps,
                        deterministic=False,
                    )
                    last_eval[f"{bot_name}_bb_per_100"] = float(bm["bb_per_100"])
            finally:
                net.train()

        # ---- 6. Checkpointing & logging -----------------------------------
        if cfg.save_every_env_steps > 0 and env_steps >= next_save_env_steps:
            ckpt_path = Path(cfg.checkpoint_dir) / f"rnad_env{env_steps}_upd{update}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                path=str(ckpt_path),
                model=net,
                optimizer=opt,
                state=CheckpointState(env_steps=env_steps, opt_steps=opt_steps, update=update),
            )
            next_save_env_steps += cfg.save_every_env_steps

        sps_now = float(env_steps) / max(1e-9, time.perf_counter() - t0)
        if writer is not None:
            writer.add_scalar("train/overall_sps", sps_now, env_steps)
            writer.add_scalar("rnad/reg_updates", float(reg_updates), env_steps)
            for k, v in last_metrics.items():
                writer.add_scalar(f"rnad/{k}", v, env_steps)
            for k, v in last_eval.items():
                writer.add_scalar(f"eval/{k}", v, env_steps)

        mlf_payload = {
            "train/overall_sps": sps_now,
            "rnad/reg_updates": float(reg_updates),
            **{f"rnad/{k}": v for k, v in last_metrics.items()},
            **{f"eval/{k}": v for k, v in last_eval.items()},
        }
        mlf.log_metrics(mlf_run, mlf_payload, step=env_steps)

        if update % max(1, cfg.log_every_env_steps // max(1, env_steps_per_rollout)) == 0:
            print(
                f"[rnad upd {update:>5}] env={env_steps:>10} sps={sps_now:>7.0f} "
                f"ev={last_metrics['explained_variance']:+.3f} "
                f"ent={last_metrics['entropy']:.3f} kl_reg={reg_kl_mean:+.4f} "
                f"reg#={reg_updates} (roll={t_rollout:.2f}s ret={t_ret:.2f}s ppo={t_ppo:.2f}s)",
                flush=True,
            )

        update += 1

    pbar.close()
    if writer is not None:
        writer.flush()
        writer.close()

    # Always save final checkpoint when checkpoint_dir is configured and saves were requested.
    if cfg.save_every_env_steps > 0:
        final_ckpt = Path(cfg.checkpoint_dir) / f"rnad_final_env{env_steps}_upd{update}.pt"
        final_ckpt.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            path=str(final_ckpt),
            model=net,
            optimizer=opt,
            state=CheckpointState(env_steps=env_steps, opt_steps=opt_steps, update=update),
        )
        if cfg.mlflow_upload_checkpoints in ("final", "periodic", "all"):
            mlf.log_checkpoint(mlf_run, final_ckpt, artifact_path="checkpoints/rnad/final")

    mlf.end_run(mlf_run, status="completed")

    elapsed_s = time.perf_counter() - t0
    overall_sps = float(env_steps) / max(1e-9, elapsed_s)
    return TrainResult(
        config=asdict(cfg),
        env_steps=env_steps,
        opt_steps=opt_steps,
        elapsed_s=elapsed_s,
        overall_sps=overall_sps,
        mean_loss=float(last_metrics.get("loss", 0.0)),
        approx_kl=float(last_metrics.get("approx_kl", 0.0)),
        clipfrac=float(last_metrics.get("clipfrac", 0.0)),
        value_loss=float(last_metrics.get("value_loss", 0.0)),
        policy_loss=float(last_metrics.get("policy_loss", 0.0)),
        entropy=float(last_metrics.get("entropy", 0.0)),
        invalid_rate=invalid_total / max(1.0, total_steps_collected),
        eval=last_eval,
    )
