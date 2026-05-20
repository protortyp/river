import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from gpu_poker import constants as constants
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import PokerPolicyNet, raise_bucket_to_amount


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class TrainSmokeConfig:
    device: str
    num_envs: int
    warmup_steps: int
    rollout_steps: int
    updates: int
    seed: int
    lr: float
    entropy_coef: float
    value_coef: float
    clip_eps: float
    gamma: float
    gae_lambda: float
    max_grad_norm: float


@dataclass(frozen=True)
class TrainSmokeResult:
    config: dict
    env_steps: int
    total_elapsed_s: float
    total_elapsed_gpu_s: float | None
    collect_elapsed_s: float
    update_elapsed_s: float
    overall_sps: float
    collect_sps: float
    update_sps: float
    invalid_rate: float
    terminated_rate: float
    mean_loss: float


def _compute_gae(
    *,
    rewards: torch.Tensor,  # [T,B]
    dones: torch.Tensor,  # [T,B] bool
    values: torch.Tensor,  # [T,B]
    last_value: torch.Tensor,  # [B]
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_steps, batch = rewards.shape
    advantages = torch.zeros((t_steps, batch), device=rewards.device, dtype=torch.float32)
    gae = torch.zeros((batch,), device=rewards.device, dtype=torch.float32)
    next_value = last_value
    for t in range(t_steps - 1, -1, -1):
        not_done = (~dones[t]).to(dtype=torch.float32)
        delta = rewards[t] + gamma * not_done * next_value - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def run(cfg: TrainSmokeConfig) -> TrainSmokeResult:
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    env = WarpPokerEnv(num_envs=cfg.num_envs, device=cfg.device)
    obs = env.reset()

    device = torch.device(cfg.device)
    scalar_dim = int(obs["scalars"].shape[1])

    net = PokerPolicyNet(scalar_dim=scalar_dim, lstm_hidden=128).to(device=device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    h_state, c_state = PokerPolicyNet.init_state(cfg.num_envs, net.lstm_hidden, device)

    # Preallocate rollout buffers (reused each update).
    cards_buf = torch.empty((cfg.rollout_steps, cfg.num_envs, 7), device=device, dtype=torch.int32)
    scalars_buf = torch.empty(
        (cfg.rollout_steps, cfg.num_envs, scalar_dim), device=device, dtype=torch.float32
    )
    mask_buf = torch.empty(
        (cfg.rollout_steps, cfg.num_envs, constants.NUM_ACTIONS),
        device=device,
        dtype=torch.bool,
    )
    reset_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device, dtype=torch.bool)

    action_type_buf = torch.empty(
        (cfg.rollout_steps, cfg.num_envs), device=device, dtype=torch.int64
    )
    raise_bucket_buf = torch.empty(
        (cfg.rollout_steps, cfg.num_envs), device=device, dtype=torch.int64
    )
    old_logprob_buf = torch.empty(
        (cfg.rollout_steps, cfg.num_envs), device=device, dtype=torch.float32
    )
    value_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device, dtype=torch.float32)
    reward_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device, dtype=torch.float32)
    done_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device, dtype=torch.bool)

    # Warmup: run a few steps to compile kernels / populate caches.
    with torch.no_grad():
        for _ in range(cfg.warmup_steps):
            out = net.forward_step(
                cards=obs["cards"],
                scalars=obs["scalars"],
                action_mask=obs["action_mask"],
                h=h_state,
                c=c_state,
                terminated=obs.get("terminated", None),
                deterministic=False,
            )
            h_state, c_state = out.h, out.c

            amounts = raise_bucket_to_amount(
                raise_bucket=out.raise_bucket,
                pot_total=obs["pot_total"],
                min_raise=obs["min_raise"],
                max_raise=obs["max_raise"],
            )
            amounts = torch.where(
                out.action_type.eq(constants.ACTION_RAISE),
                amounts,
                torch.zeros_like(amounts),
            )
            obs = env.step(out.action_type.to(dtype=torch.int32), amounts)

    _sync(cfg.device)

    invalid_accum = torch.zeros((), device=device, dtype=torch.int64)
    terminated_accum = torch.zeros((), device=device, dtype=torch.int64)
    loss_sum = 0.0

    total_start_evt = total_end_evt = None
    collect_elapsed_s = 0.0
    update_elapsed_s = 0.0
    collect_elapsed_ms = 0.0
    update_elapsed_ms = 0.0
    use_cuda_timing = cfg.device.startswith("cuda") and torch.cuda.is_available()

    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        total_start_evt = torch.cuda.Event(enable_timing=True)
        total_end_evt = torch.cuda.Event(enable_timing=True)
        total_start_evt.record()

    t0 = time.perf_counter()

    for _ in range(cfg.updates):
        # Store initial RNN state for this rollout so we can replay it for the update.
        h0 = h_state.detach()
        c0 = c_state.detach()

        if use_cuda_timing:
            collect_start = torch.cuda.Event(enable_timing=True)
            collect_end = torch.cuda.Event(enable_timing=True)
            collect_start.record()
        else:
            t_collect0 = time.perf_counter()

        with torch.no_grad():
            for t in range(cfg.rollout_steps):
                cards_buf[t].copy_(obs["cards"])
                scalars_buf[t].copy_(obs["scalars"])
                mask_buf[t].copy_(obs["action_mask"])
                reset_buf[t].copy_(obs["terminated"])

                out = net.forward_step(
                    cards=obs["cards"],
                    scalars=obs["scalars"],
                    action_mask=obs["action_mask"],
                    h=h_state,
                    c=c_state,
                    terminated=obs.get("terminated", None),
                    deterministic=False,
                )

                h_state, c_state = out.h, out.c

                action_type_buf[t].copy_(out.action_type)
                raise_bucket_buf[t].copy_(out.raise_bucket)
                old_logprob_buf[t].copy_(out.logprob)
                value_buf[t].copy_(out.value)

                amounts = raise_bucket_to_amount(
                    raise_bucket=out.raise_bucket,
                    pot_total=obs["pot_total"],
                    min_raise=obs["min_raise"],
                    max_raise=obs["max_raise"],
                )
                amounts = torch.where(
                    out.action_type.eq(constants.ACTION_RAISE),
                    amounts,
                    torch.zeros_like(amounts),
                )
                next_obs = env.step(out.action_type.to(dtype=torch.int32), amounts)

                reward_buf[t].copy_(next_obs["rewards"].to(dtype=torch.float32))
                done_buf[t].copy_(next_obs["terminated"])

                invalid_accum = invalid_accum + next_obs["invalid_action"].sum().to(
                    dtype=torch.int64
                )
                terminated_accum = terminated_accum + next_obs["terminated"].sum().to(
                    dtype=torch.int64
                )

                obs = next_obs

            # Bootstrap value for the observation after the rollout.
            out_last = net.forward_step(
                cards=obs["cards"],
                scalars=obs["scalars"],
                action_mask=obs["action_mask"],
                h=h_state,
                c=c_state,
                terminated=obs.get("terminated", None),
                deterministic=True,
            )
            last_value = out_last.value

        if use_cuda_timing:
            collect_end.record()
        else:
            collect_elapsed_s += time.perf_counter() - t_collect0

        advantages, returns = _compute_gae(
            rewards=reward_buf,
            dones=done_buf,
            values=value_buf,
            last_value=last_value,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
        advantages = (advantages - adv_mean) / adv_std

        if use_cuda_timing:
            update_end = torch.cuda.Event(enable_timing=True)
        else:
            t_update0 = time.perf_counter()

        # PPO-style update (single epoch, full batch; good enough for benchmarking).
        h_upd = h0
        c_upd = c0
        policy_loss_accum = torch.zeros((), device=device, dtype=torch.float32)
        value_loss_accum = torch.zeros((), device=device, dtype=torch.float32)
        entropy_accum = torch.zeros((), device=device, dtype=torch.float32)

        for t in range(cfg.rollout_steps):
            eval_out = net.evaluate_step(
                cards=cards_buf[t],
                scalars=scalars_buf[t],
                action_mask=mask_buf[t],
                action_type=action_type_buf[t],
                raise_bucket=raise_bucket_buf[t],
                h=h_upd,
                c=c_upd,
                terminated=reset_buf[t],
            )
            h_upd, c_upd = eval_out.h, eval_out.c

            ratio = torch.exp(eval_out.logprob - old_logprob_buf[t])
            unclipped = ratio * advantages[t]
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages[t]
            policy_loss_accum = policy_loss_accum + (-torch.minimum(unclipped, clipped).mean())

            value_loss_accum = value_loss_accum + 0.5 * (eval_out.value - returns[t]).pow(2).mean()
            entropy_accum = entropy_accum + eval_out.entropy.mean()

        policy_loss = policy_loss_accum / float(cfg.rollout_steps)
        value_loss = value_loss_accum / float(cfg.rollout_steps)
        entropy_bonus = entropy_accum / float(cfg.rollout_steps)
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.max_grad_norm)
        opt.step()

        if use_cuda_timing:
            update_end.record()
            # Defer synchronization until the end; just accumulate event times later.
        else:
            update_elapsed_s += time.perf_counter() - t_update0

        loss_sum += float(loss.detach().item())

        if use_cuda_timing:
            # Synchronize once per update to read event timings (keeps per-step syncs removed).
            update_end.synchronize()
            collect_elapsed_ms += float(collect_start.elapsed_time(collect_end))
            update_elapsed_ms += float(collect_end.elapsed_time(update_end))

    t1 = time.perf_counter()
    if total_end_evt is not None:
        total_end_evt.record()

    _sync(cfg.device)

    total_elapsed_s = t1 - t0
    total_elapsed_gpu_s = None
    if total_start_evt is not None and total_end_evt is not None:
        total_elapsed_gpu_s = float(total_start_evt.elapsed_time(total_end_evt)) / 1000.0

    if use_cuda_timing:
        collect_elapsed_s = collect_elapsed_ms / 1000.0
        update_elapsed_s = update_elapsed_ms / 1000.0

    env_steps = cfg.num_envs * cfg.rollout_steps * cfg.updates
    overall_sps = env_steps / total_elapsed_s if total_elapsed_s > 0 else float("inf")
    collect_sps = env_steps / collect_elapsed_s if collect_elapsed_s > 0 else float("inf")
    update_sps = env_steps / update_elapsed_s if update_elapsed_s > 0 else float("inf")

    invalid_rate = float(invalid_accum.item()) / env_steps
    terminated_rate = float(terminated_accum.item()) / env_steps

    return TrainSmokeResult(
        config=asdict(cfg),
        env_steps=env_steps,
        total_elapsed_s=total_elapsed_s,
        total_elapsed_gpu_s=total_elapsed_gpu_s,
        collect_elapsed_s=collect_elapsed_s,
        update_elapsed_s=update_elapsed_s,
        overall_sps=overall_sps,
        collect_sps=collect_sps,
        update_sps=update_sps,
        invalid_rate=invalid_rate,
        terminated_rate=terminated_rate,
        mean_loss=loss_sum / cfg.updates if cfg.updates > 0 else 0.0,
    )


def main() -> int:
    device = os.environ.get("POKERGPU_TRAIN_SMOKE_DEVICE", _default_device())
    num_envs = int(os.environ.get("POKERGPU_TRAIN_SMOKE_NUM_ENVS", "16384"))
    warmup_steps = int(os.environ.get("POKERGPU_TRAIN_SMOKE_WARMUP", "50"))
    rollout_steps = int(os.environ.get("POKERGPU_TRAIN_SMOKE_ROLLOUT", "64"))
    updates = int(os.environ.get("POKERGPU_TRAIN_SMOKE_UPDATES", "10"))
    seed = int(os.environ.get("POKERGPU_TRAIN_SMOKE_SEED", "0"))

    lr = float(os.environ.get("POKERGPU_TRAIN_SMOKE_LR", "3e-4"))
    entropy_coef = float(os.environ.get("POKERGPU_TRAIN_SMOKE_ENTROPY_COEF", "0.01"))
    value_coef = float(os.environ.get("POKERGPU_TRAIN_SMOKE_VALUE_COEF", "0.5"))
    clip_eps = float(os.environ.get("POKERGPU_TRAIN_SMOKE_CLIP_EPS", "0.2"))
    gamma = float(os.environ.get("POKERGPU_TRAIN_SMOKE_GAMMA", "0.99"))
    gae_lambda = float(os.environ.get("POKERGPU_TRAIN_SMOKE_GAE_LAMBDA", "0.95"))
    max_grad_norm = float(os.environ.get("POKERGPU_TRAIN_SMOKE_MAX_GRAD_NORM", "1.0"))

    cfg = TrainSmokeConfig(
        device=device,
        num_envs=num_envs,
        warmup_steps=warmup_steps,
        rollout_steps=rollout_steps,
        updates=updates,
        seed=seed,
        lr=lr,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        clip_eps=clip_eps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
    )
    result = run(cfg)

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "torch": getattr(torch, "__version__", "unknown"),
        },
        "result": asdict(result),
    }

    out_dir = Path("train/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"policy_train_smoke_{ts}_{socket.gethostname()}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    Path("train/latest_policy_train.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote: {out_path}")
    print(
        f"overall_SPS={result.overall_sps:.2f} collect_SPS={result.collect_sps:.2f} "
        f"update_SPS={result.update_sps:.2f}"
    )
    print(
        f"invalid_rate={result.invalid_rate:.6f} terminated_rate={result.terminated_rate:.6f} "
        f"mean_loss={result.mean_loss:.6f}"
    )
    if result.total_elapsed_gpu_s is not None:
        print(f"total_elapsed_gpu_s={result.total_elapsed_gpu_s:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
