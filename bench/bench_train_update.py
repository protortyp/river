"""
Benchmark a minimal train loop (collect + PPO update) to catch throughput cliffs.

This sits between the "env-only" benchmarks and the "PPO-only" benchmark:
- It uses the real WarpPokerEnv rollout collection
- It runs a PPO-style update similar to `train/train.py`
- It can intentionally allocate an extra eval env mid-run to reproduce memory cliffs

Usage:
  uv run python bench/bench_train_update.py --device cuda:0
  uv run python bench/bench_train_update.py --device cuda:0 --num-envs 32768
  uv run python bench/bench_train_update.py --device cuda:0 --trigger-eval-env-iter 5
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from gpu_poker import constants as const
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import PokerPolicyNet, raise_bucket_to_amount


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class BenchConfig:
    device: str
    num_envs: int
    rollout_steps: int
    num_epochs: int
    minibatch_size: int
    lr: float
    clip_eps: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    iters: int
    seed: int
    timing_sync: bool
    trigger_eval_env_iter: int
    eval_env_num_envs: int


def _select_player_state(
    *,
    player_id: torch.Tensor,  # [N] int64
    h0: torch.Tensor,  # [1,N,H]
    c0: torch.Tensor,  # [1,N,H]
    h1: torch.Tensor,  # [1,N,H]
    c1: torch.Tensor,  # [1,N,H]
) -> tuple[torch.Tensor, torch.Tensor]:
    pid = player_id.to(dtype=torch.int64)
    mask0 = pid.eq(0).view(1, -1, 1)
    h = torch.where(mask0, h0, h1)
    c_state = torch.where(mask0, c0, c1)
    return h, c_state


def _compute_gae(
    *,
    rewards: torch.Tensor,  # [T,N]
    dones: torch.Tensor,  # [T,N] bool
    values: torch.Tensor,  # [T,N]
    last_value: torch.Tensor,  # [N]
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


class _Rollout:
    def __init__(
        self,
        *,
        t: int,
        n: int,
        scalar_dim: int,
        mask_dim: int,
        hidden: int,
        device: torch.device,
    ) -> None:
        self.cards = torch.empty((t, n, 7), device=device, dtype=torch.int32)
        self.scalars = torch.empty((t, n, scalar_dim), device=device, dtype=torch.float32)
        self.action_mask = torch.empty((t, n, mask_dim), device=device, dtype=torch.bool)
        self.player_id = torch.empty((t, n), device=device, dtype=torch.int64)

        self.action_type = torch.empty((t, n), device=device, dtype=torch.int64)
        # Discrete raise-bucket index in [0, NUM_RAISE_BUCKETS); int64 storage.
        self.raise_bucket = torch.empty((t, n), device=device, dtype=torch.int64)
        self.logprob = torch.empty((t, n), device=device, dtype=torch.float32)
        self.value = torch.empty((t, n), device=device, dtype=torch.float32)

        self.reward = torch.empty((t, n), device=device, dtype=torch.float32)
        self.done = torch.empty((t, n), device=device, dtype=torch.bool)

        self.h = torch.empty((t, n, hidden), device=device, dtype=torch.float32)
        self.c = torch.empty((t, n, hidden), device=device, dtype=torch.float32)


def _mb_slices(total: int, mb_size: int, epoch: int) -> list[slice]:
    mb_size = max(1, int(mb_size))
    num_mbs = (total + mb_size - 1) // mb_size
    offset = (epoch * 104729) % max(1, num_mbs)
    out: list[slice] = []
    for mb_i in range(num_mbs):
        j = (mb_i + offset) % num_mbs
        start = j * mb_size
        end = min(total, start + mb_size)
        if end > start:
            out.append(slice(start, end))
    return out


def run(cfg: BenchConfig) -> dict:
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    env = WarpPokerEnv(num_envs=cfg.num_envs, device=cfg.device)
    obs = env.reset()

    scalar_dim = int(obs["scalars"].shape[1])
    mask_dim = int(obs["action_mask"].shape[1])
    net = PokerPolicyNet(scalar_dim=scalar_dim).to(device=device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    buf = _Rollout(
        t=cfg.rollout_steps,
        n=cfg.num_envs,
        scalar_dim=scalar_dim,
        mask_dim=mask_dim,
        hidden=net.lstm_hidden,
        device=device,
    )
    h0, c0 = PokerPolicyNet.init_state(cfg.num_envs, net.lstm_hidden, device)
    h1, c1 = PokerPolicyNet.init_state(cfg.num_envs, net.lstm_hidden, device)

    eval_env: WarpPokerEnv | None = None

    per_iter: list[dict[str, float]] = []
    env_steps_total = 0

    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

    for it in range(int(cfg.iters)):
        if cfg.timing_sync:
            _sync(cfg.device)
        t_col0 = time.perf_counter()

        for t in range(int(cfg.rollout_steps)):
            # IMPORTANT: rollout collection must be `no_grad` so we don't accidentally
            # backprop through the entire rollout graph during PPO updates.
            with torch.no_grad():
                pid = obs["player_id"].to(dtype=torch.int64)
                h, c_state = _select_player_state(player_id=pid, h0=h0, c0=c0, h1=h1, c1=c1)

                out = net.forward_step(
                    cards=obs["cards"],
                    scalars=obs["scalars"],
                    action_mask=obs["action_mask"],
                    h=h,
                    c=c_state,
                    terminated=None,
                    deterministic=False,
                )
                # Update per-player state.
                mask0 = pid.eq(0).view(1, -1, 1)
                h0 = torch.where(mask0, out.h, h0)
                c0 = torch.where(mask0, out.c, c0)
                h1 = torch.where(~mask0, out.h, h1)
                c1 = torch.where(~mask0, out.c, c1)
                action_type = out.action_type.to(dtype=torch.int64)
                raise_bucket = out.raise_bucket.to(dtype=torch.int64)
                amounts = raise_bucket_to_amount(
                    raise_bucket=raise_bucket,
                    pot_total=obs["pot_total"],
                    min_raise=obs["min_raise"],
                    max_raise=obs["max_raise"],
                )
                amounts = torch.where(
                    out.action_type.eq(const.ACTION_RAISE),
                    amounts,
                    torch.zeros_like(amounts),
                )

            # Save acting-player state at action time.
            buf.h[t].copy_(h.squeeze(0))
            buf.c[t].copy_(c_state.squeeze(0))
            buf.player_id[t].copy_(pid)

            buf.cards[t].copy_(obs["cards"].to(dtype=torch.int32))
            buf.scalars[t].copy_(obs["scalars"].to(dtype=torch.float32))
            buf.action_mask[t].copy_(obs["action_mask"].to(dtype=torch.bool))

            buf.action_type[t].copy_(action_type)
            buf.raise_bucket[t].copy_(raise_bucket)
            buf.logprob[t].copy_(out.logprob.to(dtype=torch.float32))
            buf.value[t].copy_(out.value.to(dtype=torch.float32))

            with torch.no_grad():
                next_obs = env.step(
                    out.action_type.to(dtype=torch.int32),
                    amounts.to(dtype=torch.int32),
                )

                bb = next_obs["big_blind"].to(dtype=torch.float32).clamp_min(1.0)
                reward_act = torch.where(pid.eq(0), next_obs["rewards"], -next_obs["rewards"]).to(
                    dtype=torch.float32
                )
                reward_act = reward_act / (bb * float(const.DEFAULT_STACK_BB))
                buf.reward[t].copy_(reward_act)
                buf.done[t].copy_(next_obs["terminated"].to(dtype=torch.bool))
                term = next_obs["terminated"].to(dtype=torch.bool).view(1, -1, 1)
                if term.any():
                    h0 = torch.where(term, torch.zeros_like(h0), h0)
                    c0 = torch.where(term, torch.zeros_like(c0), c0)
                    h1 = torch.where(term, torch.zeros_like(h1), h1)
                    c1 = torch.where(term, torch.zeros_like(c1), c1)
                obs = next_obs

        if cfg.timing_sync:
            _sync(cfg.device)
        t_col_s = time.perf_counter() - t_col0

        # Bootstrap value
        pid_last = obs["player_id"].to(dtype=torch.int64)
        h_last, c_last = _select_player_state(player_id=pid_last, h0=h0, c0=c0, h1=h1, c1=c1)
        with torch.no_grad():
            out_last = net.forward_step(
                cards=obs["cards"],
                scalars=obs["scalars"],
                action_mask=obs["action_mask"],
                h=h_last,
                c=c_last,
                terminated=None,
                deterministic=True,
            )
            last_value = out_last.value.to(dtype=torch.float32)

        if cfg.timing_sync:
            _sync(cfg.device)
        t_opt0 = time.perf_counter()

        advantages, returns = _compute_gae(
            rewards=buf.reward,
            dones=buf.done,
            values=buf.value,
            last_value=last_value,
            gamma=0.99,
            gae_lambda=0.95,
        )
        adv = advantages.reshape(-1)
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)

        cards_f = buf.cards.reshape(-1, 7)
        scalars_f = buf.scalars.reshape(-1, scalar_dim)
        mask_f = buf.action_mask.reshape(-1, mask_dim)
        action_type_f = buf.action_type.reshape(-1)
        raise_bucket_f = buf.raise_bucket.reshape(-1)
        old_logprob_f = buf.logprob.reshape(-1)
        returns_f = returns.reshape(-1)
        h_f = buf.h.reshape(-1, net.lstm_hidden)
        c_f = buf.c.reshape(-1, net.lstm_hidden)

        total = int(cards_f.shape[0])
        for epoch in range(int(cfg.num_epochs)):
            for sl in _mb_slices(total, cfg.minibatch_size, epoch):
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
                log_ratio_raw = eval_out.logprob - old_logprob_f[sl]
                ratio = torch.exp(log_ratio_raw.clamp(-20.0, 20.0))
                unclipped = ratio * adv[sl]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv[sl]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (eval_out.value - returns_f[sl]).pow(2).mean()
                entropy_bonus = eval_out.entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.max_grad_norm)
                opt.step()

        if cfg.timing_sync:
            _sync(cfg.device)
        t_opt_s = time.perf_counter() - t_opt0

        # Trigger a mid-run eval env allocation to reproduce VRAM cliffs.
        if int(cfg.trigger_eval_env_iter) >= 0 and it == int(cfg.trigger_eval_env_iter):
            if cfg.timing_sync:
                _sync(cfg.device)
            t_alloc0 = time.perf_counter()
            eval_env = WarpPokerEnv(num_envs=cfg.eval_env_num_envs, device=cfg.device)
            _ = eval_env.reset()
            if cfg.timing_sync:
                _sync(cfg.device)
            t_alloc_s = time.perf_counter() - t_alloc0
        else:
            t_alloc_s = 0.0

        env_steps_total += int(cfg.num_envs) * int(cfg.rollout_steps)
        row: dict[str, float] = {
            "iter": float(it),
            "t_collect_s": float(t_col_s),
            "t_opt_s": float(t_opt_s),
            "t_alloc_s": float(t_alloc_s),
        }
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            row["mem_allocated"] = float(torch.cuda.memory_allocated(device=device))
            row["mem_reserved"] = float(torch.cuda.memory_reserved(device=device))
            row["mem_peak_allocated"] = float(torch.cuda.max_memory_allocated(device=device))
            row["mem_peak_reserved"] = float(torch.cuda.max_memory_reserved(device=device))
        per_iter.append(row)

    elapsed_s = float(sum(r["t_collect_s"] + r["t_opt_s"] + r["t_alloc_s"] for r in per_iter))
    sps = float(env_steps_total) / max(1e-9, elapsed_s)
    out: dict[str, object] = {
        "elapsed_s": elapsed_s,
        "env_steps": int(env_steps_total),
        "sps": sps,
        "per_iter": per_iter,
    }
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        out["cuda_peak_allocated"] = float(torch.cuda.max_memory_allocated(device=device))
        out["cuda_peak_reserved"] = float(torch.cuda.max_memory_reserved(device=device))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark collect+PPO update timing")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-envs", type=int, default=65536)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=65536)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timing-sync", action="store_true")
    p.add_argument("--trigger-eval-env-iter", type=int, default=-1)
    p.add_argument("--eval-env-num-envs", type=int, default=1024)
    args = p.parse_args()

    cfg = BenchConfig(
        device=str(args.device),
        num_envs=int(args.num_envs),
        rollout_steps=int(args.rollout_steps),
        num_epochs=int(args.num_epochs),
        minibatch_size=int(args.minibatch_size),
        lr=float(args.lr),
        clip_eps=float(args.clip_eps),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        max_grad_norm=float(args.max_grad_norm),
        iters=int(args.iters),
        seed=int(args.seed),
        timing_sync=bool(args.timing_sync),
        trigger_eval_env_iter=int(args.trigger_eval_env_iter),
        eval_env_num_envs=int(args.eval_env_num_envs),
    )
    result = run(cfg)

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "torch": getattr(torch, "__version__", "unknown"),
            "device": cfg.device,
        },
        "config": asdict(cfg),
        "result": result,
    }

    out_dir = Path("bench/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"train_update_{ts}_{cfg.device.replace(':', '-')}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    (Path("bench") / "latest_train_update.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("=== pokergpu train update benchmark ===")
    print(f"device: {cfg.device}")
    print(f"num_envs: {cfg.num_envs} rollout_steps: {cfg.rollout_steps} iters: {cfg.iters}")
    print(f"num_epochs: {cfg.num_epochs} minibatch_size: {cfg.minibatch_size}")
    print(f"timing_sync: {cfg.timing_sync}")
    print(f"elapsed_s: {result['elapsed_s']:.6f}")
    print(f"env_steps: {result['env_steps']}")
    print(f"SPS: {result['sps']:.2f}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
