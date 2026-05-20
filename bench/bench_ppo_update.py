"""
Benchmark PPO update throughput (forward+backward) on synthetic rollouts.

This isolates PPO update performance from environment stepping and evaluation, and is
useful for catching stalls/regressions without waiting for a long training run.

Usage:
  uv run python bench/bench_ppo_update.py --device cuda:0
  uv run python bench/bench_ppo_update.py --device cuda:0 --num-envs 65536 --rollout-steps 64
  uv run python bench/bench_ppo_update.py --device cuda:0 --use-indexing
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

from gpu_poker import constants as c
from gpu_poker.policy import NUM_RAISE_BUCKETS, PokerPolicyNet


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class BenchConfig:
    device: str
    num_envs: int
    rollout_steps: int
    scalar_dim: int
    lstm_hidden: int
    card_embed_dim: int
    mlp_dim: int
    torso_layers: int
    head_layers: int
    head_dim: int | None
    minibatch_size: int
    num_epochs: int
    lr: float
    clip_eps: float
    value_coef: float
    entropy_coef: float
    warmup: int
    iters: int
    seed: int
    use_indexing: bool
    sync_every_iter: bool
    record_per_iter: bool


def _make_mask(*, total: int, device: torch.device) -> torch.Tensor:
    # Dense, mostly-legal mask to keep kernels realistic.
    m = torch.ones((total, c.NUM_ACTIONS), device=device, dtype=torch.bool)
    return m


def _compute_mb_slices(total: int, mb_size: int, epoch: int) -> list[slice]:
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


@torch.no_grad()
def _init_rollout_tensors(cfg: BenchConfig) -> dict[str, torch.Tensor]:
    device = torch.device(cfg.device)
    total = int(cfg.num_envs) * int(cfg.rollout_steps)

    cards = torch.randint(0, 52, (total, 7), device=device, dtype=torch.int32)
    scalars = torch.randn((total, cfg.scalar_dim), device=device, dtype=torch.float32)
    mask = _make_mask(total=total, device=device)

    # Random actions + raise buckets (logprob targets are synthetic).
    action_type = torch.randint(0, c.NUM_ACTIONS, (total,), device=device, dtype=torch.int64)
    raise_bucket = torch.randint(0, NUM_RAISE_BUCKETS, (total,), device=device, dtype=torch.int64)

    # Old logprob/value targets (synthetic).
    old_logprob = torch.randn((total,), device=device, dtype=torch.float32).clamp(-10.0, 5.0)
    returns = torch.randn((total,), device=device, dtype=torch.float32)
    adv = torch.randn((total,), device=device, dtype=torch.float32)
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)

    h = torch.zeros((total, cfg.lstm_hidden), device=device, dtype=torch.float32)
    c0 = torch.zeros((total, cfg.lstm_hidden), device=device, dtype=torch.float32)

    return {
        "cards": cards,
        "scalars": scalars,
        "mask": mask,
        "action_type": action_type,
        "raise_bucket": raise_bucket,
        "old_logprob": old_logprob,
        "returns": returns,
        "adv": adv,
        "h": h,
        "c": c0,
    }


def run(cfg: BenchConfig) -> dict:
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    net = PokerPolicyNet(
        scalar_dim=cfg.scalar_dim,
        card_embed_dim=cfg.card_embed_dim,
        mlp_dim=cfg.mlp_dim,
        torso_layers=cfg.torso_layers,
        lstm_hidden=cfg.lstm_hidden,
        head_layers=cfg.head_layers,
        head_dim=cfg.head_dim,
    ).to(device=device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    total = int(cfg.num_envs) * int(cfg.rollout_steps)
    data = _init_rollout_tensors(cfg)
    has_cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()

    def step_one_update() -> None:
        net.train()
        for epoch in range(int(cfg.num_epochs)):
            if cfg.use_indexing:
                offset = (epoch * 104729) % max(1, total)
                for start in range(0, total, cfg.minibatch_size):
                    end = min(total, start + cfg.minibatch_size)
                    if end <= start:
                        continue
                    mb = (torch.arange(start, end, device=device) + offset) % total
                    sl_cards = data["cards"][mb]
                    sl_scalars = data["scalars"][mb]
                    sl_mask = data["mask"][mb]
                    sl_action = data["action_type"][mb]
                    sl_rb = data["raise_bucket"][mb]
                    sl_h = data["h"][mb].unsqueeze(0)
                    sl_c = data["c"][mb].unsqueeze(0)
                    sl_old_lp = data["old_logprob"][mb]
                    sl_adv = data["adv"][mb]
                    sl_ret = data["returns"][mb]
                    eval_out = net.evaluate_step(
                        cards=sl_cards,
                        scalars=sl_scalars,
                        action_mask=sl_mask,
                        action_type=sl_action,
                        raise_bucket=sl_rb,
                        h=sl_h,
                        c=sl_c,
                        terminated=None,
                    )
                    log_ratio_raw = eval_out.logprob - sl_old_lp
                    ratio = torch.exp(log_ratio_raw.clamp(-20.0, 20.0))
                    unclipped = ratio * sl_adv
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * sl_adv
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (eval_out.value - sl_ret).pow(2).mean()
                    entropy_bonus = eval_out.entropy.mean()
                    loss = (
                        policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            else:
                for sl in _compute_mb_slices(total, cfg.minibatch_size, epoch):
                    eval_out = net.evaluate_step(
                        cards=data["cards"][sl],
                        scalars=data["scalars"][sl],
                        action_mask=data["mask"][sl],
                        action_type=data["action_type"][sl],
                        raise_bucket=data["raise_bucket"][sl],
                        h=data["h"][sl].unsqueeze(0),
                        c=data["c"][sl].unsqueeze(0),
                        terminated=None,
                    )
                    log_ratio_raw = eval_out.logprob - data["old_logprob"][sl]
                    ratio = torch.exp(log_ratio_raw.clamp(-20.0, 20.0))
                    unclipped = ratio * data["adv"][sl]
                    clipped = (
                        torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * data["adv"][sl]
                    )
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (eval_out.value - data["returns"][sl]).pow(2).mean()
                    entropy_bonus = eval_out.entropy.mean()
                    loss = (
                        policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

    # Warmup
    for _ in range(int(cfg.warmup)):
        step_one_update()
    _sync(cfg.device)

    if has_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)

    per_iter: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for i in range(int(cfg.iters)):
        iter_t0 = time.perf_counter()
        step_one_update()
        if cfg.sync_every_iter:
            _sync(cfg.device)
        iter_dt = time.perf_counter() - iter_t0
        if cfg.record_per_iter:
            row: dict[str, float] = {"iter": float(i), "elapsed_s": float(iter_dt)}
            if has_cuda:
                row["mem_allocated"] = float(torch.cuda.memory_allocated(device=device))
                row["mem_reserved"] = float(torch.cuda.memory_reserved(device=device))
                row["mem_peak_allocated"] = float(torch.cuda.max_memory_allocated(device=device))
                row["mem_peak_reserved"] = float(torch.cuda.max_memory_reserved(device=device))
            per_iter.append(row)
    if not cfg.sync_every_iter:
        _sync(cfg.device)
    dt = time.perf_counter() - t0

    samples = int(cfg.iters) * int(cfg.num_epochs) * total
    sps = samples / max(1e-9, dt)
    out: dict[str, object] = {"elapsed_s": dt, "samples": samples, "samples_per_s": sps}

    if cfg.record_per_iter:
        out["per_iter"] = per_iter
        times = torch.tensor([r["elapsed_s"] for r in per_iter], dtype=torch.float64)
        out["iter_stats_s"] = {
            "min": float(times.min().item()) if times.numel() else 0.0,
            "median": float(times.median().item()) if times.numel() else 0.0,
            "p95": float(times.kthvalue(max(1, int(0.95 * times.numel()))).values.item())
            if times.numel()
            else 0.0,
            "p99": float(times.kthvalue(max(1, int(0.99 * times.numel()))).values.item())
            if times.numel()
            else 0.0,
            "max": float(times.max().item()) if times.numel() else 0.0,
        }
    if has_cuda:
        out["cuda_peak_allocated"] = float(torch.cuda.max_memory_allocated(device=device))
        out["cuda_peak_reserved"] = float(torch.cuda.max_memory_reserved(device=device))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark PPO update throughput")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-envs", type=int, default=65536)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--scalar-dim", type=int, default=17)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--card-embed-dim", type=int, default=64)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--torso-layers", type=int, default=3)
    p.add_argument("--head-layers", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=0, help="0 means null/default")
    p.add_argument("--minibatch-size", type=int, default=65536)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-indexing", action="store_true")
    p.add_argument(
        "--sync-every-iter",
        action="store_true",
        help="Synchronize CUDA after each update (detects queue build-up / stalls).",
    )
    p.add_argument(
        "--record-per-iter",
        action="store_true",
        help="Record per-iter times and (CUDA) memory stats into the JSON output.",
    )
    args = p.parse_args()

    head_dim = None if int(args.head_dim) == 0 else int(args.head_dim)
    cfg = BenchConfig(
        device=args.device,
        num_envs=int(args.num_envs),
        rollout_steps=int(args.rollout_steps),
        scalar_dim=int(args.scalar_dim),
        lstm_hidden=int(args.lstm_hidden),
        card_embed_dim=int(args.card_embed_dim),
        mlp_dim=int(args.mlp_dim),
        torso_layers=int(args.torso_layers),
        head_layers=int(args.head_layers),
        head_dim=head_dim,
        minibatch_size=int(args.minibatch_size),
        num_epochs=int(args.num_epochs),
        lr=float(args.lr),
        clip_eps=float(args.clip_eps),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        warmup=int(args.warmup),
        iters=int(args.iters),
        seed=int(args.seed),
        use_indexing=bool(args.use_indexing),
        sync_every_iter=bool(args.sync_every_iter),
        record_per_iter=bool(args.record_per_iter),
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
    suffix = "indexing" if cfg.use_indexing else "slice"
    out_path = out_dir / f"ppo_update_{suffix}_{ts}_{cfg.device.replace(':', '-')}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    (Path("bench") / "latest_ppo_update.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("=== pokergpu PPO update benchmark ===")
    print(f"device: {cfg.device}")
    total = cfg.num_envs * cfg.rollout_steps
    print(f"num_envs: {cfg.num_envs} rollout_steps: {cfg.rollout_steps} total={total}")
    print(
        f"minibatch_size: {cfg.minibatch_size} num_epochs: {cfg.num_epochs} "
        f"use_indexing: {cfg.use_indexing} sync_every_iter: {cfg.sync_every_iter}"
    )
    print(f"elapsed_s: {result['elapsed_s']:.6f}")
    print(f"samples: {result['samples']}")
    print(f"samples/s: {result['samples_per_s']:.2f}")
    if "iter_stats_s" in result:
        stats = result["iter_stats_s"]
        print(f"iter_stats_s: {stats}")
    if "cuda_peak_allocated" in result:
        peak_a = float(result["cuda_peak_allocated"]) / (1024**3)
        peak_r = float(result["cuda_peak_reserved"]) / (1024**3)
        print(f"cuda_peak_allocated_gb: {peak_a:.3f} cuda_peak_reserved_gb: {peak_r:.3f}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
