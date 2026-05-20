"""
Benchmark league rollout collection throughput vs snapshot pool fragmentation.

This isolates the performance issue where many distinct snapshot opponents cause many
small forward passes per env step, tanking collection SPS.

Usage examples:
  uv run python bench/bench_league_rollout.py --device cuda:0
  uv run python bench/bench_league_rollout.py --device cuda:0 --snapshot-filled 32
  uv run python bench/bench_league_rollout.py --device cuda:0 --snapshot-filled 32 \\
    --rollout-snapshot-k 0
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
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import PokerPolicyNet, raise_bucket_to_amount


def pfsp_weights_from_bb_per_hand(
    bb_per_hand: torch.Tensor, *, temperature: float, epsilon: float
) -> torch.Tensor:
    """
    Local copy of `train.league.pfsp_weights_from_bb_per_hand`.

    Keeping this benchmark self-contained avoids `ModuleNotFoundError: train` when running
    `uv run bench/bench_league_rollout.py` directly.
    """
    t = max(1e-6, float(temperature))
    p = torch.sigmoid(bb_per_hand / t)
    w = p * (1.0 - p)
    if epsilon > 0:
        w = w + float(epsilon)
    return w


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class BenchConfig:
    device: str
    num_envs: int
    rollout_steps: int
    warmup: int
    iters: int
    snapshot_pool: int
    snapshot_filled: int
    rollout_snapshot_k: int
    league_eta_selfplay: float
    league_bot_prob: float
    seed: int


@torch.no_grad()
def run(cfg: BenchConfig) -> dict:
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    env = WarpPokerEnv(num_envs=cfg.num_envs, device=cfg.device)
    obs = env.reset()
    scalar_dim = int(obs["scalars"].shape[1])

    net = PokerPolicyNet(scalar_dim=scalar_dim).to(device=device)
    net.eval()

    # Snapshot opponents (same architecture).
    snapshots: list[PokerPolicyNet] = []
    for _ in range(int(cfg.snapshot_pool)):
        s = PokerPolicyNet(scalar_dim=scalar_dim).to(device=device)
        s.eval()
        snapshots.append(s)

    snapshot_filled = int(min(cfg.snapshot_filled, len(snapshots)))
    snapshot_scores = torch.zeros((max(1, snapshot_filled),), device=device, dtype=torch.float32)

    # Opponent assignment per env: 0=self, 1=bot (ignored here), 2=snapshot.
    opp_kind = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    opp_id = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    learner_seat = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)

    # Snapshot opponents are evaluated memoryless here; fixed zeros avoid allocations.
    h0, c0 = PokerPolicyNet.init_state(cfg.num_envs, net.lstm_hidden, device)
    snap_h0 = torch.zeros_like(h0)
    snap_c0 = torch.zeros_like(c0)

    def resample_roles(mask: torch.Tensor) -> None:
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        n = int(idx.numel())
        learner_seat[idx] = torch.randint(0, 2, (n,), device=device, dtype=torch.int64)

        u = torch.rand((n,), device=device, dtype=torch.float32)
        kind = torch.zeros((n,), device=device, dtype=torch.int64)
        non_self = u >= float(cfg.league_eta_selfplay)
        u2 = u - float(cfg.league_eta_selfplay)
        want_bot = non_self & (u2 < float(cfg.league_bot_prob))
        want_snap = non_self & (~want_bot)

        if want_bot.any():
            kind = torch.where(want_bot, torch.ones_like(kind), kind)

        if want_snap.any():
            if snapshot_filled > 0:
                kind = torch.where(want_snap, torch.full_like(kind, 2), kind)
                w = pfsp_weights_from_bb_per_hand(
                    snapshot_scores[:snapshot_filled].clone(),
                    temperature=1.0,
                    epsilon=1e-3,
                )
                w = w / w.sum().clamp_min(1e-12)
                k = int(cfg.rollout_snapshot_k)
                if k <= 0 or k >= snapshot_filled:
                    choice = torch.multinomial(
                        w, num_samples=int(want_snap.sum().item()), replacement=True
                    ).to(dtype=torch.int64)
                    opp_id[idx[want_snap]] = choice
                else:
                    active = torch.multinomial(w, num_samples=k, replacement=False)
                    active_w = w[active]
                    active_w = active_w / active_w.sum().clamp_min(1e-12)
                    pick = torch.multinomial(
                        active_w, num_samples=int(want_snap.sum().item()), replacement=True
                    ).to(dtype=torch.int64)
                    opp_id[idx[want_snap]] = active[pick]
            else:
                kind = torch.where(want_snap, torch.zeros_like(kind), kind)

        opp_kind[idx] = kind

    resample_roles(torch.ones((cfg.num_envs,), device=device, dtype=torch.bool))

    def do_rollout() -> None:
        nonlocal obs
        for _ in range(cfg.rollout_steps):
            reset = obs["terminated"].to(dtype=torch.bool)
            if reset.any():
                resample_roles(reset)

            pid = obs["player_id"].to(dtype=torch.int64)
            is_self = opp_kind.eq(0)
            is_learner_actor = is_self | pid.eq(learner_seat)
            is_opp_actor = ~is_learner_actor

            action_type = torch.full(
                (cfg.num_envs,), c.ACTION_FOLD, device=device, dtype=torch.int32
            )
            raise_bucket = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)

            # Learner-controlled decisions.
            if is_learner_actor.any():
                idx_l = torch.nonzero(is_learner_actor, as_tuple=False).squeeze(-1)
                out = net.forward_step(
                    cards=obs["cards"][idx_l],
                    scalars=obs["scalars"][idx_l],
                    action_mask=obs["action_mask"][idx_l],
                    h=h0[:, idx_l, :],
                    c=c0[:, idx_l, :],
                    terminated=None,
                    deterministic=False,
                )
                action_type[idx_l] = out.action_type.to(dtype=torch.int32)
                raise_bucket[idx_l] = out.raise_bucket

            # Snapshot opponent decisions (bot ignored here).
            snap_mask = is_opp_actor & opp_kind.eq(2)
            if snap_mask.any() and snapshot_filled > 0:
                for snap_id in torch.unique(opp_id[snap_mask]).tolist():
                    sid = int(snap_id)
                    m = snap_mask & opp_id.eq(sid)
                    if not m.any():
                        continue
                    idx_s = torch.nonzero(m, as_tuple=False).squeeze(-1)
                    out_s = snapshots[sid].forward_step(
                        cards=obs["cards"][idx_s],
                        scalars=obs["scalars"][idx_s],
                        action_mask=obs["action_mask"][idx_s],
                        h=snap_h0[:, idx_s, :],
                        c=snap_c0[:, idx_s, :],
                        terminated=None,
                        deterministic=True,
                    )
                    action_type[idx_s] = out_s.action_type.to(dtype=torch.int32)
                    raise_bucket[idx_s] = out_s.raise_bucket

            amounts = raise_bucket_to_amount(
                raise_bucket=raise_bucket,
                pot_total=obs["pot_total"],
                min_raise=obs["min_raise"],
                max_raise=obs["max_raise"],
            )
            amounts = torch.where(
                action_type.eq(c.ACTION_RAISE), amounts, torch.zeros_like(amounts)
            )
            obs = env.step(action_type, amounts)

    # Warmup
    for _ in range(int(cfg.warmup)):
        do_rollout()
    _sync(cfg.device)

    t0 = time.perf_counter()
    for _ in range(int(cfg.iters)):
        do_rollout()
    _sync(cfg.device)
    dt = time.perf_counter() - t0

    env_steps = int(cfg.iters) * int(cfg.rollout_steps) * int(cfg.num_envs)
    sps = env_steps / max(1e-9, dt)
    return {
        "elapsed_s": dt,
        "env_steps": env_steps,
        "sps": sps,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark league rollout throughput")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-envs", type=int, default=65536)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--snapshot-pool", type=int, default=32)
    p.add_argument("--snapshot-filled", type=int, default=32)
    p.add_argument("--rollout-snapshot-k", type=int, default=4)
    p.add_argument("--league-eta-selfplay", type=float, default=0.6)
    p.add_argument("--league-bot-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = BenchConfig(
        device=args.device,
        num_envs=int(args.num_envs),
        rollout_steps=int(args.rollout_steps),
        warmup=int(args.warmup),
        iters=int(args.iters),
        snapshot_pool=int(args.snapshot_pool),
        snapshot_filled=int(args.snapshot_filled),
        rollout_snapshot_k=int(args.rollout_snapshot_k),
        league_eta_selfplay=float(args.league_eta_selfplay),
        league_bot_prob=float(args.league_bot_prob),
        seed=int(args.seed),
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
    out_path = out_dir / f"league_rollout_{ts}_{cfg.device.replace(':', '-')}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    (Path("bench") / "latest_league_rollout.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("=== pokergpu league rollout benchmark ===")
    print(f"device: {cfg.device}")
    print(f"num_envs: {cfg.num_envs}")
    print(f"rollout_steps: {cfg.rollout_steps}")
    print(f"iters: {cfg.iters}")
    print(f"snapshot_filled: {cfg.snapshot_filled}")
    print(f"rollout_snapshot_k: {cfg.rollout_snapshot_k}")
    print(f"elapsed_s: {result['elapsed_s']:.6f}")
    print(f"env_steps: {result['env_steps']}")
    print(f"SPS: {result['sps']:.2f}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
