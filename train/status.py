"""Print the current training status from the latest run's TensorBoard log.

Usage (from the repo root):
    uv run python train/status.py
    uv run python train/status.py --run train/outputs/2026-05-16/15-28-40
    uv run python train/status.py --last-n 5      # show last N eval data points
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path


def _find_latest_run(outputs_root: Path) -> Path:
    runs = []
    for day_dir in outputs_root.iterdir():
        if not day_dir.is_dir():
            continue
        for time_dir in day_dir.iterdir():
            if not time_dir.is_dir():
                continue
            tb_dir = time_dir / "train" / "tensorboard"
            if tb_dir.is_dir():
                runs.append(time_dir)
    if not runs:
        raise SystemExit(f"No runs found under {outputs_root}")
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs[-1]


def _scalar(acc, tag):
    try:
        return acc.Scalars(tag)
    except KeyError:
        return None


def _last(acc, tag, fmt="{:.4g}"):
    events = _scalar(acc, tag)
    if not events:
        return None
    return fmt.format(events[-1].value)


def _series(acc, tag, n):
    events = _scalar(acc, tag)
    if not events:
        return None
    return [round(e.value, 3) for e in events[-n:]]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        type=Path,
        help="Path to a specific train/outputs/<day>/<time> dir. Defaults to latest.",
    )
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("train/outputs"),
        help="Where to look for runs (default: train/outputs).",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=5,
        help="Show the last N eval data points per metric (default: 5).",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=113664,
        help="Used only to estimate update#; defaults to V100 autotune value.",
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=64,
    )
    args = p.parse_args()

    # Lazy import so the script works in any env that has tensorboard.
    from tensorboard.backend.event_processing import event_accumulator as ea

    run_dir = args.run if args.run else _find_latest_run(args.outputs_root)
    tb_files = sorted(glob.glob(str(run_dir / "train" / "tensorboard" / "events*")))
    if not tb_files:
        raise SystemExit(f"No tensorboard events in {run_dir}")
    print(f"run: {run_dir}")
    print(f"tb : {os.path.basename(tb_files[0])}")

    acc = ea.EventAccumulator(tb_files[0], size_guidance={ea.SCALARS: 100_000})
    acc.Reload()

    sps = _scalar(acc, "train/overall_sps")
    if not sps:
        print("(no scalars yet — training just started?)")
        return 0

    steps = sps[-1].step
    steps_per_update = args.num_envs * args.rollout_steps
    update = steps // max(1, steps_per_update)
    target_steps = 4_000_000_000
    pct = 100.0 * steps / target_steps
    print()
    print(f"step      : {steps:>14,d}  (update~{update}, {pct:.1f}% of 4B target)")
    print(f"sps       : {int(sps[-1].value):>14,d}")
    print()

    print("--- training ---")
    for tag in [
        "train/loss",
        "train/value_loss",
        "train/entropy",
        "train/explained_variance",
        "train/approx_kl",
        "train/clipfrac",
    ]:
        v = _last(acc, tag)
        if v is not None:
            print(f"  {tag:32s} {v}")

    print()
    print("--- league composition ---")
    for tag in [
        "league/rollout_bot_frac",
        "league/rollout_snap_frac",
        "league/rollout_selfplay_frac",
    ]:
        v = _last(acc, tag)
        if v is not None:
            print(f"  {tag:32s} {v}")

    print()
    print("--- action distribution ---")
    for tag in [
        "actions/pct_fold",
        "actions/pct_check",
        "actions/pct_call",
        "actions/pct_raise",
    ]:
        v = _last(acc, tag)
        if v is not None:
            print(f"  {tag:32s} {v}")

    print()
    print("--- raise bucket distribution (% of raises) ---")
    labels = ["min", "0.5p", "0.75p", "p", "1.5p", "2.5p", "allin"]
    for b in range(7):
        v = _last(acc, f"actions/raise_bucket_{b}_pct")
        if v is not None:
            print(f"  bucket {b} ({labels[b]:>5s})           {v}")

    print()
    print(f"--- eval evolution (last {args.last_n}) ---")
    for tag in [
        "eval/pool_mix_bb_per_hand",
        "eval/nit_bb_per_hand",
        "eval/calling_station_bb_per_hand",
        "eval/loose_aggressive_bb_per_hand",
        "eval/loose_passive_bb_per_hand",
        "eval/random_aggressive_bb_per_hand",
    ]:
        s = _series(acc, tag, args.last_n)
        if s is None:
            continue
        print(f"  {tag:45s} {s}")

    print()
    print(f"--- self-play / snapshot (last {args.last_n}) ---")
    for tag in [
        "selfplay/vs_latest_bb_per_hand",
        "selfplay/bb_per_hand_mean",
        "selfplay/bb_per_hand_max",
        "selfplay/bb_per_hand_min",
        "league/snapshot_bb_per_hand_mean",
    ]:
        s = _series(acc, tag, args.last_n)
        if s is None:
            continue
        print(f"  {tag:45s} {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
