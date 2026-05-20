"""Query training progress (TensorBoard + Hydra outputs).

This is intentionally "copy/paste friendly" for sharing training status.

Usage:
  # Use latest Hydra run under `train/outputs/`
  uv run python train/query_training.py

  # Point at a specific Hydra run dir or a TensorBoard dir
  uv run python train/query_training.py --run train/outputs/2025-12-12/16-08-35

  # Show more history
  uv run python train/query_training.py --n 25

  # Dump available scalar tags
  uv run python train/query_training.py --list-tags
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    results_json: Path | None
    tb_dir: Path | None


def _find_repo_root(start: Path) -> Path | None:
    cur = start.resolve()
    for p in (cur, *cur.parents):
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
    return None


def _find_latest_hydra_run(*, outputs_dir: Path) -> Path | None:
    if not outputs_dir.exists():
        return None

    # Candidate sources:
    # - Hydra run dirs that have `results.json`
    # - Hydra run dirs that only have TensorBoard event files
    candidates: list[tuple[float, Path]] = []

    for results_path in outputs_dir.rglob("results.json"):
        if results_path.is_file():
            candidates.append((results_path.stat().st_mtime, results_path.parent))

    for ev in outputs_dir.rglob("events.out.tfevents.*"):
        if not ev.is_file():
            continue
        # Typical layout:
        #   train/outputs/YYYY-MM-DD/HH-MM-SS/train/tensorboard/events...
        # run_dir is 2 levels above `train/`.
        try:
            train_dir = ev.parents[1]  # .../train
            run_dir = train_dir.parent  # .../HH-MM-SS
        except IndexError:
            continue
        candidates.append((ev.stat().st_mtime, run_dir))

    if not candidates:
        return None

    # Pick the most recently modified, tie-breaking on the run-dir path so the
    # result is deterministic when two runs share an mtime. Coarse-resolution
    # filesystems (e.g. overlayfs) can stamp same-second writes identically;
    # Hydra run dirs are named `YYYY-MM-DD/HH-MM-SS`, which sorts chronologically.
    candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return candidates[0][1]


def _find_tensorboard_dir(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None

    def has_events(p: Path) -> bool:
        return any(p.glob("events.out.tfevents.*"))

    if has_events(run_dir):
        return run_dir

    for rel in ("tensorboard", "train/tensorboard"):
        cand = run_dir / rel
        if cand.exists() and has_events(cand):
            return cand

    # Fallback: find first event file within a small depth.
    for ev in run_dir.rglob("events.out.tfevents.*"):
        try:
            rel = ev.relative_to(run_dir)
        except ValueError:
            continue
        if len(rel.parts) <= 4:
            return ev.parent

    return None


def _resolve_run_paths(run: str | None) -> RunPaths:
    # Prefer the current working directory (works when running from anywhere inside
    # the repo), and fall back to this file's location.
    repo_root = _find_repo_root(Path.cwd()) or _find_repo_root(Path(__file__).resolve())  # type: ignore[arg-type]
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    outputs_dir = repo_root / "train" / "outputs"
    latest_json = repo_root / "train" / "latest_train.json"
    default_tb = repo_root / "train" / "tensorboard"

    if run is None:
        run_dir = _find_latest_hydra_run(outputs_dir=outputs_dir)
        if run_dir is None and latest_json.exists():
            # No Hydra run dirs yet, but we have a repo-level latest pointer.
            return RunPaths(
                run_dir=repo_root,
                results_json=latest_json,
                tb_dir=_find_tensorboard_dir(default_tb),
            )

        if run_dir is None and default_tb.exists():
            # No Hydra run dirs yet; allow querying pure TB logs.
            return RunPaths(
                run_dir=repo_root,
                results_json=None,
                tb_dir=_find_tensorboard_dir(default_tb),
            )

        if run_dir is None:
            # Keep this non-fatal and actionable.
            raise FileNotFoundError(
                f"No runs found under {outputs_dir}. Run `uv run python train/train.py` first, "
                f"or pass --run <path> (e.g. a Hydra run dir), or ensure TensorBoard logs exist "
                f"under a run dir like {outputs_dir}/.../train/tensorboard/."
            )
    else:
        run_path = Path(run).expanduser().resolve()
        if run_path.is_file() and run_path.name.endswith(".json"):
            # Allow pointing directly at a results json (or latest_train.json).
            return RunPaths(
                run_dir=run_path.parent,
                results_json=run_path,
                tb_dir=_find_tensorboard_dir(run_path.parent),
            )
        run_dir = run_path

    results_json = (run_dir / "results.json") if (run_dir / "results.json").exists() else None
    tb_dir = _find_tensorboard_dir(run_dir)
    return RunPaths(run_dir=run_dir, results_json=results_json, tb_dir=tb_dir)


def _load_results_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_tensorboard_scalars(tb_dir: Path) -> dict[str, dict[str, list[float]]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as e:  # pragma: no cover
        raise ModuleNotFoundError("tensorboard is not installed (try `uv add tensorboard`)") from e

    ea = event_accumulator.EventAccumulator(str(tb_dir))
    ea.Reload()

    data: dict[str, dict[str, list[float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [float(e.step) for e in events],
            "values": [float(e.value) for e in events],
        }
    return data


def _recent(values: list[float], n: int) -> list[float]:
    if n <= 0:
        return []
    return values[-n:] if len(values) > n else values


def _print_table(
    *,
    rows: list[tuple[str, list[int], list[float]]],
    col_steps: list[int],
) -> None:
    if not col_steps:
        return

    # Align every row by step, otherwise a shared TensorBoard directory can make
    # "last N" values across tags misleading.
    colw = 12
    print(f"{'metric':{colw}s}  " + "  ".join([f"U{u:>5d}" for u in col_steps]))
    print(f"{'-' * colw}  " + "  ".join(["-" * 6 for _ in col_steps]))

    def fmt_value(name: str, v: float | None) -> str:
        if v is None:
            return "   -  "
        if name in {
            "kl",
            "klabs",
            "nf_old",
            "nf_new",
            "nf_lr",
            "lp_old_min",
            "lp_old_max",
            "lp_new_min",
            "lp_new_max",
            "lr_min",
            "lr_max2",
        }:
            return f"{v:6.6f}"
        return f"{v:6.3f}"

    for name, steps, vals in rows:
        vals_by_step = {steps[i]: vals[i] for i in range(min(len(steps), len(vals)))}
        fmt = "  ".join([fmt_value(name, vals_by_step.get(s)) for s in col_steps])
        print(f"{name:{colw}s}  {fmt}")


def summarize(
    *,
    tb: dict[str, dict[str, list[float]]] | None,
    results: dict | None,
    n: int,
    samples: int = 0,
) -> None:
    print("=" * 80)
    print("pokergpu training summary")
    print("=" * 80)

    if results is not None:
        meta = results.get("meta", {})
        cfg = results.get("result", {}).get("config", {})
        print(f"time_utc: {meta.get('timestamp_utc', 'unknown')}")
        print(f"host:     {meta.get('host', 'unknown')}")
        print(f"torch:    {meta.get('torch', 'unknown')}")
        if cfg:
            print(
                "config:   "
                + ", ".join(
                    [
                        f"device={cfg.get('device')}",
                        f"num_envs={cfg.get('num_envs')}",
                        f"rollout_steps={cfg.get('rollout_steps')}",
                        f"num_updates={cfg.get('num_updates')}",
                        f"num_epochs={cfg.get('num_epochs')}",
                        f"minibatch_size={cfg.get('minibatch_size')}",
                        f"lr={cfg.get('lr')}",
                    ]
                )
            )
        print()

    if tb is None or not tb:
        print("No TensorBoard scalars found for this run.")
        if results is not None:
            res = results.get("result", {})
            print()
            print("Final metrics (results.json):")
            for k in (
                "overall_sps",
                "mean_loss",
                "approx_kl",
                "clipfrac",
                "entropy",
                "invalid_rate",
            ):
                if k in res:
                    print(f"  {k}: {res[k]}")
        print("=" * 80)
        return

    def get(tag: str) -> tuple[list[int], list[float]] | None:
        if tag not in tb:
            return None
        steps_f = tb[tag]["steps"]
        vals = tb[tag]["values"]
        steps = [int(s) for s in steps_f]
        return steps, vals

    # Use loss tag as the "clock" if present.
    clock = get("train/loss") or get("train/overall_sps")
    if clock is None:
        print("No known train/* scalar tags found.")
        print(f"tags: {sorted(tb.keys())}")
        print("=" * 80)
        return

    steps, _ = clock
    # These are TensorBoard step indices; in `train/train.py` we use environment
    # steps (`env_steps`) as the step axis.
    last_step = steps[-1] if steps else 0
    print(f"env_steps points: {len(steps)} (latest={last_step})")
    print()

    rows: list[tuple[str, list[int], list[float]]] = []
    for name, tag in [
        ("loss", "train/loss"),
        ("vloss", "train/value_loss"),
        ("kl", "train/approx_kl"),
        ("klabs", "train/approx_kl_abs"),
        ("clip", "train/clipfrac"),
        ("ent", "train/entropy"),
        ("ev", "train/explained_variance"),
        ("inv", "train/invalid_rate"),
        ("nf_old", "debug/logprob_old_nonfinite_rate"),
        ("nf_new", "debug/logprob_new_nonfinite_rate"),
        ("nf_lr", "debug/log_ratio_nonfinite_rate"),
        ("lr_abs", "debug/log_ratio_abs_mean"),
        ("lr_max", "debug/log_ratio_abs_max"),
        ("lp_old_min", "debug/logprob_old_min"),
        ("lp_old_max", "debug/logprob_old_max"),
        ("lp_new_min", "debug/logprob_new_min"),
        ("lp_new_max", "debug/logprob_new_max"),
        ("lr_min", "debug/log_ratio_min"),
        ("lr_max2", "debug/log_ratio_max"),
        ("sps_k", "train/overall_sps"),
        # Optional phase timings (seconds).
        ("t_col", "time/collect_s"),
        ("t_opt", "time/opt_s"),
        ("t_eval", "time/eval_s"),
        ("t_lg", "time/league_eval_s"),
        ("t_sp", "time/selfplay_eval_s"),
        ("t_misc", "time/misc_s"),
        ("t_tot", "time/update_total_s"),
    ]:
        got = get(tag)
        if got is None:
            continue
        tag_steps, vals = got
        if name == "sps_k":
            vals = [v / 1000.0 for v in vals]
        rows.append((name, tag_steps, vals))

    # Column selection: either last N points or evenly spaced samples across the run.
    col_steps: list[int] = []
    if samples and samples > 0 and steps:
        idx = np.linspace(0, len(steps) - 1, int(samples)).round().astype(int)
        idx = [int(i) for i in idx]
        # Unique, ordered.
        seen: set[int] = set()
        for i in idx:
            if i not in seen:
                seen.add(i)
                col_steps.append(steps[i])
        # Ensure endpoints.
        if col_steps and col_steps[0] != steps[0]:
            col_steps = [steps[0], *col_steps]
        if col_steps and col_steps[-1] != steps[-1]:
            col_steps = [*col_steps, steps[-1]]
    elif n > 0 and steps:
        col_steps = steps[-min(n, len(steps)) :]
    else:
        col_steps = steps

    _print_table(rows=rows, col_steps=col_steps)
    print()

    # Eval: support multiple bots, discovered by tag name.
    bot_names: list[str] = []
    for tag in tb:
        if not tag.startswith("eval/"):
            continue
        if not tag.endswith("_bb_per_hand"):
            continue
        name = tag[len("eval/") : -len("_bb_per_hand")]
        if name:
            bot_names.append(name)
    bot_names = sorted(set(bot_names))

    for bot in bot_names:
        eval_bb = get(f"eval/{bot}_bb_per_hand")
        if eval_bb is None:
            continue
        ev_steps, ev_vals = eval_bb
        recent = _recent(ev_vals, min(5, len(ev_vals)))
        mean_recent = float(np.mean(recent))
        print(f"eval/{bot} bb/hand (last {len(recent)}): mean={mean_recent:.4f}")
        hist_n = min(10, len(ev_vals))
        if hist_n > 0:
            print(f"eval/{bot} history:")
            bb100 = get(f"eval/{bot}_bb_per_100")
            hands = get(f"eval/{bot}_hands")
            bb100_vals = bb100[1] if bb100 is not None else None
            hands_vals = hands[1] if hands is not None else None
            for i, (s, v) in enumerate(zip(ev_steps[-hist_n:], ev_vals[-hist_n:], strict=True)):
                idx = len(ev_vals) - hist_n + i
                bb100_i = (
                    bb100_vals[idx] if bb100_vals is not None and idx < len(bb100_vals) else None
                )
                hands_i = (
                    hands_vals[idx] if hands_vals is not None and idx < len(hands_vals) else None
                )
                extra = ""
                if bb100_i is not None:
                    extra += f" bb/100={bb100_i:.2f}"
                if hands_i is not None:
                    extra += f" hands={int(hands_i)}"
                print(f"  env_steps={int(s)} bb/hand={v:.4f}{extra}")

    # Convenience: show pool-mix eval if present (treated like a "bot" tag).
    pool_mix = get("eval/pool_mix_bb_per_hand")
    if pool_mix is not None and "pool_mix" not in bot_names:
        ev_steps, ev_vals = pool_mix
        recent = _recent(ev_vals, min(5, len(ev_vals)))
        mean_recent = float(np.mean(recent))
        print()
        print(f"eval/pool_mix bb/hand (last {len(recent)}): mean={mean_recent:.4f}")

    # Timing: highlight last non-zero eval/league/selfplay timing entries since they can be sparse.
    def _print_last_nonzero(tag: str, label: str) -> None:
        got = get(tag)
        if got is None:
            return
        st, vals = got
        pairs = [(int(s), float(v)) for s, v in zip(st, vals, strict=True)]
        nz = [(s, v) for (s, v) in pairs if v > 0.0]
        if not nz:
            return
        s, v = nz[-1]
        print(f"{label}: last_nonzero env_steps={s} value_s={v:.6f}")

    print()
    _print_last_nonzero("time/eval_s", "time/eval_s")
    _print_last_nonzero("time/league_eval_s", "time/league_eval_s")
    _print_last_nonzero("time/selfplay_eval_s", "time/selfplay_eval_s")

    # Self-play snapshot eval (if enabled).
    sp_mean = get("selfplay/bb_per_hand_mean")
    if sp_mean is not None:
        sp_steps, sp_vals = sp_mean
        recent = _recent(sp_vals, min(5, len(sp_vals)))
        mean_recent = float(np.mean(recent))
        print()
        print(f"selfplay/snapshot bb/hand (last {len(recent)}): mean={mean_recent:.4f}")
        hist_n = min(10, len(sp_vals))
        if hist_n > 0:
            sp_min = get("selfplay/bb_per_hand_min")
            sp_max = get("selfplay/bb_per_hand_max")
            sp_latest = get("selfplay/vs_latest_bb_per_hand")
            sp_min_vals = sp_min[1] if sp_min is not None else None
            sp_max_vals = sp_max[1] if sp_max is not None else None
            sp_latest_vals = sp_latest[1] if sp_latest is not None else None
            print("selfplay/snapshot history:")
            for i, (s, v) in enumerate(zip(sp_steps[-hist_n:], sp_vals[-hist_n:], strict=True)):
                idx = len(sp_vals) - hist_n + i
                extra = ""
                if sp_min_vals is not None and idx < len(sp_min_vals):
                    extra += f" min={sp_min_vals[idx]:.4f}"
                if sp_max_vals is not None and idx < len(sp_max_vals):
                    extra += f" max={sp_max_vals[idx]:.4f}"
                if sp_latest_vals is not None and idx < len(sp_latest_vals):
                    extra += f" vs_latest={sp_latest_vals[idx]:.4f}"
                print(f"  env_steps={int(s)} mean={v:.4f}{extra}")

    print("=" * 80)


def main() -> int:
    p = argparse.ArgumentParser(description="Query pokergpu training progress")
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help="Hydra run dir or TB dir (default: latest)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of recent env_steps points to show in the main table (default: 10)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Show this many evenly spaced env_steps points from start->end (overrides --n)",
    )
    p.add_argument("--list-tags", action="store_true", help="Print available scalar tags and exit")
    args = p.parse_args()

    paths = _resolve_run_paths(args.run)
    results = _load_results_json(paths.results_json) if paths.results_json else None

    tb = None
    if paths.tb_dir is not None:
        tb = _load_tensorboard_scalars(paths.tb_dir)
        if args.list_tags:
            print(f"run_dir: {paths.run_dir}")
            print(f"tb_dir:  {paths.tb_dir}")
            print("\n".join(sorted(tb.keys())))
            return 0

    summarize(tb=tb, results=results, n=args.n, samples=int(args.samples or 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
