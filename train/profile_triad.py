"""Kernel-level torch.profiler harness for the triad training loop.

Wraps the existing ``run`` / ``_run_impl_triad`` execution in a
``torch.profiler.profile`` context with the standard schedule:

    skip_first=1 update   -- ignore the cold-start update entirely
                              (CUDA kernels JIT-compile, autotune kicks in)
    wait=0
    warmup=1 update       -- prime caches, no recording
    active=1 update       -- the recorded one
    repeat=1

Trace is emitted to ``--out-dir`` (default ``/root/profile_trace``) as
both a Chrome Trace JSON (``trace.json``, viewable in chrome://tracing
or perfetto.dev) and a key-averages summary printed to stdout.

Usage:
    uv run python train/profile_triad.py --config-name=train_triad_v100 \\
        autotune=true total_env_steps=10000000 \\
        amp_dtype=bfloat16

All Hydra overrides after the script path go to the underlying
``train.train`` invocation. The script monkey-patches the per-update
loop to call ``prof.step()`` at the right boundary.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# This file lives in train/, the same package as train.py. Add repo root so
# `train.train` resolves whether run via `uv run python train/profile_triad.py`
# or `uv run python -m train.profile_triad`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

OUT_DIR = Path(os.environ.get("PROFILE_OUT_DIR", "/root/profile_trace"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _on_trace_ready(prof: torch.profiler.profile) -> None:
    """Save a TEXT summary first (cheap and robust), then attempt the chrome
    trace dump. Print everything to stdout AND write to disk so we don't
    lose it if the process exits during the JSON export."""
    summary_self = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    summary_tot = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    text = (
        "\n[profile] top-30 ops by self_cuda_time_total:\n"
        + summary_self
        + "\n\n[profile] top-30 ops by cuda_time_total (includes children):\n"
        + summary_tot
        + "\n"
    )
    print(text, flush=True)
    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text(text)
    print(f"[profile] summary -> {summary_path}", flush=True)

    # Now try the (much larger, slower, error-prone) chrome trace. If this
    # fails or gets killed mid-write, we still have the text summary.
    trace_path = OUT_DIR / "trace.json"
    try:
        prof.export_chrome_trace(str(trace_path))
        print(f"[profile] trace -> {trace_path}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[profile] chrome trace export failed: {e!r}", flush=True)


def install_profiler_into_triad() -> torch.profiler.profile:
    """Monkey-patch ``_run_impl_triad`` to call ``prof.step()`` per update.

    We hook by wrapping the existing tqdm ``pbar.update`` call (which fires
    exactly once per update at line ~2851 of train.py after the rollout).
    """
    import train.train as tmod

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(skip_first=1, wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=_on_trace_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    _orig_run_impl_triad = tmod._run_impl_triad

    def patched(cfg):  # type: ignore[no-untyped-def]
        # Patch the global tqdm reference so its ``update()`` advances the
        # profiler. Done before _run_impl_triad runs.
        original_tqdm = tmod.tqdm

        class _TqdmWithProfStep(original_tqdm):  # type: ignore[misc, valid-type]
            def update(self, n: int = 1):  # noqa: D401
                ret = super().update(n)
                prof.step()
                return ret

        tmod.tqdm = _TqdmWithProfStep  # type: ignore[assignment]
        try:
            with prof:
                return _orig_run_impl_triad(cfg)
        finally:
            tmod.tqdm = original_tqdm  # type: ignore[assignment]

    tmod._run_impl_triad = patched  # type: ignore[assignment]
    return prof


def main() -> int:
    print(f"[profile] output dir: {OUT_DIR}", flush=True)
    install_profiler_into_triad()
    # Delegate the rest to train.train's main() so all Hydra plumbing,
    # config composition, etc. still works exactly as the user expects.
    from train.train import main as train_main

    t0 = time.perf_counter()
    rc = train_main()
    print(f"[profile] train exited rc={rc} after {time.perf_counter() - t0:.1f}s", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
