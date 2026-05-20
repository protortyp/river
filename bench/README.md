# Benchmarks

Primary entrypoint:
- `uv run python bench/run.py`

Outputs:
- `bench/latest.json` (overwritten)
- `bench/results/bench_<timestamp>_<host>.json` (appended)

Environment overrides (optional):
- `POKERGPU_BENCH_NUM_ENVS` (default: `131072`)
- `POKERGPU_BENCH_WARMUP` (default: `50`)
- `POKERGPU_BENCH_STEPS` (default: `500`)

Notes:
- The first run will include Warp JIT compile time (cached afterwards).
- `bench/run.py` includes both end-to-end `env.step()` cases and “pure kernel” cases that avoid Torch→Warp copies and can generate actions on-GPU.

