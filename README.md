# PokerGPU

GPU-first heads-up no-limit Hold'em environment for reinforcement learning.

The environment is built around Warp kernels with Torch zero-copy views so large
batches of poker hands can be stepped from training code with minimal host
round-trips.

## Quick Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Benchmarks And Smoke Runs

```bash
uv run python bench/run.py
uv run python train/torchrl_smoke.py
uv run python train/train.py
uv run python train/train.py league_enabled=true
```

`bench/run.py` writes timestamped JSON under `bench/results/` and updates
`bench/latest.json`. Training runs write Hydra `results.json` files and update
`train/latest_train.json`.

## Useful Docs

- [Performance and roadmap](docs/performance_and_roadmap.md)
- [Training plan](docs/training_plan.md)
- [Session handoff](docs/session_handoff.md)
