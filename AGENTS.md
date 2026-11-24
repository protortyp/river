# Agent Guide (River)

River (`protortyp/river`) is a GPU-first Heads-Up No-Limit Hold'em environment (NVIDIA Warp + CUDA) and RL training stack.

## Ground Rules

- **GPU-first hot paths:** avoid `.cpu()`, `.numpy()`, or host synchronizations inside rollout/env loops unless explicitly required for tests or benchmarks.
- **Correctness invariants:** chip conservation, uncalled bet returns, split pots, legal-action mask/bound consistency, and reliable hand boundaries via `terminated` / `episode_id` (env auto-resets on terminal transitions).
- **Benchmark perf changes:** run `uv run python bench/run.py` whenever modifying kernels, observations, legal actions, or sampling.
- **Telemetry never crashes training:** all `mlflow.*` calls go through `train/mlflow_logging.py` with broad exception catching and latch-disable.

## Architecture Summary

- **Env (`src/gpu_poker/env.py`, re-exported via `src/river/`):** `WarpPokerEnv` with SoA Warp `GameState`, Cactus Kev prime/flush LUT evaluator, per-env `reset_with_config(...)`, zero-copy `wp.to_torch(...)` views.
  - `obs["cards"]`: `[N, 7]` int32 (`-1` hidden)
  - `obs["scalars"]`: `[N, 17]` fp32 (position, stacks, bets, pot, to-call, stage, raises, aggressor, public history, `bb/starting_stack`, `effective_stack_bb/200`)
  - `obs["action_mask"]`: `[N, 4]` bool (`{fold, check, call, raise}`)
- **Policies (`src/gpu_poker/policy.py`, `src/gpu_poker/policy_transformer.py`):** both expose identical `forward_step` / `evaluate_step` contracts and 7 discrete pot-fraction raise buckets (`RAISE_BUCKET_FRACTIONS`).
  - `PokerPolicyNet`: per-seat LSTM `(h, c)` stored in rollout buffer.
  - `PokerTransformerPolicyNet`: encoder-only Transformer over up to 32 public-action tokens packed into `[1, B, 129]`.
- **Training (`train/`):**
  - `train/config.py`: `TrainConfig`, `TrainResult`, path/AMP helpers.
  - `train/buffer.py`: `RolloutBuffer` (fp32 ratio-sensitive tensors).
  - `train/policy_factory.py`: backbone construction, snapshot loading, token pushing.
  - `train/single_runner.py`: single-net self-play & NFSP-lite league PPO.
  - `train/triad_runner.py` + `train/triad.py`: AlphaStar triad (`main`, `me`, `le`) on partitioned `num_envs`, stochastic state-coherent opponents, periodic seat-swapped eval updating PFSP scores, per-role FIFO pool pruning, 3-file checkpoint save/resume.
  - `train/train.py`: Hydra CLI entrypoint and compatibility re-exports.
- **ONNX & Web (`train/export_onnx.py`, `web/`):** exports both LSTM and Transformer backbones with identical I/O signatures; `cd web && npm test` generates encoder + ONNX fixtures and verifies Python↔JS parity (`max diff < 1e-7`) and `onnxruntime-web` forward pass.

## Optional MLflow Tracking

Opt-in (`mlflow_enabled: false` by default). Configure `MLFLOW_TRACKING_URI` (and optional `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD`) in gitignored `.env` (see `.env.example`), then run with `mlflow_enabled=true`.

## Verification Commands

```bash
just check      # uv run ruff check . && uv run ruff format --check . && uv run ty check
just test       # uv run pytest
just test-web   # cd web && npm test
just bench      # uv run python bench/run.py
```
