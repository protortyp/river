# River

GPU-first Heads-Up No-Limit Hold'em (HUNL) environment in [NVIDIA Warp](https://github.com/NVIDIA/warp) with zero-copy PyTorch tensors, Regularized Nash Dynamics (R-NaD) + AlphaStar Triad RL training, ReBeL Turn/River DCFR subgame search, Slumbot evaluation, and a browser-playable ONNX client.

## Throughput

Measured on a single GPU (`131,072` parallel environments, `uv run python bench/run.py`):

| Benchmark | RTX 4090 | RTX 3090 |
|---|---:|---:|
| Warp step kernel (no obs/legal) | **1.38B SPS** | **536.6M SPS** |
| Warp kernels (step + obs + legal, constant / random) | **854.2M / 440.1M SPS** | **325.6M / 192.3M SPS** |
| `WarpPokerEnv.step()` via PyTorch (constant / random) | **516.9M / 431.2M SPS** | **280.7M / 185.4M SPS** |

## Benchmarks (200 BB HUNL, 50/100)

Trained 48× suit-canonical Card-Token Transformer blueprints with Regularized Nash Dynamics to **`4.50B` env steps** — `rnad_45b.pt` (`conf/train_rnad.yaml`, `3.2M` params) and `rnad_v3_45b.pt` (`conf/train_rnad_v3.yaml`, `11.3M` params + CTDE 52-card Bayesian opponent hand-reading head, `ev=+0.258`) — paired with an exact `52!`-card-permutation-equivariant, blocker-aware `RiverCFVNet v2` (`4.57M` params, **`1.46% pot`** reach-weighted val RMSE across 60,000 exact 4-size Nash river equilibria — `11.3×` lower than check-down equity `16.54%`) and calibrated 4-size `(0.33, 0.75, 1.5, all-in)` nested DCFR subgame search:

| Benchmark | Result | Sample |
|---|---:|---:|
| **Neural Best-Response Exploitability** (`rnad_45b.pt`, 50 BR updates / `52.4M` BR steps; `num_envs=32768`) | **`-90.74 bb/100`** (`-907.4 mbb/g`) | 40,320 duplicate hands |
| **Apples-to-Apples Neural BR** (`50` BR updates, `num_envs=16384`): `rnad_45b.pt` (`3.2M`) / `rnad_v3_45b.pt` (`11.3M`) | **`-41.43`** / **`-27.62 bb/100`** | ~40,270 dup hands each |
| **Duplicate H2H Scaling (`v2`):** `4.50B` vs `3.93B` / `4.50B` vs `2.25B` / `3.93B` vs `2.25B` | **`+5.84`** / **`+7.05`** / **`+7.45 bb/100`** | ~40,400 dup hands each |
| **Duplicate H2H Scaling (`v3`, `200 BB`):** `4.50B` vs `3.15B` / `2.10B` / `1.05B` (`v3` vs `v2` `4.50B`: `-0.26`) | **`+11.93`** / **`+14.06`** / **`+14.23 bb/100`** | ~40,500 dup hands each |
| **vs. Slumbot — Calibrated 4-Size River DCFR** (combined `576M`–`4.50B`, 0 errors) | **`+7.12 ± 9.95 bb/100`** (`+71.2 ± 99.5 mbb/g`) | 29,000 hands |
| **vs. Slumbot — Decoupled ReBeL (`--turn-max-raises 2`, `RiverCFVNet`)** (combined `rnad_45b` 5k + `rnad_v3_45b` 10k, 0 errors; `+27.76 bb/100` on 14,859 `<100bb` hands) | **`+5.35 ± 16.65 bb/100`** (`+53.5 ± 166.5 mbb/g`) | 14,998 hands |
| **vs. Slumbot — Locked `rnad_v3_45b` + `RiverCFVNet v2` Decoupled ReBeL** (`0` errors; `pf +0.7 \| fl +9.8 \| tu +25.5 \| ri -39.2`; `+25.58 bb/100` on 9,900 `<100bb` hands) | **`-3.17 ± 20.99 bb/100`** (`-31.7 ± 209.9 mbb/g`) | 10,000 hands |
| **vs. Slumbot — All Calibrated Search Runs Combined** (`29k` River DCFR + `15k` PR #28 ReBeL, 0 errors) | **`+6.52 ± 8.55 bb/100`** (`+65.2 ± 85.5 mbb/g`) | 43,998 hands |

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). CUDA GPU optional (Warp falls back to CPU for development/tests).

```bash
git clone https://github.com/protortyp/river.git
cd river
uv sync --group dev
```

## Quickstart

```python
import torch
from river import WarpPokerEnv

env = WarpPokerEnv(num_envs=131_072, starting_stack=20_000, small_blind=50, big_blind=100, device="cuda:0")
obs = env.reset()

# obs["cards"]: [N, 7] int32   (2 hole + 5 board, -1 if hidden)
# obs["scalars"]: [N, 17] fp32 (stacks, pot, bets, stage, aggression, stack depth)
# obs["action_mask"]: [N, 4] bool ({fold, check, call, raise})
actions = torch.full((env.num_envs,), 2, device="cuda:0", dtype=torch.int32)
out = env.step(actions)
```

## Training

```bash
# Regularized Nash Dynamics (R-NaD) + suit-canonical Card-Token Transformer + CTDE critic
uv run python train/train.py --config-name=train_rnad

# AlphaStar Triad (Main + Main Exploiter + League Exploiter)
uv run python train/train.py --config-name=train_triad_v100

# Single-agent self-play / league PPO
uv run python train/train.py

# Override any Hydra parameter on the CLI
uv run python train/train.py --config-name=train_rnad device=cpu num_envs=64 rollout_steps=8 total_env_steps=1024
```

## Evaluation & Play

```bash
# Measure empirical best-response exploitability (duplicate-deck bb/100)
uv run python train/exploitability.py --checkpoint checkpoints/rnad/rnad_final.pt --updates 100 --eval-hands 10000

# Export a trained checkpoint to ONNX (LSTM or Transformer, verified against PyTorch)
uv run python train/export_onnx.py --checkpoint path/to/ckpt.pt --out model.onnx

# Benchmark against Slumbot (50/100, 200 BB; add --river-search and optional --turn-search --cfv-net)
uv run python train/slumbot_eval.py --checkpoint path/to/ckpt.pt --config conf/train_rnad.yaml --hands 3000 --river-search --search-iters 100 --workers 10

# Train exact 52!-permutation-equivariant ReBeL RiverCFVNet on exact Nash river equilibria
uv run python train/train_river_cfv.py --num-samples 25000 --gen-device cpu --gen-workers 16 --device cuda:0 --out checkpoints/river_cfv_v1.pt

# Play in the terminal against the latest snapshot
uv run python play/interactive.py

# Play in the browser (drop any exported .onnx file)
cd web && npm ci && npm run dev
```

## Repository Layout

| Path | Contents |
|---|---|
| `src/gpu_poker/` | Warp SoA `GameState`, CUDA kernels (`state`, `actions`, `legal_actions`, `observations`, `evaluator`), `WarpPokerEnv`, `PokerPolicyNet` (LSTM), `PokerTransformerPolicyNet`, `RiverDCFRSolver` / `RiverSubgameSession` (`search.py`), `RiverCFVNet` (`cfv_net.py`), TorchRL wrapper |
| `src/river/` | Public top-level re-exports (`from river import WarpPokerEnv, ...`) |
| `train/` | R-NaD (`rnad_runner`), PPO + League/PFSP + AlphaStar Triad (`single_runner`, `triad_runner`, `triad`, `league`), ReBeL CFV generator/trainer (`train_river_cfv`), neural BR `exploitability`, `eval`, `export_onnx`, `slumbot_eval` |
| `conf/` | Hydra configs (`train.yaml`, `train_rnad.yaml`, `train_rnad_v3.yaml`, `train_triad_v100.yaml`) |
| `bench/` | GPU throughput benchmarks and kernel/training profilers |
| `web/` | Static Svelte 5 + `onnxruntime-web` browser client with Python↔JS encoder parity tests |
| `play/` | Interactive Rich terminal UI |
| `docs/` | [Architecture & Profiling](docs/architecture.md) |

## Development

```bash
just check      # ruff lint + format check + ty type check
just test       # pytest suite (176 tests)
just test-web   # JS encoder/engine parity + onnxruntime-web load/forward tests
just bench      # GPU environment benchmark
```

## License

[MIT](LICENSE)
