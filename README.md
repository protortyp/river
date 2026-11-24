# River

GPU-first Heads-Up No-Limit Hold'em (HUNL) environment in [NVIDIA Warp](https://github.com/NVIDIA/warp) with zero-copy PyTorch tensors, AlphaStar-style Triad RL training, Slumbot evaluation, and a browser-playable ONNX client.

## Throughput

Measured on a single GPU (`131,072` parallel environments, `uv run python bench/run.py`):

| Benchmark | RTX 4090 | RTX 3090 |
|---|---:|---:|
| Warp step kernel (no obs/legal) | **1.38B SPS** | **536.6M SPS** |
| Warp kernels (step + obs + legal, constant / random) | **854.2M / 440.1M SPS** | **325.6M / 192.3M SPS** |
| `WarpPokerEnv.step()` via PyTorch (constant / random) | **516.9M / 431.2M SPS** | **280.7M / 185.4M SPS** |

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

# Benchmark against Slumbot (50/100, 200 BB; add --river-search for online DCFR river solving)
uv run python train/slumbot_eval.py --model model.onnx --hands 1000 --stochastic --river-search

# Play in the terminal against the latest snapshot
uv run python play/interactive.py

# Play in the browser (drop any exported .onnx file)
cd web && npm ci && npm run dev
```

## Repository Layout

| Path | Contents |
|---|---|
| `src/gpu_poker/` | Warp SoA `GameState`, CUDA kernels (`state`, `actions`, `legal_actions`, `observations`, `evaluator`), `WarpPokerEnv`, `PokerPolicyNet` (LSTM), `PokerTransformerPolicyNet`, TorchRL wrapper |
| `src/river/` | Public top-level re-exports (`from river import WarpPokerEnv, ...`) |
| `train/` | Modular PPO + League/PFSP + AlphaStar Triad trainer (`config`, `buffer`, `policy_factory`, `single_runner`, `triad_runner`, `triad`, `league`, `eval`, `bots`, `export_onnx`, `slumbot_eval`) |
| `conf/` | Hydra configs (`train.yaml`, `train_triad_v100.yaml`) |
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
