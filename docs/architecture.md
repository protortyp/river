# Architecture & Profiling

## 1. GPU Environment (`src/gpu_poker/`)

`WarpPokerEnv` stores all `N` parallel HUNL tables in a Structure-of-Arrays Warp struct (`GameState` in `src/gpu_poker/struct_types.py`).

- **Zero-copy outputs:** `reset()` and `step()` return PyTorch tensor views backed by Warp GPU buffers (`wp.to_torch`). Callers must copy observations into rollout storage before calling `step()` again.
- **Auto-reset & episode boundaries:** terminal hands settle chips and reset in-kernel; `terminated` (`episode_id` changed) is the authoritative hand boundary.
- **Per-env stakes:** `reset_with_config(starting_stack=..., small_blind=..., big_blind=...)` supports per-env randomized stack depths (e.g., 40–200 BB) persisted across auto-resets.
- **Hand evaluator (`kernels/evaluator.py`):** Cactus Kev prime-product (`unsuited_lut`) and rank-bitmask (`flush_lut`) 5-card lookup tables from `lookup_data/hand_ranks.npz`, 21-way unrolled for 7-card evaluation. Tables are cached once per device in `WarpPokerEnv._LOOKUP_CACHE`.

### Observation Schema

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `cards` | `[N, 7]` | `int32` | 2 acting-player hole cards + 5 board slots (`-1` if unrevealed) |
| `scalars` | `[N, 17]` | `float32` | `0`: position (`0=SB/BTN`, `1=BB`), `1..2`: stacks/start, `3..4`: bets/pot, `5`: pot/(2·start), `6`: to_call/stack, `7`: stage/5, `8`: last_raise/start, `9`: num_raises/4, `10`: aggressor, `11..14`: street actions & last action, `15`: `bb/start`, `16`: `min(eff_bb/200, 1)` |
| `action_mask` | `[N, 4]` | `bool` | Legal mask for `{fold=0, check=1, call=2, raise=3}` |
| `min_raise`, `max_raise` | `[N]` | `int32` | Legal raise-delta chip bounds |
| `pot_total` | `[N]` | `int32` | Total chips committed this hand (`pot + bets.sum(-1)`) |
| `player_id` | `[N]` | `int32` | Acting seat (`0` or `1`) |
| `terminated`, `episode_id` | `[N]` | `bool`, `int32` | Hand completion flag and monotonic hand counter |

## 2. Policy Backbones

Both backbones output a masked 4-way action distribution, a 7-way discrete pot-fraction raise distribution (`min-raise, 0.5p, 0.75p, 1.0p, 1.5p, 2.5p, all-in`), and a scalar value estimate:

- **LSTM (`src/gpu_poker/policy.py`):** card embeddings + scalar MLP + pre-LSTM torso + per-seat LSTM `(h, c)` + post-LSTM MLP heads.
- **Transformer (`src/gpu_poker/policy_transformer.py`):** encoder-only Transformer over `[1 scalar query, 7 factored card tokens, 32 public-action tokens]` with branchless **48× suit-isomorphism + hole-order canonicalization** baked into the model (identical in PyTorch, ONNX, and browser), packed history state `[1, B, 129]`, and an optional training-only **CTDE omniscient critic** (`omniscient_value_head`) conditioned on `obs["opp_cards"]` to cut advantage variance without inference leakage.

## 3. Training Stack (`train/`)

- **R-NaD (`train/rnad_runner.py`, `conf/train_rnad.yaml`):** DeepNash-style Regularized Nash Dynamics with strictly zero-sum reward transform (`r - eta * log(pi / pi_reg)`), periodic anchor updates, CTDE omniscient critic, and duplicate-deck evaluation.
- **AlphaStar Triad (`train/triad_runner.py` + `train/triad.py`):** partitions `num_envs` (default `60/20/20`) across three independent networks and optimizers (Main, Main Exploiter, League Exploiter) with duplicate-deck eval, PFSP score updates, per-role FIFO pool pruning, and 3-file checkpoint save/resume.
- **Single-net (`train/single_runner.py`):** pure self-play PPO (GAE) or NFSP-lite league PPO (MC returns).
- **Exploitability (`train/exploitability.py`):** trains a fresh best-response Transformer against any frozen checkpoint and reports duplicate-deck `bb/100` / `mbb/g` exploitability.

## 4. ONNX Export & Web Parity

`train/export_onnx.py` exports deterministic graphs (`cards, scalars, action_mask, h, c -> action_logits, raise_logits, value, h_new, c_new`) for both LSTM and Transformer backbones and validates max abs diff `< 1e-3` against PyTorch. `web/src/inference.mjs` detects the backbone from `h`'s static width (`129` for Transformer) and runs sampling + token pushing in JS.

## 5. Profiling

```bash
# Environment-only throughput ceiling
uv run python bench/run.py

# Kernel micro-benchmarks (step vs obs vs copies)
uv run python bench/profile_kernels.py --num-envs 65536 --iterations 100

# Full training step profiler (emits Chrome trace)
uv run python bench/profile_training.py --num-envs 65536 --steps 50

# Triad update profiler
uv run python train/profile_triad.py --config-name=train_triad_v100 total_env_steps=10000000
```
