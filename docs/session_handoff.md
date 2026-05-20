# Session Handoff (pokergpu)

This is a “memory dump” so a new session can restart quickly without losing context.

## Project Goal

Build a **gym-like, massively-parallel HU NLHE poker environment that runs GPU-first**
(Warp kernels + Torch integration) so RL training can scale to very high SPS and avoid CPU round-trips.

Primary goals:
- Correctness first (chip conservation, legal actions, pot settlement, showdown, auto-reset boundaries).
- Performance guardrails (benchmarks + smoke scripts).
- Training stack that stays on GPU where possible (Torch policy + Warp env) with strong diagnostics.

## Environment Overview (GPU poker env)

- Core env: `src/gpu_poker/env.py` (`WarpPokerEnv`)
- GPU state: Warp `GameState` (SoA arrays).
- Zero-copy outputs: torch tensors via `wp.to_torch(...)` (important for training buffer snapshots).

### Episode boundaries

- Env auto-resets on terminal transitions inside kernels.
- Reliable hand boundary signal is `terminated` (computed as `episode_id` changed during `step`).
- Training uses `terminated` rather than “dones”.

### Per-env configuration (stacks/blinds)

Supported:
- `WarpPokerEnv.reset_with_config(starting_stack=..., small_blind=..., big_blind=...)`
- Per-env cfg stored in state fields and used for auto-reset and raise floors.

### Observations

- `obs["cards"]`: `[N, 7]` active player hole cards + visible board cards; hidden encoded as `-1`.
- `obs["scalars"]`: `(N, 17)` including public-history scalars and bb-normalization aids.
- `obs["action_mask"]`: `(N, 4)` legal action mask for `{fold, check, call, raise}`.
- `obs["min_raise"]`, `obs["max_raise"]`: integer raise bounds.
- `obs["player_id"]`: acting seat id in `{0,1}`.
- `obs["big_blind"]`: per-env configured big blind.

## Performance & Benchmarks

Benchmarks:
- `bench/run.py` writes gitignored JSON results:
  - `bench/latest.json`
  - `bench/results/*.json`
- It uses CUDA event timing on CUDA devices and wall-clock timing when Warp is
  running CPU-only.

Smoke scripts:
- `train/torchrl_smoke.py`: collector-style throughput (writes JSON)
- `train/policy_train_smoke.py`: PPO-style throughput breakdown (writes JSON)

## Training (PPO + League/PFSP)

### Trainer entrypoint

- Main trainer: `train/train.py` (Hydra config `conf/train.yaml`)
- Copy/paste summaries: `train/query_training.py` (reads TensorBoard and/or `results.json`)

### Policy net

- `src/gpu_poker/policy.py`: `PokerPolicyNet` actor-critic with LSTM and Beta raise head.
- Model scaling knobs are wired through Hydra: `card_embed_dim`, `mlp_dim`, `torso_layers`,
  `lstm_hidden`, `head_layers`, `head_dim`.

### Critical correctness fix (must keep)

Env outputs are zero-copy views into buffers overwritten each `env.step()`.
Training must snapshot the observation **before** stepping or PPO will train on mismatched `(s_t, a_t)`.

This is handled in `train/train.py` by copying `obs` into the rollout buffer before `env.step()`.

### Training modes

#### 1) Pure self-play (GAE)

- Both seats are controlled by the same learner policy (active-player perspective).
- Uses GAE + bootstrap value at the end of rollout.

#### 2) League mode (NFSP-lite opponent sampling)

Enabled by `league_enabled=true` in `conf/train.yaml`.

Opponents are sampled per-env at hand boundaries:
- self-play vs latest (prob `league_eta_selfplay`)
- otherwise vs the unified opponent pool of bots + snapshots (PFSP-weighted)

Important: **league mode uses Monte-Carlo terminal returns (no GAE)** to avoid
requiring value targets on opponent-to-act states (env observation includes acting player hole cards).

### Snapshot pool (disk-backed + in-memory ring)

- Pool implementation: `train/league.py::UnifiedOpponentPool`
- Stored under `checkpoint_dir/league_pool/`:
  - `unified_index.json` (metadata)
  - `snap_env*_upd*.pt` (model state_dict)
- In-memory snapshot ring for fast inference; on startup it loads up to `league_snapshot_pool` most recent snapshots.

Rollout performance:
- `league_rollout_snapshot_k` limits the number of distinct snapshot opponents sampled
  during a role-resampling batch. Bots remain available.

### Snapshot eval + seat bias fix

Every `league_eval_every_updates`, the trainer evaluates learner vs a subset of snapshots:
- always includes the most recent few snapshots
- plus a random fill up to `league_eval_subset`

Seat bias is mitigated by `league_eval_seat_swap=true`:
- eval learner as P0 vs snapshot as P1
- eval snapshot as P0 vs learner as P1
- stored snapshot score is the average from learner perspective

Metadata stored per snapshot (in `index.json`):
- `bb_per_hand`: learner score used for PFSP weighting
- `hands`: total hands across both seat assignments (when seat swap enabled)
- `bb_per_hand_p0`: learner-as-P0 bb/hand (learner view)
- `bb_per_hand_p1`: learner-as-P1 bb/hand (learner view)

### PFSP weighting

PFSP weights are derived from snapshot `bb/hand` estimates:
- map to win-prob with `sigmoid(bb/hand / league_pfsp_temperature)`
- weight by `p*(1-p)` (+ `league_pfsp_epsilon`)

Diagnostics logged to TensorBoard:
- `league/pfsp_weight_{min,max,entropy,ess}` (monitor weight collapse)

### Bots + evaluation

Bots:
- `train/bots.py`: `calling_station`, `random_aggressive`
  - plus: `nit`, `loose_passive`, `loose_aggressive`

Eval harness:
- `train/eval.py`: `eval_vs_bot`, `eval_vs_policy`

Trainer logs eval vs *all* scripted bots to TensorBoard:
- `eval/<bot>_bb_per_hand`, `eval/<bot>_bb_per_100`, `eval/<bot>_hands`

### Self-play snapshot eval (stronger than bots)

To detect plateaus and cycling that bots may miss, the trainer can periodically evaluate the current
policy vs a subset of the league snapshot pool (seat-swapped, stochastic by default) and logs:
- `selfplay/bb_per_hand_{mean,min,max}`
- `selfplay/vs_latest_bb_per_hand`

Config lives in `conf/train.yaml` under `selfplay_eval_*` and requires `league_snapshot_pool > 0`
and snapshots being created (`league_snapshot_every_updates > 0`).

## Checkpointing

- `train/checkpointing.py`: save/load model+optimizer+RNG+counters
- `train/train.py` supports:
  - `checkpoint_dir`, `save_every_env_steps`, `resume_from`
- Relative `checkpoint_dir` and `resume_from` paths are resolved from the repo
  root before Hydra changes into the run directory.

## Commands

- Train: `uv run python train/train.py`
- Query: `uv run python train/query_training.py`
- Resume: `uv run python train/train.py resume_from=checkpoints/<ckpt>.pt`
- Bench: `uv run python bench/run.py`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`

## Current High-ROI Next Steps

1) Expand bot set (tight/passive, tight/aggressive, etc.) and track eval vs each.
2) Consider adding a league pool “recent always + reservoir diversity” bucket if cycling appears.
3) Improve PFSP calibration (mapping bb/hand -> win-prob) and log more league stats in `query_training.py`.
4) Regime randomization for stacks/blinds during training (curriculum).
