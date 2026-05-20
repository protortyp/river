# Training Plan (pokergpu)

Goal: train a strong, GPU-first HU NLHE agent that generalizes across stack depths and blind sizes, with correctness + performance guardrails.

## Principles

- **Keep the learning problem well-posed**: start simple (narrow regime), expand via curriculum.
- **Normalize to remove units**: represent chips in **big-blind (bb) units** wherever possible.
- **Evaluate continuously**: don’t trust training curves without winrate/bb metrics vs fixed opponents.
- **Stay GPU-first**: avoid CPU round-trips in the hot path; measure SPS for collection and update separately.

## Observation / Normalization (for stack & blinds)

We want the policy to generalize across different blind levels and stack depths.

### Recommended normalization

- Express chip quantities in **bb units**:
  - `stack_bb = stack_chips / big_blind`
  - `pot_bb = pot_chips / big_blind`
  - `to_call_bb = to_call_chips / big_blind`
  - `min_raise_bb = min_raise_chips / big_blind`
  - `max_raise_bb = max_raise_chips / big_blind`
- Additionally include a **stack-depth regime indicator**:
  - `effective_stack_bb = min(stack0_bb, stack1_bb)`
  - Use `log_effective_stack_bb` (or a clipped/normalized version) as a scalar feature.

### What to encode explicitly

Even with bb normalization, it helps to include:

- `effective_stack_bb` (or `log_effective_stack_bb`)
- optional: `big_blind` normalized (useful if we later mix bb ratios / rake / antes)

## Regime Randomization (curriculum)

Instead of fixed `starting_stack=1000`, randomize **effective stack depth in bb** per episode (or per env reset):

### Phase 0 (sanity)

- Fixed blinds (e.g. SB=1, BB=2)
- Fixed stack depth (e.g. `effective_stack_bb = 50`)
- Purpose: debug PPO loop + action semantics, confirm it beats trivial baselines.

### Phase 1 (small discrete set)

- Sample `effective_stack_bb ∈ {10, 20, 40, 80, 160}` with equal probability.
- Keep SB/BB ratio fixed (SB = BB/2).
- Purpose: teach short-stack and deep-stack regimes without continuous distribution noise.

### Phase 2 (continuous distribution)

- Sample `effective_stack_bb ~ log-uniform([8, 200])` (or a mixture with more mass at common depths).
- Optional: vary SB/BB ratio if desired later, but keep it fixed initially.

Implementation note: when sampling `effective_stack_bb`, set `starting_stack = effective_stack_bb * big_blind` (per env on reset). This keeps “units” consistent while still varying the depth.

Repository status: per-env reset parameters are implemented via `WarpPokerEnv.reset_with_config(starting_stack=..., small_blind=..., big_blind=...)` and are used for step-time auto-resets as well.

## Training Loop (v0 → v1)

### v0: PPO self-play baseline (no league)

- One policy controls both seats from the active-player perspective.
- LSTM from day one (configurable; default `lstm_hidden=256` in `conf/train.yaml`).
- Actions:
  - categorical over `{fold, check, call, raise}` with env `action_mask`
  - `raise_frac ∈ [0,1]` (Beta distribution), mapped into `[min_raise,max_raise]`
- Rollouts:
  - fixed horizon `T` (64/128), regardless of hand termination
  - `terminated` cuts GAE traces and resets RNN state
- Updates:
  - PPO with clipping, value loss, entropy bonus
  - minibatches + multiple epochs per rollout (realistic settings)

Repository status: a PPO trainer exists at `train/train.py` (GPU-first, uses
`WarpPokerEnv` directly). Hydra runs write `results.json` in the run directory and
also update repo-level `train/latest_train.json`.

Convenience: use `uv run python train/query_training.py` to print a copy/paste friendly summary of the latest run (TensorBoard + `results.json`).

Checkpointing: `train/train.py` supports periodic checkpoint saves to
`checkpoint_dir` and resuming via `resume_from=<path>`. Relative checkpoint
paths are interpreted from the repository root, even though Hydra changes into
the run directory.

### v1: NFSP-lite stabilization (snapshot pool)

Pure self-play is highly non-stationary. Add a small snapshot pool:

- Every `K` updates, save a checkpoint into a pool (reservoir or FIFO).
- During collection, for some fraction of envs (or per rollout), play vs a sampled snapshot instead of the latest policy.
  - Start with something like `eta=0.8` self-play, `0.2` vs snapshot.

Keep this minimal; defer “league archetypes” until we have a stable baseline.

Repository status:
- `train/train.py` supports an NFSP-lite "league" mode via `conf/train.yaml`:
  - `league_enabled`: enable opponent sampling during training
  - `league_eta_selfplay`: fraction of envs that remain pure self-play
  - unified opponent pool: when not self-play, sample bots and snapshots with PFSP weights
  - `league_snapshot_pool`: in-memory snapshot pool size (0 disables snapshots)
  - `league_snapshot_every_updates`: add a new snapshot every N PPO updates (disk-backed under `checkpoints/league_pool/`)
  - `league_eval_every_updates`: evaluate learner vs a subset of snapshots every N updates (for PFSP weights)
  - `league_eval_hands`: hands per snapshot eval
  - `league_eval_subset`: number of snapshots to eval per cycle (always includes the most recent few)
  - `league_eval_deterministic`: stochastic vs deterministic eval (stochastic is preferred for mixed strategies)
  - `league_eval_seat_swap`: evaluate both seats and average (reduces SB/BB bias)
  - `league_pfsp_temperature`, `league_pfsp_epsilon`: PFSP weighting parameters
  - `league_rollout_snapshot_k`: cap distinct snapshot opponents used in a resampling batch

Note: league mode currently uses Monte-Carlo terminal returns (no GAE) to avoid
information leakage from opponent-to-act observations (the env exposes the acting
player's hole cards in `obs["cards"]`).

PFSP-style sampling:
- The league can bias snapshot opponent sampling toward "near-50%" matchups using a PFSP-style weight
  (maps `bb/hand` -> win-prob via sigmoid, then weights by `p*(1-p)`).

Recommended league starting point:
- Start with `league_eta_selfplay=0.6`, bots present in the unified pool, and snapshots
  disabled or rare.
- Then introduce snapshots:
  - `league_snapshot_pool=8`, `league_snapshot_every_updates=20`

## Evaluation Protocol (required)

Evaluation should answer “does it beat anything real?” and “is it improving?”.

### Baselines

- Scripted bots (vectorized):
  - calling station (always check/call)
  - random aggressive (prefers raises)
  - nit (folds to aggression)
  - loose-passive (calls/checks a lot)
  - loose-aggressive (raises a lot)
  - optionally a tight-aggressive heuristic
- Frozen snapshots:
  - latest vs a set of historical checkpoints
  - periodic self-play eval vs the league snapshot pool (seat-swapped), to detect plateaus/cycling that bots may miss

### Metrics

- `bb/hand` and `bb/100` vs each baseline (with CI if possible)
- terminal-rate distribution and average hand length (to detect degenerate strategies)
- policy stats: entropy, KL (old/new), clip fraction, grad norm

## Reward design

- Primary reward should remain **true game reward** (net chips / bb) at terminal transitions.
- Variance is expected; at scale, PPO learns expected value via averaging.
- Avoid dense shaping based on “hand strength” as the main reward; it can teach the wrong incentives.

## Logging Roadmap

Near-term (already supported):

- JSON logs in Hydra run directories and repo-level `train/latest_train.json`.
- Smoke/benchmark scripts still write their own `train/runs/` JSON files.

Next:

- Hydra configs for reproducible runs and clean overrides.
- TensorBoard scalars:
  - throughput: collection/update SPS
  - PPO: losses, entropy, KL, clip frac, value loss
  - eval: bb/hand vs each baseline

Repository status:
- `train/train.py` supports TensorBoard via `POKERGPU_TRAIN_TB_DIR` (default: `train/tensorboard`).
- `train/train.py` supports periodic eval vs calling station via `POKERGPU_TRAIN_EVAL_EVERY` and `POKERGPU_TRAIN_EVAL_STEPS`.
- `train/train.py` is Hydra-driven by default; override via CLI, e.g. `uv run python train/train.py num_envs=8192 eval_every=5`.
- `train/train.py` shows a `tqdm` progress bar by default; set `POKERGPU_TRAIN_TQDM=0` to disable.

Self-play snapshot eval (added):
- `train/train.py` can periodically evaluate the current policy vs a subset of the snapshot pool and log:
  - `selfplay/bb_per_hand_{mean,min,max}`
  - `selfplay/vs_latest_bb_per_hand`
- Config knobs live in `conf/train.yaml` (`selfplay_eval_*`). This requires `league_snapshot_pool > 0` and snapshots being created.

## “Plays vs humans” considerations (later)

- The training objective should target general exploitability (approximate Nash) rather than overfitting to a single opponent.
- For human play, consider:
  - evaluation across diverse opponents
  - optional opponent-conditioning / session-level memory only in evaluation or with explicit opponent_id/session structure
