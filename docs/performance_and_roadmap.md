# Performance & Roadmap

This repo’s goal is a **gym-like poker environment that runs almost entirely on GPU** (Warp + CUDA), enabling extremely high simulation throughput for RL.

## Where We Are

Core environment:
- `src/gpu_poker/env.py`: `WarpPokerEnv` (GPU state, `reset`, `step`, `step_from_buffers`)
- `src/gpu_poker/kernels/state.py`: game rules + auto-reset
- `src/gpu_poker/kernels/observations.py`: observations
- `src/gpu_poker/kernels/legal_actions.py`: action mask + raise bounds
- `src/gpu_poker/kernels/actions.py`: betting math + illegal-action policy (`force_legal_action`)

Training plumbing:
- `src/gpu_poker/torchrl_env.py`: TorchRL `EnvBase` wrapper (`WarpPokerTorchRLEnv`)
- `train/torchrl_smoke.py`: “collector-like” throughput script (writes JSON)
- `train/policy_train_smoke.py`: PPO-style throughput script (collection vs update SPS, writes JSON)
- `train/train.py`: PPO trainer v0 (per-player LSTM state, JSON logging)

Benchmarks:
- `bench/run.py`: benchmark runner (writes JSON, uses CUDA event timing on CUDA
  and wall-clock timing on CPU-only Warp builds)
- `bench/latest.json`: latest benchmark snapshot (gitignored)

## Benchmarks: What To Run

Environment throughput:
- `uv run python bench/run.py`

On CPU-only machines, `bench/run.py` still runs as a functional smoke benchmark,
but the numbers are not comparable to CUDA throughput baselines.

Training/collector-style throughput:
- `uv run python train/torchrl_smoke.py`

PPO baseline training (with optional TensorBoard + eval):
- `uv run python train/train.py`
- League training (opponent sampling): `uv run python train/train.py league_enabled=true`

Training plan:
- `docs/training_plan.md`

Policy scaling (Hydra config):
- `conf/train.yaml` exposes `card_embed_dim`, `mlp_dim`, `torso_layers`, `lstm_hidden`, `head_layers`, `head_dim` so you can increase model capacity gradually while watching SPS/KL/entropy/EV.

League/PFSP diagnostics (TensorBoard):
- `league/snapshots_filled`, `league/snapshot_bb_per_hand_{mean,min,max}`
- `league/pfsp_weight_{min,max,entropy,ess}` (monitor PFSP collapse)

Both scripts write timestamped JSON plus a `latest.json` file under a gitignored path:
- `bench/results/*.json`, `bench/latest.json`
- smoke scripts: `train/runs/*.json`, `train/latest.json`
- trainer: Hydra run-dir `results.json`, plus repo-level `train/latest_train.json`

## Interpreting Results

You’ll typically see:
- **Env-only ceiling** (pure kernels, minimal host work): extremely high SPS.
- **Training loop throughput**: lower due to action selection, TensorDict overhead, and per-step control flow.

The smoke script prints:
- `SPS`: env-steps per second (`num_envs * steps / elapsed`)
- `invalid_rate`: fraction of steps where sanitization was needed (should be ~0 if sampling respects mask/bounds)
- `terminated_rate`: fraction of envs ending per step (expected to be high under random play)
- `elapsed_gpu_s`: CUDA-event measured loop time (should match wall time closely)
- `breakdown_gpu_s`: microbench GPU timings for sampling/step/post kernels

## Current Bottleneck (Typical)

After moving action sampling on-GPU and adding `step_from_buffers`, remaining slowdown vs the kernel ceiling is usually:
- per-step Python/TorchRL overhead
- kernel launch overhead (multiple kernels per step)
- memory passes across multiple kernels (`sampling`, `step`, `post`)

## Next Performance Improvements (Options)

### 1) Fuse sampling into `step_kernel` (recommended next)
**What:** sample `action_type`/`amount` inside `step_kernel` (or a new kernel variant).

**Why:** removes an entire kernel launch and intermediate writes/reads of actions.

**Impact:** meaningful (launch overhead is non-trivial at >100k envs).

**Risk:** low–medium (keep old path for comparison and correctness).

### 2) Fuse post-processing (obs + legal) into the step kernel
**What:** combine `state.step` + `write_observation` + `write_legal_actions` into one kernel.

**Why:** single launch per global step, fewer global memory passes.

**Impact:** large.

**Risk:** medium (more code in one kernel; careful about registers/divergence).

### 3) Reduce TorchRL/TensorDict churn
**What:** collect multi-step rollouts with fewer Python round-trips, reuse TensorDict structures where possible.

**Impact:** large for “training SPS”.

**Risk:** low (mostly tooling/loop structure).

### 4) Add CUDA profiling hooks
**What:** optional NVTX ranges or Warp timers around kernels in benchmark scripts.

**Impact:** improves diagnosis, not raw speed.

**Risk:** low.

## Correctness Guardrails (Must Keep)

- Chip conservation invariants (see tests like `tests/test_invariants.py`)
- Legal-action generation + bounds must remain consistent with action application
- `episode_id` + `terminated` must reliably signal episode boundaries even with auto-reset
- No unintended GPU→CPU syncs in hot paths (avoid `.cpu()`, `.numpy()`, excessive `synchronize`)

## Hydra / TensorBoard (Future)

When introducing Hydra/TensorBoard:
- Keep benchmark scripts runnable without Hydra (so quick perf checks stay frictionless).
- Add Hydra configs for training runs, and write logs/checkpoints under gitignored directories.
- Use TensorBoard for:
  - SPS (env + training loop)
  - invalid/terminated rates
  - reward statistics
  - GPU memory/utilization snapshots (optional)
