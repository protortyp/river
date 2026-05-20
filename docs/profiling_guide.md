# Profiling Guide: Identifying Performance Bottlenecks

This guide explains how to profile the pokergpu training pipeline to find optimization opportunities beyond the current ~300k env steps/second.

## Quick Start

```bash
# Install profiling tools (already in pyproject.toml dev dependencies)
uv sync --group dev

# Run micro-benchmarks for kernel-level analysis
uv run python bench/profile_kernels.py --num-envs 65536

# Run full training profiling with torch.profiler
uv run python bench/profile_training.py --num-envs 65536 --steps 50

# For detailed GPU kernel analysis (requires NVIDIA Nsight Systems)
nsys profile -t cuda,nvtx -o profile_report uv run python bench/profile_training.py
# Then open profile_report.nsys-rep in Nsight Systems
```

## Profiling Tools Available

### 1. `bench/profile_kernels.py` - Micro-benchmarks
Measures individual operations to identify specific bottlenecks:
- Kernel launch times (step_kernel, obs_kernel)
- Memory operations (torch↔warp conversions)
- Python control flow overhead (league opponent loops)
- GPU sync points

### 2. `bench/profile_training.py` - Full Pipeline Profiling
Profiles the complete training loop:
- Environment stepping
- Network inference
- PPO update phase
- Outputs Chrome trace files for visualization

### 3. NVIDIA Nsight Systems
For detailed GPU kernel analysis:
```bash
nsys profile -t cuda,nvtx -o report uv run python bench/profile_training.py
```
Then visualize in Nsight Systems GUI.

### 4. torch.profiler (built-in)
Generates Chrome trace files showing:
- CPU vs GPU time breakdown
- Kernel launch patterns
- Memory allocation

Open trace in: `chrome://tracing` or `https://ui.perfetto.dev/`

## Known Bottlenecks (from code analysis)

### 1. Two Kernel Launches Per Step (~15-20% overhead)

**Location:** `src/gpu_poker/env.py:510-546`

```python
# Current: 2 separate kernel launches
wp.launch(step_kernel_per_env, ...)   # Game state transition
wp.launch(get_obs_legal_kernel, ...)  # Observation extraction
```

**Impact:** Each kernel launch has overhead (~5-20µs). At 65K envs, this is non-trivial.

**Fix:** Fuse into single kernel:
```python
# Proposed: 1 kernel that does step + obs in one pass
wp.launch(step_and_get_obs_kernel, ...)
```

### 2. League Opponent Python Loop (~30-50% slowdown when league_enabled)

**Location:** `train/train.py:883-925`

```python
# Problem: Python loop over unique opponents causes:
# 1. GPU->CPU sync (torch.unique().tolist())
# 2. Multiple small forward passes
# 3. Poor GPU utilization

snap_mask = is_opponent_actor & opp_kind.eq(2)
if snap_mask.any():
    for pool_idx in torch.unique(opp_id[snap_mask]).tolist():  # <-- SYNC!
        # ... separate forward pass for each opponent
```

**Impact:** With 32 snapshots, can cause 32 separate network forward passes per step.

**Current Mitigation:** `league_rollout_snapshot_k` limits active snapshots per hand.

**Better Fix:** Batch all opponents together or use GPU-side control flow.

### 3. GPU-CPU Sync Points

**Locations to audit:**
- `torch.unique(...).tolist()` - train.py:861, 885
- `m.any()` checks - multiple locations (usually OK if not used for control flow)
- `.item()` calls - avoided in hot loop, but check diagnostics code

**Detection:** Run with `CUDA_LAUNCH_BLOCKING=1` to identify hidden syncs.

### 4. torch→warp Copy Overhead

**Location:** `src/gpu_poker/env.py:506-507`

```python
wp.copy(self.actions, wp.from_torch(actions_torch, dtype=wp.int32))
wp.copy(self.amounts, wp.from_torch(amounts_torch, dtype=wp.int32))
```

**Impact:** Small but measurable (~10-50µs per step).

**Fix:** Use `step_from_buffers()` when actions are already GPU-resident:
```python
# Already implemented in env.py:567
env.step_from_buffers()  # Skips torch->warp copy
```

### 5. LSTM Sequential Dependency

**Location:** `src/gpu_poker/policy.py:145`

```python
y, (h2, c2) = self.lstm(x, (h_state, c_state))  # Sequential operation
```

**Impact:** LSTM can't fully parallelize across sequence dimension.

**Long-term Fix:** Consider Transformer architecture or stateless policy.

## Profiling Workflow

### Step 1: Establish Baseline
```bash
# Environment-only ceiling
uv run python bench/bench_sps.py --num-envs 131072

# Full training throughput
uv run python bench/bench_league_rollout.py --snapshot-filled 0  # No opponents
uv run python bench/bench_league_rollout.py --snapshot-filled 8 --rollout-snapshot-k 4  # With league
```

### Step 2: Run Micro-benchmarks
```bash
uv run python bench/profile_kernels.py --num-envs 65536 --iterations 100
```

Look for:
- Kernel launch times vs full step time → kernel overhead
- `torch.unique().tolist()` timing → sync overhead
- Index selection/scatter timing → league fragmentation cost

### Step 3: Profile Full Training
```bash
uv run python bench/profile_training.py --num-envs 65536
```

Look at the output breakdown:
- `env.step` vs `network.forward_step` → where is time spent?
- `PPO minibatch` breakdown → forward vs backward vs optimizer

### Step 4: Deep GPU Analysis
```bash
nsys profile -t cuda,nvtx,osrt -o detailed_profile \
  uv run python bench/profile_training.py --num-envs 32768 --steps 20
```

In Nsight Systems, look for:
- Gaps between kernel launches (kernel overhead)
- Memory transfer events (data movement)
- CPU activity during GPU work (sync stalls)

## Optimization Priority Order

Based on typical impact:

1. **Kernel fusion** (step + obs) - Medium effort, ~20% gain
2. **Eliminate Python loops** in league code - Medium effort, ~30-50% gain when league enabled
3. **Use step_from_buffers** - Low effort, ~5% gain
4. **Batch opponent forward passes** - High effort, major gain for many opponents
5. **Reduce observation size** - Low effort, minor gain
6. **LSTM → Transformer** - High effort, enables better parallelism

## Expected Results

| Configuration | Expected SPS | Notes |
|--------------|--------------|-------|
| Env only (bench_sps.py) | 2-5M | Kernel ceiling |
| Self-play (no league) | 400-600k | Network overhead |
| League (k=4) | 200-350k | Opponent fragmentation |
| League (k=0, 32 snapshots) | 50-100k | Severe fragmentation |

## Common Issues

### "CUDA out of memory"
- Reduce `num_envs`
- Reduce `snapshot_pool` size
- Check for memory leaks (TensorDict accumulation)

### Low GPU utilization
- Check for CPU-GPU sync points
- Increase `num_envs` or `minibatch_size`
- Profile to find blocking operations

### SPS varies wildly
- Ensure warmup steps (JIT compilation)
- Use `timing_sync=True` for accurate measurements
- Check for thermal throttling

## Files Reference

| File | Purpose |
|------|---------|
| `bench/profile_training.py` | Full training profiler with torch.profiler |
| `bench/profile_kernels.py` | Micro-benchmarks for individual operations |
| `bench/bench_sps.py` | Environment-only throughput |
| `bench/bench_league_rollout.py` | League rollout with opponent fragmentation |
| `bench/bench_ppo_update.py` | PPO update phase |
| `src/gpu_poker/env.py` | Environment stepping (hot path) |
| `train/train.py` | Training loop (lines 725-936 for collection) |
