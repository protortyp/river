#!/usr/bin/env python3
"""
Micro-benchmarks for individual kernel operations and potential bottlenecks.

This script measures:
1. Individual kernel launch times
2. Memory copy overhead (torch <-> warp)
3. Kernel fusion potential (measuring 2 kernels vs hypothetical fused)
4. Tensor operations that happen in training loop
5. Python control flow overhead (the league loops)

Usage:
  uv run python bench/profile_kernels.py --num-envs 65536
"""

import argparse
import time
from dataclasses import dataclass

import torch
import warp as wp

from gpu_poker.env import (
    WarpPokerEnv,
    get_obs_legal_kernel,
    step_kernel_per_env,
)


def sync_all(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wp.synchronize()


@dataclass
class BenchResult:
    name: str
    total_ms: float
    iterations: int
    ops_per_iter: int = 1

    @property
    def avg_us(self) -> float:
        """Average time per iteration in microseconds."""
        return (self.total_ms / self.iterations) * 1000

    @property
    def throughput(self) -> float:
        """Operations per second."""
        return (
            (self.iterations * self.ops_per_iter) / (self.total_ms / 1000)
            if self.total_ms > 0
            else 0
        )


def measure(
    name: str, fn, iterations: int, warmup: int, device: str, ops_per_iter: int = 1
) -> BenchResult:
    """Run a benchmark with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    sync_all(device)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    sync_all(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return BenchResult(
        name=name, total_ms=elapsed_ms, iterations=iterations, ops_per_iter=ops_per_iter
    )


def bench_kernel_launches(num_envs: int, device: str, iterations: int = 100) -> list[BenchResult]:
    """Benchmark individual kernel launch overhead."""
    results = []

    wp.init()
    env = WarpPokerEnv(num_envs=num_envs, device=device)

    # Pre-populate action buffers
    action_types = wp.zeros(num_envs, dtype=wp.int32, device=device)
    amounts = wp.zeros(num_envs, dtype=wp.int32, device=device)
    wp.copy(env.actions, action_types)
    wp.copy(env.amounts, amounts)
    sync_all(device)

    # 1. Step kernel only
    def run_step_kernel():
        wp.launch(
            kernel=step_kernel_per_env,
            dim=num_envs,
            inputs=[
                env.state,
                env.actions,
                env.amounts,
                env.rewards,
                env.invalid_action,
                env.terminated,
                env.primes,
                env.unsuited_lut,
                env.flush_lut,
                env.starting_stack_cfg,
                env.small_blind_cfg,
                env.big_blind_cfg,
            ],
            device=device,
        )

    results.append(
        measure("step_kernel_per_env", run_step_kernel, iterations, 20, device, num_envs)
    )

    # 2. Observation kernel only
    def run_obs_kernel():
        wp.launch(
            kernel=get_obs_legal_kernel,
            dim=num_envs,
            inputs=[
                env.state,
                env.obs_cards,
                env.obs_scalars,
                env.primes,
                env.unsuited_lut,
                env.flush_lut,
                env.action_mask,
                env.min_raise,
                env.max_raise,
            ],
            device=device,
        )

    results.append(
        measure("get_obs_legal_kernel", run_obs_kernel, iterations, 20, device, num_envs)
    )

    # 3. Both kernels (current implementation)
    def run_both_kernels():
        run_step_kernel()
        run_obs_kernel()

    results.append(
        measure("both_kernels_separate", run_both_kernels, iterations, 20, device, num_envs)
    )

    # 4. Full env.step (includes torch->warp copy)
    action_torch = torch.zeros(num_envs, dtype=torch.int32, device=device)
    amounts_torch = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def run_env_step():
        env.step(action_torch, amounts_torch)

    results.append(measure("env.step (full)", run_env_step, iterations, 20, device, num_envs))

    # 5. step_from_buffers (skips torch->warp copy)
    def run_step_from_buffers():
        env.step_from_buffers()

    results.append(
        measure("env.step_from_buffers", run_step_from_buffers, iterations, 20, device, num_envs)
    )

    return results


def bench_memory_operations(num_envs: int, device: str, iterations: int = 100) -> list[BenchResult]:
    """Benchmark memory copy and conversion overhead."""
    results = []

    wp.init()

    # Create arrays
    wp_arr = wp.zeros(num_envs, dtype=wp.int32, device=device)
    torch_tensor = torch.zeros(num_envs, dtype=torch.int32, device=device)

    # 2D arrays (like observations)
    wp_arr_2d = wp.zeros((num_envs, 17), dtype=wp.float32, device=device)

    sync_all(device)

    # 1. wp.copy (warp to warp)
    wp_arr2 = wp.zeros(num_envs, dtype=wp.int32, device=device)

    def wp_copy():
        wp.copy(wp_arr2, wp_arr)

    results.append(measure("wp.copy (1D, int32)", wp_copy, iterations, 20, device))

    # 2. wp.from_torch (creates view)
    def from_torch():
        _ = wp.from_torch(torch_tensor, dtype=wp.int32)

    results.append(measure("wp.from_torch (view)", from_torch, iterations, 20, device))

    # 3. wp.to_torch (creates view)
    def to_torch():
        _ = wp.to_torch(wp_arr)

    results.append(measure("wp.to_torch (view)", to_torch, iterations, 20, device))

    # 4. Full copy: torch -> warp array via wp.copy
    def full_torch_to_warp():
        wp.copy(wp_arr, wp.from_torch(torch_tensor, dtype=wp.int32))

    results.append(measure("torch->warp (copy)", full_torch_to_warp, iterations, 20, device))

    # 5. torch.clone (for comparison)
    def torch_clone():
        _ = torch_tensor.clone()

    results.append(measure("torch.clone (1D)", torch_clone, iterations, 20, device))

    # 6. Large 2D observation tensor
    torch.zeros(num_envs, 17, dtype=torch.float32, device=device)

    def to_torch_2d():
        _ = wp.to_torch(wp_arr_2d)

    results.append(measure("wp.to_torch (2D float32)", to_torch_2d, iterations, 20, device))

    return results


def bench_tensor_operations(num_envs: int, device: str, iterations: int = 100) -> list[BenchResult]:
    """Benchmark PyTorch operations that happen in the training loop."""
    results = []

    # Typical training loop tensors
    torch.randint(0, 4, (num_envs,), dtype=torch.int32, device=device)
    raise_frac = torch.rand(num_envs, 1, dtype=torch.float32, device=device)
    min_raise = torch.randint(10, 100, (num_envs,), dtype=torch.int32, device=device)
    max_raise = torch.randint(100, 1000, (num_envs,), dtype=torch.int32, device=device)
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    terminated[::100] = True  # ~1% terminated

    # LSTM hidden states
    h = torch.randn(1, num_envs, 256, dtype=torch.float32, device=device)
    c_state = torch.randn(1, num_envs, 256, dtype=torch.float32, device=device)
    z = torch.zeros_like(h)

    sync_all(device)

    # 1. Reset check (.any())
    def reset_any():
        return terminated.any()

    results.append(measure("terminated.any()", reset_any, iterations, 20, device))

    # 2. torch.where for hidden state reset
    reset = terminated.view(1, -1, 1)

    def hidden_reset():
        _ = torch.where(reset, z, h)
        _ = torch.where(reset, z, c_state)

    results.append(measure("torch.where (hidden reset)", hidden_reset, iterations, 20, device))

    # 3. raise_frac_to_amount conversion
    def raise_conversion():
        min_i = min_raise.to(dtype=torch.int32)
        max_i = max_raise.to(dtype=torch.int32)
        amount_f = min_i.float() + raise_frac.squeeze(-1) * (max_i - min_i).float()
        return amount_f.round().to(dtype=torch.int32)

    results.append(measure("raise_frac_to_amount", raise_conversion, iterations, 20, device))

    # 4. torch.unique (problematic - causes GPU->CPU sync)
    opp_id = torch.randint(0, 8, (num_envs,), dtype=torch.int32, device=device)

    def unique_tolist():
        return torch.unique(opp_id).tolist()

    results.append(measure("torch.unique().tolist() [SYNC]", unique_tolist, iterations, 20, device))

    # 5. torch.nonzero (also causes sync if result is used)
    mask = terminated.clone()
    mask[::10] = True

    def nonzero_op():
        return torch.nonzero(mask, as_tuple=False)

    results.append(measure("torch.nonzero", nonzero_op, iterations, 20, device))

    # 6. Index selection (subsetting for league)
    idx = torch.arange(0, num_envs, 10, device=device)  # ~10% subset
    cards = torch.randint(0, 52, (num_envs, 7), dtype=torch.int32, device=device)
    scalars = torch.randn(num_envs, 17, dtype=torch.float32, device=device)

    def index_select():
        _ = cards[idx]
        _ = scalars[idx]
        _ = h[:, idx, :]

    results.append(measure("index_select (10% subset)", index_select, iterations, 20, device))

    # 7. Index scatter (updating subset of hidden states)
    h_new = torch.randn(1, len(idx), 256, dtype=torch.float32, device=device)

    def index_scatter():
        h[:, idx, :] = h_new

    results.append(measure("index_scatter (10% subset)", index_scatter, iterations, 20, device))

    # 8. Buffer copy (rollout storage)
    buf_cards = torch.zeros(64, num_envs, 7, dtype=torch.int32, device=device)

    def buffer_copy():
        buf_cards[0].copy_(cards)

    results.append(measure("buffer.copy_ (cards)", buffer_copy, iterations, 20, device))

    return results


def bench_python_overhead(num_envs: int, device: str, iterations: int = 50) -> list[BenchResult]:
    """Benchmark Python control flow overhead in the training loop."""
    results = []

    # Simulate league opponent logic
    opp_kind = torch.randint(0, 3, (num_envs,), dtype=torch.int32, device=device)
    opp_id = torch.randint(0, 8, (num_envs,), dtype=torch.int32, device=device)
    pid = torch.randint(0, 2, (num_envs,), dtype=torch.int64, device=device)
    learner_seat = 0

    sync_all(device)

    # 1. League mask computation
    def league_masks():
        is_selfplay = opp_kind.eq(0)
        is_learner_actor = is_selfplay | pid.eq(learner_seat)
        is_opponent_actor = (~is_learner_actor) & (~is_selfplay)
        bot_mask = is_opponent_actor & opp_kind.eq(1)
        snap_mask = is_opponent_actor & opp_kind.eq(2)
        return is_learner_actor, bot_mask, snap_mask

    results.append(measure("league_mask_computation", league_masks, iterations, 10, device))

    # 2. Python loop overhead (simulating unique opponent iteration)
    snap_mask = (opp_kind.eq(2)) & pid.ne(learner_seat)

    def python_loop_overhead():
        unique_ids = torch.unique(opp_id[snap_mask]).tolist()
        for pool_idx in unique_ids:
            m = snap_mask & opp_id.eq(pool_idx)
            if m.any():
                _ = torch.nonzero(m, as_tuple=False).squeeze(-1)

    results.append(
        measure("python_loop (unique opps) [SYNC]", python_loop_overhead, iterations, 10, device)
    )

    # 3. Alternative: all-GPU opponent handling (no Python loop)
    def gpu_opponent_handling():
        # Count how many unique opponents (without going to CPU)
        unique_ids = torch.unique(opp_id[snap_mask])
        # Create masks for all opponents at once
        masks = opp_id.unsqueeze(0) == unique_ids.unsqueeze(1)  # [num_unique, num_envs]
        masks = masks & snap_mask.unsqueeze(0)
        return masks

    results.append(
        measure("gpu_opponent_masks (no sync)", gpu_opponent_handling, iterations, 10, device)
    )

    return results


def print_results(title: str, results: list[BenchResult], num_envs: int):
    """Print benchmark results."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")
    print(f"{'Operation':<40} {'Avg (µs)':>10} {'Total (ms)':>12} {'Throughput':>15}")
    print(f"{'-' * 70}")

    for r in results:
        if r.ops_per_iter > 1:
            tp_str = f"{r.throughput / 1e6:.2f}M envs/s"
        else:
            tp_str = f"{r.iterations / (r.total_ms / 1000):.0f} calls/s"
        print(f"{r.name:<40} {r.avg_us:>10.1f} {r.total_ms:>12.2f} {tp_str:>15}")


def main():
    parser = argparse.ArgumentParser(description="Kernel-level micro-benchmarks")
    parser.add_argument("--num-envs", type=int, default=65536)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print(f"Running micro-benchmarks with {args.num_envs:,} environments on {args.device}")
    print("=" * 70)

    # 1. Kernel launches
    kernel_results = bench_kernel_launches(args.num_envs, args.device, args.iterations)
    print_results("Kernel Launch Benchmarks", kernel_results, args.num_envs)

    # Calculate kernel fusion potential
    step_time = next(r.avg_us for r in kernel_results if r.name == "step_kernel_per_env")
    obs_time = next(r.avg_us for r in kernel_results if r.name == "get_obs_legal_kernel")
    both_time = next(r.avg_us for r in kernel_results if r.name == "both_kernels_separate")
    full_step_time = next(r.avg_us for r in kernel_results if r.name == "env.step (full)")
    fast_step_time = next(r.avg_us for r in kernel_results if r.name == "env.step_from_buffers")

    print("\n  Analysis:")
    print(f"    Kernel overhead (2 launches vs sum): {both_time - (step_time + obs_time):.1f}µs")
    print(f"    torch->warp copy overhead: {full_step_time - fast_step_time:.1f}µs")
    kernel_overhead_pct = (both_time / full_step_time) * 100
    print(f"    Kernels are {kernel_overhead_pct:.1f}% of full step")

    # 2. Memory operations
    mem_results = bench_memory_operations(args.num_envs, args.device, args.iterations)
    print_results("Memory Operation Benchmarks", mem_results, args.num_envs)

    # 3. Tensor operations
    tensor_results = bench_tensor_operations(args.num_envs, args.device, args.iterations)
    print_results("PyTorch Tensor Operation Benchmarks", tensor_results, args.num_envs)

    # 4. Python overhead
    python_results = bench_python_overhead(args.num_envs, args.device, 50)
    print_results("Python Control Flow Benchmarks", python_results, args.num_envs)

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY: Key Bottlenecks Identified")
    print("=" * 70)

    # Find operations with [SYNC] marker
    sync_ops = [r for r in tensor_results + python_results if "[SYNC]" in r.name]
    if sync_ops:
        print("\n  GPU->CPU Sync Points (cause pipeline stalls):")
        for r in sync_ops:
            print(f"    - {r.name}: {r.avg_us:.1f}µs per call")

    step_sps = args.num_envs / (step_time / 1e6) / 1e6
    obs_sps = args.num_envs / (obs_time / 1e6) / 1e6
    full_sps = args.num_envs / (full_step_time / 1e6) / 1e6
    fused_save_us = both_time - max(step_time, obs_time)
    fast_step_save_us = full_step_time - fast_step_time
    unique_sync_us = next((r.avg_us for r in tensor_results if "unique" in r.name), 0)

    print("\n  Kernel Performance:")
    print(f"    - step_kernel: {step_time:.1f}µs ({step_sps:.2f}M envs/s theoretical)")
    print(f"    - obs_kernel: {obs_time:.1f}µs ({obs_sps:.2f}M envs/s theoretical)")
    print(f"    - Full env.step: {full_step_time:.1f}µs ({full_sps:.2f}M envs/s)")
    print("\n  Optimization Opportunities:")
    print(f"    1. Fuse step + obs kernels → save ~{fused_save_us:.0f}µs/step")
    print(f"    2. Use step_from_buffers → save ~{fast_step_save_us:.0f}µs/step")
    print(f"    3. Eliminate torch.unique().tolist() → avoid {unique_sync_us:.0f}µs sync")
    print("    4. Batch opponent forward passes → reduce Python loop overhead")


if __name__ == "__main__":
    main()
