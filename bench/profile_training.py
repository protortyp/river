#!/usr/bin/env python3
"""
Profiling script to identify performance bottlenecks in the training loop.

Outputs:
  - Chrome trace file for torch.profiler (open in chrome://tracing or Perfetto)
  - Summary breakdown of time spent in each phase
  - NVTX markers for Nsight Systems (run with: nsys profile -t cuda,nvtx python profile_training.py)
  - torch.compile benchmark comparing baseline vs compiled network

Usage:
  uv run python bench/profile_training.py --num-envs 65536 --steps 50

  # For Nsight Systems (most detailed GPU profiling):
  nsys profile -t cuda,nvtx -o profile_report uv run python bench/profile_training.py
"""

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import nvtx
import torch
import warp as wp

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import NUM_RAISE_BUCKETS, PokerPolicyNet, raise_bucket_to_amount


@contextmanager
def nvtx_range(name: str, color: str = "blue"):
    """Context manager for NVTX markers."""
    rng = nvtx.start_range(name, color=color)
    try:
        yield
    finally:
        nvtx.end_range(rng)


def sync_all(device: str):
    """Full synchronization for accurate timing."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wp.synchronize()


@dataclass
class TimingResult:
    name: str
    total_ms: float
    count: int

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


class Profiler:
    """Simple profiler for tracking phase durations."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.timings: dict[str, list[float]] = {}
        self._start_time: float | None = None
        self._current_name: str | None = None

    @contextmanager
    def measure(self, name: str):
        """Measure a code block with synchronization."""
        sync_all(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync_all(self.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed_ms)

    def results(self) -> list[TimingResult]:
        """Get timing results sorted by total time."""
        results = []
        for name, times in self.timings.items():
            results.append(
                TimingResult(
                    name=name,
                    total_ms=sum(times),
                    count=len(times),
                )
            )
        return sorted(results, key=lambda r: -r.total_ms)

    def print_summary(self, title: str = "Profiling Summary"):
        """Print a formatted summary."""
        results = self.results()
        total = sum(r.total_ms for r in results)

        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        print(f"{'Phase':<35} {'Total (ms)':>10} {'Avg (ms)':>10} {'%':>6} {'Count':>6}")
        print(f"{'-' * 60}")

        for r in results:
            pct = (r.total_ms / total * 100) if total > 0 else 0
            print(f"{r.name:<35} {r.total_ms:>10.2f} {r.avg_ms:>10.3f} {pct:>5.1f}% {r.count:>6}")

        print(f"{'-' * 60}")
        print(f"{'TOTAL':<35} {total:>10.2f}")


def profile_env_only(
    num_envs: int = 65536,
    steps: int = 100,
    warmup: int = 20,
    device: str = "cuda",
) -> Profiler:
    """Profile environment stepping only (no neural network)."""

    print(f"\n[1/3] Profiling environment stepping only ({num_envs} envs, {steps} steps)...")

    wp.init()
    env = WarpPokerEnv(num_envs=num_envs, device=device)

    # Pre-allocate actions on GPU
    action_types = torch.full((num_envs,), c.ACTION_CALL, dtype=torch.int32, device=device)
    amounts = torch.zeros((num_envs,), dtype=torch.int32, device=device)

    # Warmup
    for _ in range(warmup):
        _ = env.step(action_types, amounts)
    sync_all(device)

    profiler = Profiler(device)

    for _ in range(steps):
        with profiler.measure("env.step (total)"), nvtx_range("env.step", color="green"):
            env.step(action_types, amounts)

        # Also measure sub-components
        with profiler.measure("torch->warp copy"):
            with nvtx_range("torch->warp copy", color="yellow"):
                wp.copy(env.actions, wp.from_torch(action_types, dtype=wp.int32))
                wp.copy(env.amounts, wp.from_torch(amounts, dtype=wp.int32))
                sync_all(device)

        with profiler.measure("step_kernel_per_env"):
            with nvtx_range("step_kernel", color="red"):
                from gpu_poker.env import step_kernel_per_env

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
                sync_all(device)

        with profiler.measure("get_obs_legal_kernel"):
            with nvtx_range("obs_kernel", color="blue"):
                from gpu_poker.env import get_obs_legal_kernel

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
                sync_all(device)

        with profiler.measure("warp->torch (zero-copy)"):
            with nvtx_range("warp->torch", color="orange"):
                _ = wp.to_torch(env.obs_cards)
                _ = wp.to_torch(env.obs_scalars)
                _ = wp.to_torch(env.action_mask)
                sync_all(device)

    return profiler


def profile_with_network(
    num_envs: int = 65536,
    steps: int = 50,
    warmup: int = 10,
    device: str = "cuda",
    use_amp: bool = True,
) -> Profiler:
    """Profile environment + network inference (simulating training collection)."""

    amp_str = "AMP=BF16" if use_amp else "AMP=off"
    print(
        f"\n[2/3] Profiling env + network inference ({num_envs} envs, {steps} steps, {amp_str})..."
    )

    wp.init()
    env = WarpPokerEnv(num_envs=num_envs, device=device)

    # Create policy network
    net = PokerPolicyNet(
        scalar_dim=17,
        card_embed_dim=64,
        mlp_dim=256,
        torso_layers=2,
        lstm_hidden=256,
        head_layers=2,
        head_dim=128,
    ).to(device=device)
    net.eval()

    # Hidden states
    h = torch.zeros(1, num_envs, 256, device=device)
    c_state = torch.zeros(1, num_envs, 256, device=device)

    # Warmup
    obs = env.reset()
    for _ in range(warmup):
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp),
        ):
            out = net.forward_step(
                cards=obs["cards"],
                scalars=obs["scalars"],
                action_mask=obs["action_mask"],
                h=h,
                c=c_state,
                terminated=None,
                deterministic=False,
            )
        obs = env.step(
            out.action_type.to(torch.int32), torch.zeros(num_envs, dtype=torch.int32, device=device)
        )
    sync_all(device)

    profiler = Profiler(device)

    obs = env.reset()
    for _ in range(steps):
        with profiler.measure("total_step"), nvtx_range("total_step", color="green"):
            with profiler.measure("network.forward_step"):
                with nvtx_range("network_forward", color="blue"):
                    with (
                        torch.no_grad(),
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp),
                    ):
                        out = net.forward_step(
                            cards=obs["cards"],
                            scalars=obs["scalars"],
                            action_mask=obs["action_mask"],
                            h=h,
                            c=c_state,
                            terminated=None,
                            deterministic=False,
                        )
                    h, c_state = out.h, out.c
                    sync_all(device)

            with profiler.measure("action_postprocess"):
                with nvtx_range("action_postprocess", color="yellow"):
                    action_type = out.action_type.to(dtype=torch.int32)
                    amounts = raise_bucket_to_amount(
                        raise_bucket=out.raise_bucket,
                        pot_total=obs["pot_total"],
                        min_raise=obs["min_raise"],
                        max_raise=obs["max_raise"],
                    ).to(torch.int32)
                    amounts = torch.where(
                        action_type == c.ACTION_RAISE, amounts, torch.zeros_like(amounts)
                    )
                    sync_all(device)

            with profiler.measure("env.step"), nvtx_range("env_step", color="red"):
                obs = env.step(action_type, amounts)
                sync_all(device)

    return profiler


def profile_ppo_update(
    num_envs: int = 65536,
    rollout_steps: int = 64,
    num_epochs: int = 2,
    minibatch_size: int = 2048,
    device: str = "cuda",
) -> Profiler:
    """Profile PPO update phase (forward + backward through network)."""

    print(
        f"\n[3/3] Profiling PPO update phase "
        f"({num_envs}x{rollout_steps} samples, {num_epochs} epochs)..."
    )

    total_samples = num_envs * rollout_steps
    num_minibatches = total_samples // minibatch_size

    # Create network
    net = PokerPolicyNet(
        scalar_dim=17,
        card_embed_dim=64,
        mlp_dim=256,
        torso_layers=2,
        lstm_hidden=256,
        head_layers=2,
        head_dim=128,
    ).to(device=device)
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    # Create synthetic rollout data
    cards = torch.randint(0, 52, (total_samples, 7), device=device, dtype=torch.int32)
    scalars = torch.randn(total_samples, 17, device=device)
    action_mask = torch.ones(total_samples, c.NUM_ACTIONS, device=device, dtype=torch.bool)
    h = torch.zeros(total_samples, 256, device=device)
    c_state = torch.zeros(total_samples, 256, device=device)
    old_action_type = torch.randint(0, c.NUM_ACTIONS, (total_samples,), device=device)
    old_raise_bucket = torch.randint(
        0, NUM_RAISE_BUCKETS, (total_samples,), device=device, dtype=torch.int64
    )
    old_logprob = torch.randn(total_samples, device=device)
    advantages = torch.randn(total_samples, device=device)
    returns = torch.randn(total_samples, device=device)

    sync_all(device)
    profiler = Profiler(device)

    for _epoch in range(num_epochs):
        indices = torch.randperm(total_samples, device=device)

        for mb in range(num_minibatches):
            mb_idx = indices[mb * minibatch_size : (mb + 1) * minibatch_size]

            with profiler.measure("minibatch_total"):
                with nvtx_range("minibatch", color="green"):
                    with profiler.measure("gather_minibatch"):
                        with nvtx_range("gather", color="yellow"):
                            mb_cards = cards[mb_idx]
                            mb_scalars = scalars[mb_idx]
                            mb_mask = action_mask[mb_idx]
                            mb_h = h[mb_idx].unsqueeze(0)
                            mb_c = c_state[mb_idx].unsqueeze(0)
                            mb_old_at = old_action_type[mb_idx]
                            mb_old_rb = old_raise_bucket[mb_idx]
                            mb_old_lp = old_logprob[mb_idx]
                            mb_adv = advantages[mb_idx]
                            mb_ret = returns[mb_idx]
                            sync_all(device)

                    with profiler.measure("forward_pass"):
                        with nvtx_range("forward", color="blue"):
                            out = net.evaluate_step(
                                cards=mb_cards,
                                scalars=mb_scalars,
                                action_mask=mb_mask,
                                h=mb_h,
                                c=mb_c,
                                action_type=mb_old_at,
                                raise_bucket=mb_old_rb,
                            )
                            sync_all(device)

                    with profiler.measure("loss_computation"):
                        with nvtx_range("loss", color="orange"):
                            # Simplified PPO loss
                            log_ratio = out.logprob - mb_old_lp
                            ratio = torch.exp(log_ratio)
                            pg_loss1 = -mb_adv * ratio
                            pg_loss2 = -mb_adv * torch.clamp(ratio, 0.8, 1.2)
                            policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                            value_loss = 0.5 * (out.value - mb_ret).pow(2).mean()
                            entropy = out.entropy.mean()
                            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                            sync_all(device)

                    with profiler.measure("backward_pass"):
                        with nvtx_range("backward", color="red"):
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            sync_all(device)

                    with profiler.measure("optimizer_step"):
                        with nvtx_range("optimizer", color="purple"):
                            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                            opt.step()
                            sync_all(device)

    return profiler


def run_torch_profiler(
    num_envs: int = 32768,
    steps: int = 20,
    output_dir: str = "profile_output",
    device: str = "cuda",
):
    """Run torch.profiler for detailed CUDA kernel analysis."""

    print(f"\n[Bonus] Running torch.profiler for Chrome trace ({num_envs} envs, {steps} steps)...")

    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    wp.init()
    env = WarpPokerEnv(num_envs=num_envs, device=device)

    net = PokerPolicyNet(
        scalar_dim=17,
        card_embed_dim=64,
        mlp_dim=256,
        torso_layers=2,
        lstm_hidden=256,
        head_layers=2,
        head_dim=128,
    ).to(device=device)
    net.eval()

    h = torch.zeros(1, num_envs, 256, device=device)
    c_state = torch.zeros(1, num_envs, 256, device=device)

    obs = env.reset()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            out = net.forward_step(
                cards=obs["cards"],
                scalars=obs["scalars"],
                action_mask=obs["action_mask"],
                h=h,
                c=c_state,
                terminated=None,
                deterministic=False,
            )
        obs = env.step(
            out.action_type.to(torch.int32), torch.zeros(num_envs, dtype=torch.int32, device=device)
        )
    sync_all(device)

    # Profile with torch.profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(steps):
            with torch.profiler.record_function("forward_step"):
                with torch.no_grad():
                    out = net.forward_step(
                        cards=obs["cards"],
                        scalars=obs["scalars"],
                        action_mask=obs["action_mask"],
                        h=h,
                        c=c_state,
                        terminated=None,
                        deterministic=False,
                    )
                h, c_state = out.h, out.c

            with torch.profiler.record_function("action_processing"):
                action_type = out.action_type.to(dtype=torch.int32)
                amounts = torch.zeros(num_envs, dtype=torch.int32, device=device)

            with torch.profiler.record_function("env_step"):
                obs = env.step(action_type, amounts)

    # Export trace
    trace_path = out_path / "torch_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to: {trace_path}")
    print("  Open in: chrome://tracing or https://ui.perfetto.dev/")

    # Print summary
    print("\nTop 20 CUDA operations by total time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\nTop 20 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


def benchmark_torch_compile(
    num_envs: int = 65536,
    warmup: int = 20,
    steps: int = 100,
    device: str = "cuda",
) -> None:
    """Benchmark network forward pass with different optimization combinations."""

    print(f"\n[Optimizations] Benchmarking AMP + torch.compile ({num_envs} envs, {steps} steps)...")

    # Create synthetic inputs (no env needed for pure network benchmark)
    cards = torch.randint(0, 52, (num_envs, 7), device=device, dtype=torch.int32)
    scalars = torch.randn(num_envs, 17, device=device)
    action_mask = torch.ones(num_envs, c.NUM_ACTIONS, device=device, dtype=torch.bool)
    h = torch.zeros(1, num_envs, 256, device=device)
    c_state = torch.zeros(1, num_envs, 256, device=device)

    def create_network() -> PokerPolicyNet:
        return PokerPolicyNet(
            scalar_dim=17,
            card_embed_dim=64,
            mlp_dim=256,
            torso_layers=2,
            lstm_hidden=256,
            head_layers=2,
            head_dim=128,
        ).to(device=device)

    def benchmark_forward(net: PokerPolicyNet, label: str, use_amp: bool = False) -> float:
        net.eval()

        # Warmup
        for _ in range(warmup):
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp),
            ):
                _ = net.forward_step(
                    cards=cards,
                    scalars=scalars,
                    action_mask=action_mask,
                    h=h,
                    c=c_state,
                    terminated=None,
                    deterministic=False,
                )
        sync_all(device)

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(steps):
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp),
            ):
                _ = net.forward_step(
                    cards=cards,
                    scalars=scalars,
                    action_mask=action_mask,
                    h=h,
                    c=c_state,
                    terminated=None,
                    deterministic=False,
                )
        sync_all(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / steps
        print(f"  {label}: {avg_ms:.3f} ms/step")
        return avg_ms

    results = {}

    # 1. Baseline FP32 (no compile, no AMP)
    print("\n  [1/4] Baseline FP32...")
    net1 = create_network()
    results["fp32"] = benchmark_forward(net1, "FP32 baseline", use_amp=False)

    # 2. AMP BF16 (no compile)
    print("\n  [2/4] AMP BF16...")
    net2 = create_network()
    results["bf16"] = benchmark_forward(net2, "BF16 (AMP)", use_amp=True)

    # 3. Compiled FP32
    print("\n  [3/4] Compiled FP32 (first run compiles)...")
    net3 = create_network()
    compile_t0 = time.perf_counter()
    net3 = torch.compile(net3, mode="reduce-overhead")  # type: ignore[assignment]
    with torch.no_grad():
        _ = net3.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=h,
            c=c_state,
            terminated=None,
            deterministic=False,
        )
    sync_all(device)
    compile_overhead_fp32 = (time.perf_counter() - compile_t0) * 1000
    results["compiled_fp32"] = benchmark_forward(net3, "FP32 + compile", use_amp=False)

    # 4. Compiled BF16
    print("\n  [4/4] Compiled BF16 (first run compiles)...")
    net4 = create_network()
    compile_t0 = time.perf_counter()
    net4 = torch.compile(net4, mode="reduce-overhead")  # type: ignore[assignment]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        _ = net4.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=h,
            c=c_state,
            terminated=None,
            deterministic=False,
        )
    sync_all(device)
    compile_overhead_bf16 = (time.perf_counter() - compile_t0) * 1000
    results["compiled_bf16"] = benchmark_forward(net4, "BF16 + compile", use_amp=True)

    # Summary
    print("\n" + "=" * 50)
    print(" OPTIMIZATION SUMMARY")
    print("=" * 50)
    baseline = results["fp32"]
    print(f"  FP32 baseline:      {results['fp32']:.3f} ms  (1.00x)")
    print(f"  BF16 (AMP):         {results['bf16']:.3f} ms  ({baseline / results['bf16']:.2f}x)")
    compiled_fp32_speedup = baseline / results["compiled_fp32"]
    compiled_bf16_speedup = baseline / results["compiled_bf16"]
    print(
        f"  FP32 + compile:     {results['compiled_fp32']:.3f} ms  ({compiled_fp32_speedup:.2f}x)"
    )
    print(
        f"  BF16 + compile:     {results['compiled_bf16']:.3f} ms  "
        f"({compiled_bf16_speedup:.2f}x) <-- RECOMMENDED"
    )
    print(
        f"\n  Compile overhead: FP32={compile_overhead_fp32:.0f}ms, "
        f"BF16={compile_overhead_bf16:.0f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Profile pokergpu training bottlenecks")
    parser.add_argument("--num-envs", type=int, default=65536, help="Number of parallel envs")
    parser.add_argument("--steps", type=int, default=50, help="Steps per profiling phase")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output-dir", type=str, default="profile_output", help="Output directory")
    parser.add_argument("--skip-env", action="store_true", help="Skip env-only profiling")
    parser.add_argument("--skip-network", action="store_true", help="Skip network profiling")
    parser.add_argument("--skip-ppo", action="store_true", help="Skip PPO update profiling")
    parser.add_argument("--skip-torch-profiler", action="store_true", help="Skip torch.profiler")
    parser.add_argument("--skip-compile", action="store_true", help="Skip torch.compile benchmark")
    parser.add_argument(
        "--use-amp", action="store_true", default=True, help="Use AMP (BF16) for network profiling"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP for network profiling")
    args = parser.parse_args()

    # Handle --no-amp flag
    use_amp = args.use_amp and not args.no_amp

    results = []

    # 1. Profile env stepping only
    if not args.skip_env:
        env_profiler = profile_env_only(
            num_envs=args.num_envs,
            steps=args.steps,
            device=args.device,
        )
        env_profiler.print_summary("Environment Stepping Breakdown")
        results.append(("env", env_profiler))

    # 2. Profile env + network inference
    if not args.skip_network:
        net_profiler = profile_with_network(
            num_envs=args.num_envs,
            steps=args.steps,
            device=args.device,
            use_amp=use_amp,
        )
        net_profiler.print_summary("Collection Phase Breakdown (Env + Network)")
        results.append(("network", net_profiler))

    # 3. Profile PPO update
    if not args.skip_ppo:
        ppo_profiler = profile_ppo_update(
            num_envs=args.num_envs,
            rollout_steps=64,
            num_epochs=2,
            minibatch_size=2048,
            device=args.device,
        )
        ppo_profiler.print_summary("PPO Update Breakdown")
        results.append(("ppo", ppo_profiler))

    # 4. Run torch.profiler for detailed trace
    if not args.skip_torch_profiler:
        run_torch_profiler(
            num_envs=min(args.num_envs, 32768),  # Smaller for trace size
            steps=20,
            output_dir=args.output_dir,
            device=args.device,
        )

    # 5. Benchmark torch.compile impact
    if not args.skip_compile:
        benchmark_torch_compile(
            num_envs=args.num_envs,
            steps=args.steps,
            device=args.device,
        )

    # Final summary
    print("\n" + "=" * 70)
    print(" OVERALL ANALYSIS")
    print("=" * 70)

    if results:
        print("\nKey findings from profiling:")

        for name, prof in results:
            total = sum(r.total_ms for r in prof.results())
            top = prof.results()[0] if prof.results() else None
            if top:
                pct = top.total_ms / total * 100 if total > 0 else 0
                print(f"  [{name}] Biggest bottleneck: {top.name} ({pct:.1f}% of phase)")

    print("\n" + "=" * 70)
    print(" OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on typical bottlenecks in GPU RL training:

1. KERNEL LAUNCH OVERHEAD
   - Currently: 2 kernels per step (step_kernel + obs_kernel)
   - Optimization: Fuse into single kernel
   - Expected gain: 10-20% on small batches

2. LEAGUE OPPONENT FRAGMENTATION
   - Problem: Multiple small forward passes when many snapshots active
   - Optimization: Limit active snapshots (league_rollout_snapshot_k)
   - Expected gain: 2-5x when league enabled

3. CPU-GPU SYNC POINTS
   - torch.unique(...).tolist() forces GPU->CPU transfer
   - torch.any() checks can cause implicit syncs
   - Optimization: Keep control flow on GPU where possible

4. MEMORY BANDWIDTH
   - Large observation tensors (7 cards + 17 scalars per env)
   - Optimization: Quantize observations, reduce precision

5. NETWORK THROUGHPUT
   - LSTM sequential dependency limits parallelism
   - Optimization: Use Transformer or increase batch size

To get detailed GPU kernel timing, run:
  nsys profile -t cuda,nvtx -o profile_report uv run python bench/profile_training.py

Then open profile_report.nsys-rep in Nsight Systems.
""")


if __name__ == "__main__":
    main()
