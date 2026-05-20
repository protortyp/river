#!/usr/bin/env python3
"""
Benchmark end-to-end training throughput with different optimization configurations.

Compares:
  1. Baseline (FP32, no compile)
  2. AMP only (BF16)
  3. Compile only (FP32)
  4. AMP + Compile (BF16)
  5. AMP + CUDA Graph (BF16) - captures PPO update as a graph

Usage:
  uv run python bench/bench_training_sps.py
  uv run python bench/bench_training_sps.py --num-envs 65536 --rollout-steps 64
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import warp as wp

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import PokerPolicyNet, raise_bucket_to_amount


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    wp.synchronize()


@dataclass
class BenchResult:
    label: str
    collection_sps: float
    collection_ms: float
    ppo_ms: float
    total_ms: float
    overall_sps: float


def run_training_iteration(
    *,
    env: WarpPokerEnv,
    net: PokerPolicyNet,
    opt: torch.optim.Optimizer,
    num_envs: int,
    rollout_steps: int,
    num_epochs: int,
    minibatch_size: int,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Run one collection + PPO update cycle, return (collection_ms, ppo_ms)."""

    # Hidden states
    h, c_state = PokerPolicyNet.init_state(num_envs, net.lstm_hidden, torch.device(device))

    # Storage for rollout
    obs_cards = torch.zeros((rollout_steps, num_envs, 7), dtype=torch.int32, device=device)
    obs_scalars = torch.zeros((rollout_steps, num_envs, 17), dtype=torch.float32, device=device)
    obs_mask = torch.zeros(
        (rollout_steps, num_envs, c.NUM_ACTIONS), dtype=torch.bool, device=device
    )
    actions = torch.zeros((rollout_steps, num_envs), dtype=torch.int64, device=device)
    logprobs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((rollout_steps, num_envs), dtype=torch.bool, device=device)
    h_store = torch.zeros(
        (rollout_steps, num_envs, net.lstm_hidden), dtype=torch.float32, device=device
    )
    c_store = torch.zeros(
        (rollout_steps, num_envs, net.lstm_hidden), dtype=torch.float32, device=device
    )
    raise_buckets = torch.zeros((rollout_steps, num_envs), dtype=torch.int64, device=device)

    obs = env.reset()
    _sync(device)

    # === Collection phase ===
    t0 = time.perf_counter()
    net.eval()
    with torch.no_grad():
        for t in range(rollout_steps):
            obs_cards[t] = obs["cards"]
            obs_scalars[t] = obs["scalars"]
            obs_mask[t] = obs["action_mask"]
            h_store[t] = h.squeeze(0)
            c_store[t] = c_state.squeeze(0)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
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

            actions[t] = out.action_type
            logprobs[t] = out.logprob
            values[t] = out.value
            raise_buckets[t] = out.raise_bucket

            amounts = raise_bucket_to_amount(
                raise_bucket=out.raise_bucket,
                pot_total=obs["pot_total"],
                min_raise=obs["min_raise"],
                max_raise=obs["max_raise"],
            )
            amounts = torch.where(
                out.action_type.eq(c.ACTION_RAISE),
                amounts,
                torch.zeros_like(amounts),
            )

            next_obs = env.step(out.action_type.to(dtype=torch.int32), amounts)
            rewards[t] = next_obs["rewards"].float()
            dones[t] = next_obs["terminated"]
            obs = next_obs

    _sync(device)
    collection_ms = (time.perf_counter() - t0) * 1000

    # === PPO update phase ===
    t0 = time.perf_counter()
    net.train()

    # Flatten rollout
    total = rollout_steps * num_envs
    cards_f = obs_cards.reshape(total, 7)
    scalars_f = obs_scalars.reshape(total, 17)
    mask_f = obs_mask.reshape(total, c.NUM_ACTIONS)
    actions_f = actions.reshape(total)
    old_logprobs_f = logprobs.reshape(total)
    h_f = h_store.reshape(total, net.lstm_hidden)
    c_f = c_store.reshape(total, net.lstm_hidden)
    raise_buckets_f = raise_buckets.reshape(total)

    # Simple advantage estimation (just use rewards as proxy for benchmark)
    advantages = rewards.reshape(total)
    returns = rewards.reshape(total)

    mb_size = min(minibatch_size, total)
    num_mbs = total // mb_size

    for _epoch in range(num_epochs):
        for mb_i in range(num_mbs):
            start = mb_i * mb_size
            end = start + mb_size
            sl = slice(start, end)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                eval_out = net.evaluate_step(
                    cards=cards_f[sl],
                    scalars=scalars_f[sl],
                    action_mask=mask_f[sl],
                    action_type=actions_f[sl],
                    raise_bucket=raise_buckets_f[sl],
                    h=h_f[sl].unsqueeze(0),
                    c=c_f[sl].unsqueeze(0),
                    terminated=None,
                )

            # Simplified PPO loss
            ratio = torch.exp(eval_out.logprob - old_logprobs_f[sl])
            policy_loss = -(ratio * advantages[sl]).mean()
            value_loss = 0.5 * (eval_out.value - returns[sl]).pow(2).mean()
            entropy = eval_out.entropy.mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()

    _sync(device)
    ppo_ms = (time.perf_counter() - t0) * 1000

    return collection_ms, ppo_ms


class CUDAGraphPPO:
    """Captures PPO update as a CUDA graph for faster replay."""

    def __init__(
        self,
        *,
        net: PokerPolicyNet,
        opt: torch.optim.Optimizer,
        minibatch_size: int,
        hidden_dim: int,
        device: str,
        amp_dtype: torch.dtype,
        use_amp: bool,
    ):
        self.net = net
        self.opt = opt
        self.mb_size = minibatch_size
        self.device = device
        self.amp_dtype = amp_dtype
        self.use_amp = use_amp
        self.graph: torch.cuda.CUDAGraph | None = None

        # Static input buffers (data copied here before graph replay)
        self.static_cards = torch.zeros((minibatch_size, 7), dtype=torch.int32, device=device)
        self.static_scalars = torch.zeros((minibatch_size, 17), dtype=torch.float32, device=device)
        self.static_mask = torch.zeros(
            (minibatch_size, c.NUM_ACTIONS), dtype=torch.bool, device=device
        )
        self.static_actions = torch.zeros((minibatch_size,), dtype=torch.int64, device=device)
        self.static_old_logprobs = torch.zeros(
            (minibatch_size,), dtype=torch.float32, device=device
        )
        self.static_h = torch.zeros(
            (1, minibatch_size, hidden_dim), dtype=torch.float32, device=device
        )
        self.static_c = torch.zeros(
            (1, minibatch_size, hidden_dim), dtype=torch.float32, device=device
        )
        self.static_raise_buckets = torch.zeros((minibatch_size,), dtype=torch.int64, device=device)
        self.static_advantages = torch.zeros((minibatch_size,), dtype=torch.float32, device=device)
        self.static_returns = torch.zeros((minibatch_size,), dtype=torch.float32, device=device)

    def _run_ppo_step(self) -> None:
        """Single PPO minibatch step - used for warmup and capture."""
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            eval_out = self.net.evaluate_step(
                cards=self.static_cards,
                scalars=self.static_scalars,
                action_mask=self.static_mask,
                action_type=self.static_actions,
                raise_bucket=self.static_raise_buckets,
                h=self.static_h,
                c=self.static_c,
                terminated=None,
            )

        ratio = torch.exp(eval_out.logprob - self.static_old_logprobs)
        policy_loss = -(ratio * self.static_advantages).mean()
        value_loss = 0.5 * (eval_out.value - self.static_returns).pow(2).mean()
        entropy = eval_out.entropy.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # Note: clip_grad_norm_ works in CUDA graphs as long as we don't sync
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.opt.step()

    def warmup_and_capture(self, warmup_iters: int = 3) -> None:
        """Warmup the network and capture the CUDA graph."""
        self.net.train()

        # Warmup on side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                self._run_ppo_step()
        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._run_ppo_step()

    def replay(
        self,
        cards: torch.Tensor,
        scalars: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        h: torch.Tensor,
        c_state: torch.Tensor,
        raise_buckets: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        """Copy data to static buffers and replay the graph."""
        assert self.graph is not None, "Must call warmup_and_capture first"

        # Copy to static buffers
        self.static_cards.copy_(cards)
        self.static_scalars.copy_(scalars)
        self.static_mask.copy_(mask)
        self.static_actions.copy_(actions)
        self.static_old_logprobs.copy_(old_logprobs)
        self.static_h.copy_(h)
        self.static_c.copy_(c_state)
        self.static_raise_buckets.copy_(raise_buckets)
        self.static_advantages.copy_(advantages)
        self.static_returns.copy_(returns)

        # Replay
        self.graph.replay()


def run_training_iteration_with_cuda_graph(
    *,
    env: WarpPokerEnv,
    net: PokerPolicyNet,
    opt: torch.optim.Optimizer,
    cuda_graph_ppo: CUDAGraphPPO,
    num_envs: int,
    rollout_steps: int,
    num_epochs: int,
    minibatch_size: int,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Run one collection + PPO update cycle with CUDA graph for PPO."""

    # Hidden states
    h, c_state = PokerPolicyNet.init_state(num_envs, net.lstm_hidden, torch.device(device))

    # Storage for rollout
    obs_cards = torch.zeros((rollout_steps, num_envs, 7), dtype=torch.int32, device=device)
    obs_scalars = torch.zeros((rollout_steps, num_envs, 17), dtype=torch.float32, device=device)
    obs_mask = torch.zeros(
        (rollout_steps, num_envs, c.NUM_ACTIONS), dtype=torch.bool, device=device
    )
    actions = torch.zeros((rollout_steps, num_envs), dtype=torch.int64, device=device)
    logprobs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((rollout_steps, num_envs), dtype=torch.bool, device=device)
    h_store = torch.zeros(
        (rollout_steps, num_envs, net.lstm_hidden), dtype=torch.float32, device=device
    )
    c_store = torch.zeros(
        (rollout_steps, num_envs, net.lstm_hidden), dtype=torch.float32, device=device
    )
    raise_buckets = torch.zeros((rollout_steps, num_envs), dtype=torch.int64, device=device)

    obs = env.reset()
    _sync(device)

    # === Collection phase ===
    t0 = time.perf_counter()
    net.eval()
    with torch.no_grad():
        for t in range(rollout_steps):
            obs_cards[t] = obs["cards"]
            obs_scalars[t] = obs["scalars"]
            obs_mask[t] = obs["action_mask"]
            h_store[t] = h.squeeze(0)
            c_store[t] = c_state.squeeze(0)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
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

            actions[t] = out.action_type
            logprobs[t] = out.logprob
            values[t] = out.value
            raise_buckets[t] = out.raise_bucket

            amounts = raise_bucket_to_amount(
                raise_bucket=out.raise_bucket,
                pot_total=obs["pot_total"],
                min_raise=obs["min_raise"],
                max_raise=obs["max_raise"],
            )
            amounts = torch.where(
                out.action_type.eq(c.ACTION_RAISE),
                amounts,
                torch.zeros_like(amounts),
            )

            next_obs = env.step(out.action_type.to(dtype=torch.int32), amounts)
            rewards[t] = next_obs["rewards"].float()
            dones[t] = next_obs["terminated"]
            obs = next_obs

    _sync(device)
    collection_ms = (time.perf_counter() - t0) * 1000

    # === PPO update phase with CUDA graph ===
    t0 = time.perf_counter()
    net.train()

    # Flatten rollout
    total = rollout_steps * num_envs
    cards_f = obs_cards.reshape(total, 7)
    scalars_f = obs_scalars.reshape(total, 17)
    mask_f = obs_mask.reshape(total, c.NUM_ACTIONS)
    actions_f = actions.reshape(total)
    old_logprobs_f = logprobs.reshape(total)
    h_f = h_store.reshape(total, net.lstm_hidden)
    c_f = c_store.reshape(total, net.lstm_hidden)
    raise_buckets_f = raise_buckets.reshape(total)

    # Simple advantage estimation
    advantages = rewards.reshape(total)
    returns = rewards.reshape(total)

    mb_size = min(minibatch_size, total)
    num_mbs = total // mb_size

    for _epoch in range(num_epochs):
        for mb_i in range(num_mbs):
            start = mb_i * mb_size
            end = start + mb_size

            cuda_graph_ppo.replay(
                cards=cards_f[start:end],
                scalars=scalars_f[start:end],
                mask=mask_f[start:end],
                actions=actions_f[start:end],
                old_logprobs=old_logprobs_f[start:end],
                h=h_f[start:end].unsqueeze(0),
                c_state=c_f[start:end].unsqueeze(0),
                raise_buckets=raise_buckets_f[start:end],
                advantages=advantages[start:end],
                returns=returns[start:end],
            )

    _sync(device)
    ppo_ms = (time.perf_counter() - t0) * 1000

    return collection_ms, ppo_ms


def benchmark_config(
    *,
    label: str,
    num_envs: int,
    rollout_steps: int,
    num_epochs: int,
    minibatch_size: int,
    warmup_iters: int,
    timed_iters: int,
    device: str,
    use_amp: bool,
    use_compile: bool,
    use_cuda_graph: bool = False,
) -> BenchResult:
    """Benchmark a specific configuration."""

    amp_dtype = torch.bfloat16

    wp.init()
    env = WarpPokerEnv(num_envs=num_envs, device=device)
    obs = env.reset()
    scalar_dim = int(obs["scalars"].shape[1])

    net = PokerPolicyNet(
        scalar_dim=scalar_dim,
        card_embed_dim=64,
        mlp_dim=256,
        torso_layers=2,
        lstm_hidden=256,
        head_layers=2,
        head_dim=128,
    ).to(device=device)

    if use_compile:
        net = torch.compile(net, mode="reduce-overhead")  # type: ignore[assignment]

    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, fused=True, capturable=use_cuda_graph)

    # Setup CUDA graph if enabled
    cuda_graph_ppo: CUDAGraphPPO | None = None
    if use_cuda_graph:
        cuda_graph_ppo = CUDAGraphPPO(
            net=net,
            opt=opt,
            minibatch_size=minibatch_size,
            hidden_dim=net.lstm_hidden,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        print("    Capturing CUDA graph...")
        cuda_graph_ppo.warmup_and_capture(warmup_iters=5)

    # Warmup
    print(f"    Warming up ({warmup_iters} iters)...")
    for _ in range(warmup_iters):
        if use_cuda_graph and cuda_graph_ppo is not None:
            run_training_iteration_with_cuda_graph(
                env=env,
                net=net,
                opt=opt,
                cuda_graph_ppo=cuda_graph_ppo,
                num_envs=num_envs,
                rollout_steps=rollout_steps,
                num_epochs=num_epochs,
                minibatch_size=minibatch_size,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        else:
            run_training_iteration(
                env=env,
                net=net,
                opt=opt,
                num_envs=num_envs,
                rollout_steps=rollout_steps,
                num_epochs=num_epochs,
                minibatch_size=minibatch_size,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

    # Timed runs
    print(f"    Running benchmark ({timed_iters} iters)...")
    collection_times = []
    ppo_times = []

    for _ in range(timed_iters):
        if use_cuda_graph and cuda_graph_ppo is not None:
            coll_ms, ppo_ms = run_training_iteration_with_cuda_graph(
                env=env,
                net=net,
                opt=opt,
                cuda_graph_ppo=cuda_graph_ppo,
                num_envs=num_envs,
                rollout_steps=rollout_steps,
                num_epochs=num_epochs,
                minibatch_size=minibatch_size,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        else:
            coll_ms, ppo_ms = run_training_iteration(
                env=env,
                net=net,
                opt=opt,
                num_envs=num_envs,
                rollout_steps=rollout_steps,
                num_epochs=num_epochs,
                minibatch_size=minibatch_size,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        collection_times.append(coll_ms)
        ppo_times.append(ppo_ms)

    avg_coll_ms = sum(collection_times) / len(collection_times)
    avg_ppo_ms = sum(ppo_times) / len(ppo_times)
    total_ms = avg_coll_ms + avg_ppo_ms

    env_steps_per_iter = num_envs * rollout_steps
    collection_sps = env_steps_per_iter / (avg_coll_ms / 1000)
    overall_sps = env_steps_per_iter / (total_ms / 1000)

    return BenchResult(
        label=label,
        collection_sps=collection_sps,
        collection_ms=avg_coll_ms,
        ppo_ms=avg_ppo_ms,
        total_ms=total_ms,
        overall_sps=overall_sps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark training SPS with different optimizations"
    )
    parser.add_argument("--num-envs", type=int, default=65536, help="Number of parallel envs")
    parser.add_argument("--rollout-steps", type=int, default=64, help="Steps per rollout")
    parser.add_argument("--num-epochs", type=int, default=4, help="PPO epochs")
    parser.add_argument("--minibatch-size", type=int, default=65536, help="Minibatch size")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    env_steps_per_iter = args.num_envs * args.rollout_steps

    print("=" * 70)
    print(" TRAINING SPS BENCHMARK")
    print("=" * 70)
    print(f"  num_envs: {args.num_envs}")
    print(f"  rollout_steps: {args.rollout_steps}")
    print(f"  env_steps_per_iter: {env_steps_per_iter:,}")
    print(f"  num_epochs: {args.num_epochs}")
    print(f"  minibatch_size: {args.minibatch_size}")
    print()

    # (label, use_amp, use_compile, use_cuda_graph)
    configs = [
        ("FP32 baseline", False, False, False),
        ("BF16 (AMP)", True, False, False),
        ("FP32 + compile", False, True, False),
        ("BF16 + compile", True, True, False),
        ("BF16 + CUDA graph", True, False, True),
    ]

    results = []
    for label, use_amp, use_compile, use_cuda_graph in configs:
        print(f"  [{label}]")
        result = benchmark_config(
            label=label,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            warmup_iters=args.warmup,
            timed_iters=args.iters,
            device=args.device,
            use_amp=use_amp,
            use_compile=use_compile,
            use_cuda_graph=use_cuda_graph,
        )
        results.append(result)
        print(
            f"    -> Collection: {result.collection_sps / 1e6:.2f}M SPS, PPO: {result.ppo_ms:.0f}ms"
        )
        print()

    # Summary
    print("=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)
    print()
    header = (
        f"{'Config':<20} {'Collect SPS':>12} {'Collect':>10} "
        f"{'PPO':>10} {'Total':>10} {'Overall SPS':>12}"
    )
    print(header)
    print("-" * 70)

    baseline_sps = results[0].overall_sps
    for r in results:
        speedup = r.overall_sps / baseline_sps
        row = (
            f"{r.label:<20} {r.collection_sps / 1e6:>10.2f}M "
            f"{r.collection_ms:>9.0f}ms {r.ppo_ms:>9.0f}ms "
            f"{r.total_ms:>9.0f}ms {r.overall_sps / 1e6:>10.2f}M "
            f"({speedup:.2f}x)"
        )
        print(row)

    print("-" * 70)
    print()
    best = max(results, key=lambda r: r.overall_sps)
    print(f"  Best config: {best.label}")
    print(f"  Overall SPS: {best.overall_sps / 1e6:.2f}M steps/sec")
    print(f"  Speedup vs baseline: {best.overall_sps / baseline_sps:.2f}x")
    print()


if __name__ == "__main__":
    main()
