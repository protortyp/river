import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import warp as wp

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv, get_obs_legal_kernel, step_kernel


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    wp.synchronize()


def _uses_cuda_events(device: str) -> bool:
    return device.startswith("cuda")


def _default_device() -> str:
    try:
        if wp.is_device_available("cuda") and torch.cuda.is_available():
            return "cuda:0"
    except RuntimeError:
        pass
    return "cpu"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    mode: str
    action_source: str  # torch_constant | torch_random | warp_constant | warp_random
    include_obs: bool
    include_legal: bool


@dataclass(frozen=True)
class BenchmarkResult:
    case: str
    num_envs: int
    warmup_steps: int
    timed_steps: int
    elapsed_s: float
    elapsed_gpu_s: float
    env_steps: int
    sps: float
    breakdown_s: dict[str, float]


@wp.kernel
def _fill_constant_actions(
    action_types: wp.array(dtype=wp.int32),
    amounts: wp.array(dtype=wp.int32),
    action_type: wp.int32,
    amount: wp.int32,
):
    i = wp.tid()
    action_types[i] = action_type
    amounts[i] = amount


@wp.kernel
def _fill_random_actions(
    rng_state: wp.array(dtype=wp.uint32),
    action_types: wp.array(dtype=wp.int32),
    amounts: wp.array(dtype=wp.int32),
    max_amount: wp.int32,
):
    i = wp.tid()
    seed = rng_state[i]
    r = wp.rand_init(wp.int32(seed))
    action_types[i] = wp.int32(wp.randi(r, 0, c.NUM_ACTIONS))
    amounts[i] = wp.int32(wp.randi(r, 0, max_amount + 1))
    rng_state[i] = seed * wp.uint32(1664525) + wp.uint32(1013904223)


def _torch_actions_constant(
    mode: str, num_envs: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "call_check":
        a = torch.full((num_envs,), c.ACTION_CALL, dtype=torch.int32, device=device)
        amt = torch.zeros((num_envs,), dtype=torch.int32, device=device)
        return a, amt
    if mode == "fold":
        a = torch.full((num_envs,), c.ACTION_FOLD, dtype=torch.int32, device=device)
        amt = torch.zeros((num_envs,), dtype=torch.int32, device=device)
        return a, amt
    raise ValueError(mode)


def _torch_actions_random(
    num_envs: int, device: str, starting_stack: int
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randint(0, c.NUM_ACTIONS, (num_envs,), dtype=torch.int32, device=device)
    amt = torch.randint(0, starting_stack, (num_envs,), dtype=torch.int32, device=device)
    return a, amt


def _bench_env_step(
    env: WarpPokerEnv,
    *,
    mode: str,
    warmup: int,
    steps: int,
    action_source: str,
) -> BenchmarkResult:
    if action_source == "torch_constant":
        action_types, amounts = _torch_actions_constant(mode, env.num_envs, env.device)
    elif action_source == "torch_random":
        action_types, amounts = _torch_actions_random(env.num_envs, env.device, env.starting_stack)
    else:
        raise ValueError(action_source)

    for _ in range(warmup):
        if action_source == "torch_random":
            action_types, amounts = _torch_actions_random(
                env.num_envs, env.device, env.starting_stack
            )
        env.step(action_types, amounts)

    _sync(env.device)
    use_cuda_events = _uses_cuda_events(env.device)
    if use_cuda_events:
        start_evt = wp.Event(env.device, enable_timing=True)
        end_evt = wp.Event(env.device, enable_timing=True)

    t0 = time.perf_counter()
    if use_cuda_events:
        wp.record_event(start_evt)
    for _ in range(steps):
        if action_source == "torch_random":
            action_types, amounts = _torch_actions_random(
                env.num_envs, env.device, env.starting_stack
            )
        env.step(action_types, amounts)

    if use_cuda_events:
        wp.record_event(end_evt)
    _sync(env.device)
    if use_cuda_events:
        wp.synchronize_event(end_evt)
    t1 = time.perf_counter()

    elapsed_s = t1 - t0
    elapsed_gpu_s = (
        wp.get_event_elapsed_time(start_evt, end_evt) / 1000.0 if use_cuda_events else elapsed_s
    )
    env_steps = env.num_envs * steps
    sps = env_steps / elapsed_s

    return BenchmarkResult(
        case=f"env.step::{mode}::{action_source}",
        num_envs=env.num_envs,
        warmup_steps=warmup,
        timed_steps=steps,
        elapsed_s=elapsed_s,
        elapsed_gpu_s=elapsed_gpu_s,
        env_steps=env_steps,
        sps=sps,
        breakdown_s={},
    )


def _bench_kernels(
    env: WarpPokerEnv,
    *,
    mode: str,
    warmup: int,
    steps: int,
    action_source: str,
    include_obs: bool,
    include_legal: bool,
) -> BenchmarkResult:
    d = env.device

    def fill_actions():
        if action_source == "warp_constant":
            if mode == "call_check":
                at, amt = c.ACTION_CALL, 0
            elif mode == "fold":
                at, amt = c.ACTION_FOLD, 0
            else:
                raise ValueError(mode)
            wp.launch(
                _fill_constant_actions,
                dim=env.num_envs,
                inputs=[env.actions, env.amounts, at, amt],
                device=d,
            )
        elif action_source == "warp_random":
            wp.launch(
                _fill_random_actions,
                dim=env.num_envs,
                inputs=[env.state.rng_state, env.actions, env.amounts, env.starting_stack],
                device=d,
            )
        else:
            raise ValueError(action_source)

    def run_step():
        wp.launch(
            kernel=step_kernel,
            dim=env.num_envs,
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
                env.starting_stack,
                env.small_blind,
                env.big_blind,
            ],
            device=d,
        )

    def run_post():
        if not include_obs and not include_legal:
            return
        wp.launch(
            kernel=get_obs_legal_kernel,
            dim=env.num_envs,
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
            device=d,
        )

    def time_block(fn, iters: int) -> float:
        _sync(d)
        use_cuda_events = _uses_cuda_events(d)
        if use_cuda_events:
            start_evt = wp.Event(d, enable_timing=True)
            end_evt = wp.Event(d, enable_timing=True)
            wp.record_event(start_evt)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if use_cuda_events:
            wp.record_event(end_evt)
        _sync(d)
        if use_cuda_events:
            wp.synchronize_event(end_evt)
            return wp.get_event_elapsed_time(start_evt, end_evt) / 1000.0
        return time.perf_counter() - t0

    # Warmup
    for _ in range(warmup):
        fill_actions()
        run_step()
        run_post()
    _sync(d)

    # Total pipeline timings (GPU + wall)
    use_cuda_events = _uses_cuda_events(d)
    if use_cuda_events:
        start_evt_total = wp.Event(d, enable_timing=True)
        end_evt_total = wp.Event(d, enable_timing=True)
    t0 = time.perf_counter()
    if use_cuda_events:
        wp.record_event(start_evt_total)
    for _ in range(steps):
        fill_actions()
        run_step()
        run_post()
    if use_cuda_events:
        wp.record_event(end_evt_total)
    _sync(d)
    if use_cuda_events:
        wp.synchronize_event(end_evt_total)
    t1 = time.perf_counter()

    elapsed_s = t1 - t0
    elapsed_gpu_s = (
        wp.get_event_elapsed_time(start_evt_total, end_evt_total) / 1000.0
        if use_cuda_events
        else elapsed_s
    )
    env_steps = env.num_envs * steps
    sps = env_steps / elapsed_s

    # Microbench-style breakdown: each block measured alone.
    breakdown = {
        "fill_actions": time_block(fill_actions, steps),
        "step": time_block(run_step, steps),
        "post(obs+legal)": time_block(run_post, steps) if include_obs or include_legal else 0.0,
    }

    return BenchmarkResult(
        case=f"kernels::{mode}::{action_source}::obs={include_obs}::legal={include_legal}",
        num_envs=env.num_envs,
        warmup_steps=warmup,
        timed_steps=steps,
        elapsed_s=elapsed_s,
        elapsed_gpu_s=elapsed_gpu_s,
        env_steps=env_steps,
        sps=sps,
        breakdown_s=breakdown,
    )


def _git_info() -> dict[str, str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        return {"commit": sha, "dirty": str(dirty)}
    except Exception:
        return {"commit": "unknown", "dirty": "unknown"}


def main() -> int:
    wp.init()
    device = _default_device()

    num_envs = int(os.environ.get("POKERGPU_BENCH_NUM_ENVS", c.DEFAULT_NUM_ENVS))
    warmup = int(os.environ.get("POKERGPU_BENCH_WARMUP", "50"))
    steps = int(os.environ.get("POKERGPU_BENCH_STEPS", "500"))

    env = WarpPokerEnv(num_envs=num_envs, device=device)
    env.reset()
    _sync(device)

    cases = [
        BenchmarkCase(
            name="env_step_call_check_torch_constant",
            mode="call_check",
            action_source="torch_constant",
            include_obs=True,
            include_legal=True,
        ),
        BenchmarkCase(
            name="env_step_random_torch_random",
            mode="call_check",
            action_source="torch_random",
            include_obs=True,
            include_legal=True,
        ),
        BenchmarkCase(
            name="kernels_call_check_warp_constant_full",
            mode="call_check",
            action_source="warp_constant",
            include_obs=True,
            include_legal=True,
        ),
        BenchmarkCase(
            name="kernels_call_check_warp_constant_step_only",
            mode="call_check",
            action_source="warp_constant",
            include_obs=False,
            include_legal=False,
        ),
        BenchmarkCase(
            name="kernels_random_warp_random_full",
            mode="call_check",
            action_source="warp_random",
            include_obs=True,
            include_legal=True,
        ),
    ]

    results: list[BenchmarkResult] = []
    for case in cases:
        if case.action_source.startswith("torch_"):
            results.append(
                _bench_env_step(
                    env,
                    mode=case.mode,
                    warmup=warmup,
                    steps=steps,
                    action_source=case.action_source,
                )
            )
        else:
            results.append(
                _bench_kernels(
                    env,
                    mode=case.mode,
                    warmup=warmup,
                    steps=steps,
                    action_source=case.action_source,
                    include_obs=case.include_obs,
                    include_legal=case.include_legal,
                )
            )

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unknown"),
        "warp": getattr(wp, "__version__", "unknown"),
        "device": device,
        "git": _git_info(),
    }
    payload = {"meta": meta, "results": [asdict(r) for r in results]}

    out_dir = Path("bench/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"bench_{ts}_{socket.gethostname()}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    Path("bench/latest.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote: {out_path}")
    for r in results:
        print(
            f"{r.case} SPS={r.sps:.2f} elapsed_s={r.elapsed_s:.6f} "
            f"elapsed_gpu_s={r.elapsed_gpu_s:.6f}"
        )
        if r.breakdown_s:
            print(f"  breakdown_s: {r.breakdown_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
