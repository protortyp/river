import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import warp as wp

from gpu_poker import constants as c
from gpu_poker.env import get_obs_legal_kernel, step_kernel
from gpu_poker.kernels.policy_sampling import sample_actions_kernel
from gpu_poker.torchrl_env import WarpPokerTorchRLEnv

try:
    from tensordict import TensorDict

    _HAS_TD = True
except ModuleNotFoundError:  # pragma: no cover
    TensorDict = None  # type: ignore[assignment]
    _HAS_TD = False


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample_action_type_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: bool [N, 4]
    returns: int64 [N]
    """
    # Convert to probabilities; fold is always legal, so sum >= 1.
    probs = mask.to(dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int64)


@dataclass(frozen=True)
class SmokeConfig:
    device: str
    num_envs: int
    warmup_steps: int
    steps: int
    seed: int


@dataclass(frozen=True)
class SmokeResult:
    config: dict
    elapsed_s: float
    elapsed_gpu_s: float | None
    env_steps: int
    sps: float
    invalid_rate: float
    terminated_rate: float
    breakdown_gpu_s: dict[str, float]


def run(cfg: SmokeConfig) -> SmokeResult:
    if not _HAS_TD:  # pragma: no cover
        raise ModuleNotFoundError("tensordict not installed")

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    env = WarpPokerTorchRLEnv(num_envs=cfg.num_envs, device=cfg.device)
    td = env.reset()

    # Use Warp sampling on CUDA to avoid torch multinomial/rand overhead.
    use_warp_sampling = cfg.device.startswith("cuda") and torch.cuda.is_available()

    if use_warp_sampling:
        base_env = env._env  # noqa: SLF001 (intentional: smoke script)
        # Buffers are sampled directly into `base_env.actions/base_env.amounts` on GPU.

        @wp.kernel
        def fill_constant_actions(
            actions_arr: wp.array(dtype=wp.int32),
            amounts_arr: wp.array(dtype=wp.int32),
        ):
            i = wp.tid()
            actions_arr[i] = c.ACTION_CALL
            amounts_arr[i] = 0

        def time_block_gpu(fn, iters: int) -> float:
            wp.synchronize()
            start_evt = wp.Event(cfg.device, enable_timing=True)
            end_evt = wp.Event(cfg.device, enable_timing=True)
            wp.record_event(start_evt)
            for _ in range(iters):
                fn()
            wp.record_event(end_evt)
            wp.synchronize_event(end_evt)
            return wp.get_event_elapsed_time(start_evt, end_evt) / 1000.0

    # Warmup
    for _ in range(cfg.warmup_steps):
        if use_warp_sampling:
            wp.launch(
                sample_actions_kernel,
                dim=cfg.num_envs,
                inputs=[
                    base_env.state.rng_state,
                    base_env.action_mask,
                    base_env.min_raise,
                    base_env.max_raise,
                    base_env.actions,
                    base_env.amounts,
                ],
                device=cfg.device,
            )
            td = env.step_from_env_buffers()
        else:
            action_type = _sample_action_type_from_mask(td["action_mask"])
            raise_frac = torch.rand((cfg.num_envs, 1), device=cfg.device, dtype=torch.float32)
            a = TensorDict(
                {"action_type": action_type, "raise_frac": raise_frac},
                batch_size=[cfg.num_envs],
            )
            td = TensorDict({"action": a}, batch_size=[cfg.num_envs], device=cfg.device)
            td = env.step(td)["next"]

    _sync(cfg.device)

    invalid_count = 0
    terminated_count = 0
    breakdown_gpu_s: dict[str, float] = {}
    elapsed_gpu_s: float | None = None

    if use_warp_sampling:

        def launch_sampling():
            wp.launch(
                sample_actions_kernel,
                dim=cfg.num_envs,
                inputs=[
                    base_env.state.rng_state,
                    base_env.action_mask,
                    base_env.min_raise,
                    base_env.max_raise,
                    base_env.actions,
                    base_env.amounts,
                ],
                device=cfg.device,
            )

        def launch_step_only():
            wp.launch(
                kernel=step_kernel,
                dim=cfg.num_envs,
                inputs=[
                    base_env.state,
                    base_env.actions,
                    base_env.amounts,
                    base_env.rewards,
                    base_env.invalid_action,
                    base_env.terminated,
                    base_env.primes,
                    base_env.unsuited_lut,
                    base_env.flush_lut,
                    base_env.starting_stack,
                    base_env.small_blind,
                    base_env.big_blind,
                ],
                device=cfg.device,
            )

        def launch_post_only():
            wp.launch(
                kernel=get_obs_legal_kernel,
                dim=cfg.num_envs,
                inputs=[
                    base_env.state,
                    base_env.obs_cards,
                    base_env.obs_scalars,
                    base_env.primes,
                    base_env.unsuited_lut,
                    base_env.flush_lut,
                    base_env.action_mask,
                    base_env.min_raise,
                    base_env.max_raise,
                ],
                device=cfg.device,
            )

        # Microbench breakdown (measured in isolation, not during the main loop).
        breakdown_gpu_s["sampling_only"] = time_block_gpu(launch_sampling, cfg.steps)

        wp.launch(
            fill_constant_actions,
            dim=cfg.num_envs,
            inputs=[base_env.actions, base_env.amounts],
            device=cfg.device,
        )
        breakdown_gpu_s["step_only"] = time_block_gpu(launch_step_only, cfg.steps)
        breakdown_gpu_s["post_only"] = time_block_gpu(launch_post_only, cfg.steps)

    t0 = time.perf_counter()
    if use_warp_sampling:
        wp.synchronize()
        start_evt = wp.Event(cfg.device, enable_timing=True)
        end_evt = wp.Event(cfg.device, enable_timing=True)
        wp.record_event(start_evt)

    for _ in range(cfg.steps):
        if use_warp_sampling:
            wp.launch(
                sample_actions_kernel,
                dim=cfg.num_envs,
                inputs=[
                    base_env.state.rng_state,
                    base_env.action_mask,
                    base_env.min_raise,
                    base_env.max_raise,
                    base_env.actions,
                    base_env.amounts,
                ],
                device=cfg.device,
            )
            td = env.step_from_env_buffers()
        else:
            action_type = _sample_action_type_from_mask(td["action_mask"])
            raise_frac = torch.rand((cfg.num_envs, 1), device=cfg.device, dtype=torch.float32)
            a = TensorDict(
                {"action_type": action_type, "raise_frac": raise_frac},
                batch_size=[cfg.num_envs],
            )
            td = TensorDict({"action": a}, batch_size=[cfg.num_envs], device=cfg.device)
            td = env.step(td)["next"]

        invalid_count += int(td["invalid_action"].sum().item())
        terminated_count += int(td["terminated"].sum().item())

    if use_warp_sampling:
        wp.record_event(end_evt)

    _sync(cfg.device)
    t1 = time.perf_counter()
    if use_warp_sampling:
        wp.synchronize_event(end_evt)
        elapsed_gpu_s = wp.get_event_elapsed_time(start_evt, end_evt) / 1000.0

    elapsed_s = t1 - t0
    env_steps = cfg.num_envs * cfg.steps
    sps = env_steps / elapsed_s if elapsed_s > 0 else float("inf")

    invalid_rate = invalid_count / env_steps
    terminated_rate = terminated_count / env_steps

    return SmokeResult(
        config=asdict(cfg),
        elapsed_s=elapsed_s,
        elapsed_gpu_s=elapsed_gpu_s,
        env_steps=env_steps,
        sps=sps,
        invalid_rate=invalid_rate,
        terminated_rate=terminated_rate,
        breakdown_gpu_s=breakdown_gpu_s,
    )


def main() -> int:
    device = os.environ.get("POKERGPU_SMOKE_DEVICE", _default_device())
    num_envs = int(os.environ.get("POKERGPU_SMOKE_NUM_ENVS", "131072"))
    warmup_steps = int(os.environ.get("POKERGPU_SMOKE_WARMUP", "50"))
    steps = int(os.environ.get("POKERGPU_SMOKE_STEPS", "500"))
    seed = int(os.environ.get("POKERGPU_SMOKE_SEED", "0"))

    cfg = SmokeConfig(
        device=device,
        num_envs=num_envs,
        warmup_steps=warmup_steps,
        steps=steps,
        seed=seed,
    )
    result = run(cfg)

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "torch": getattr(torch, "__version__", "unknown"),
        },
        "result": asdict(result),
    }

    out_dir = Path("train/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"torchrl_smoke_{ts}_{socket.gethostname()}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    Path("train/latest.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote: {out_path}")
    print(
        f"SPS={result.sps:.2f} invalid_rate={result.invalid_rate:.6f} "
        f"terminated_rate={result.terminated_rate:.6f}"
    )
    if result.elapsed_gpu_s is not None:
        print(f"elapsed_gpu_s={result.elapsed_gpu_s:.6f}")
    if result.breakdown_gpu_s:
        print(f"breakdown_gpu_s={result.breakdown_gpu_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
