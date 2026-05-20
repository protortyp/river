import argparse
import time

import torch
import warp as wp

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    wp.synchronize()


def _make_actions(
    *,
    mode: str,
    num_envs: int,
    device: str,
    starting_stack: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "fold":
        action_types = torch.full((num_envs,), c.ACTION_FOLD, dtype=torch.int32, device=device)
        amounts = torch.zeros((num_envs,), dtype=torch.int32, device=device)
        return action_types, amounts

    if mode == "call_check":
        action_types = torch.full((num_envs,), c.ACTION_CALL, dtype=torch.int32, device=device)
        amounts = torch.zeros((num_envs,), dtype=torch.int32, device=device)
        return action_types, amounts

    if mode == "random":
        action_types = torch.randint(
            low=0,
            high=c.NUM_ACTIONS,
            size=(num_envs,),
            dtype=torch.int32,
            device=device,
        )
        amounts = torch.randint(
            low=0,
            high=starting_stack,
            size=(num_envs,),
            dtype=torch.int32,
            device=device,
        )
        return action_types, amounts

    raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU poker environment throughput benchmark.")
    parser.add_argument("--device", type=str, default="cuda", help="Warp/Torch device, e.g. cuda")
    parser.add_argument("--num-envs", type=int, default=131072, help="Parallel env count")
    parser.add_argument("--steps", type=int, default=200, help="Timed steps to run")
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup steps (excluded from timing)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="call_check",
        choices=["fold", "call_check", "random"],
        help="Action generation pattern",
    )
    parser.add_argument("--starting-stack", type=int, default=c.STARTING_STACK)
    parser.add_argument("--small-blind", type=int, default=c.SMALL_BLIND)
    parser.add_argument("--big-blind", type=int, default=c.BIG_BLIND)
    args = parser.parse_args()

    wp.init()

    env = WarpPokerEnv(
        num_envs=args.num_envs,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        device=args.device,
    )

    obs = env.reset()
    _ = obs["cards"], obs["scalars"], obs["action_mask"], obs["min_raise"], obs["max_raise"]

    action_types, amounts = _make_actions(
        mode=args.mode,
        num_envs=args.num_envs,
        device=args.device,
        starting_stack=args.starting_stack,
    )

    # Warmup (JIT compile + caches).
    for _ in range(args.warmup):
        if args.mode == "random":
            action_types, amounts = _make_actions(
                mode=args.mode,
                num_envs=args.num_envs,
                device=args.device,
                starting_stack=args.starting_stack,
            )
        _ = env.step(action_types, amounts)

    _sync(args.device)

    # Timed section.
    t0 = time.perf_counter()
    for _ in range(args.steps):
        if args.mode == "random":
            action_types, amounts = _make_actions(
                mode=args.mode,
                num_envs=args.num_envs,
                device=args.device,
                starting_stack=args.starting_stack,
            )
        _ = env.step(action_types, amounts)
    _sync(args.device)
    t1 = time.perf_counter()

    elapsed_s = t1 - t0
    total_env_steps = args.num_envs * args.steps
    sps = total_env_steps / elapsed_s if elapsed_s > 0 else float("inf")

    print("=== pokergpu benchmark ===")
    print(f"device: {args.device}")
    print(f"num_envs: {args.num_envs}")
    print(f"warmup_steps: {args.warmup}")
    print(f"timed_steps: {args.steps}")
    print(f"mode: {args.mode}")
    print(f"elapsed_s: {elapsed_s:.6f}")
    print(f"env_steps: {total_env_steps}")
    print(f"SPS: {sps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
