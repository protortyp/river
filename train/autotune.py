"""
Autotune training "load" parameters to maximize GPU utilization without OOM.

Two-stage strategy:
  1. ``probe_peak(num_envs, ...)``  — actually allocates env + all concurrent
     nets + rollout buffer with the *real* backbone (LSTM or transformer),
     runs a representative forward/backward, and returns peak VRAM.
  2. ``autotune_num_envs(cfg)``     — does a two-point calibration at two
     small sizes, fits the linear memory model
         M(N) = fixed + per_env * N
     then solves for the largest N whose predicted peak leaves a safety
     margin under total VRAM. Linear extrapolation is the right model
     because the dominant memory terms (rollout buffer + activations) are
     all O(num_envs); only model weights + Adam moments + CUDA context
     are constant.

This replaces the prior boolean OK/fail binary search, which:
  - Always used the LSTM net even when policy_backbone=transformer
  - Only allocated one net (triad has three loaded concurrently)
  - Returned OK based on "didn't OOM at probe time", which under-counted
    the real training peak by 3-4x.

Standalone CLI use is unchanged:
  uv run python train/autotune.py --device cuda:0
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch

from gpu_poker.env import WarpPokerEnv


def _mem(device: str) -> tuple[int, int] | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(torch.device(device))
    return int(free), int(total)


def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PiB"


def _build_net(
    *,
    policy_backbone: str,
    scalar_dim: int,
    card_embed_dim: int,
    mlp_dim: int,
    torso_layers: int,
    lstm_hidden: int,
    head_layers: int,
    head_dim: int | None,
    transformer_d_model: int,
    transformer_n_heads: int,
    transformer_n_layers: int,
    transformer_ffn_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    """Construct the policy net matching the user's config."""
    if policy_backbone == "transformer":
        from gpu_poker.policy_transformer import PokerTransformerPolicyNet

        return PokerTransformerPolicyNet(
            scalar_dim=scalar_dim,
            card_embed_dim=card_embed_dim,
            d_model=transformer_d_model,
            n_heads=transformer_n_heads,
            n_layers=transformer_n_layers,
            ffn_dim=transformer_ffn_dim,
        ).to(device=device)
    from gpu_poker.policy import PokerPolicyNet

    return PokerPolicyNet(
        scalar_dim=scalar_dim,
        card_embed_dim=card_embed_dim,
        mlp_dim=mlp_dim,
        torso_layers=torso_layers,
        lstm_hidden=lstm_hidden,
        head_layers=head_layers,
        head_dim=head_dim,
    ).to(device=device)


def probe_peak(
    *,
    device: str,
    num_envs: int,
    rollout_steps: int,
    include_eval_env: bool,
    n_concurrent_nets: int,
    policy_backbone: str = "lstm",
    card_embed_dim: int = 64,
    mlp_dim: int = 256,
    torso_layers: int = 3,
    lstm_hidden: int = 256,
    head_layers: int = 2,
    head_dim: int | None = None,
    transformer_d_model: int = 256,
    transformer_n_heads: int = 4,
    transformer_n_layers: int = 4,
    transformer_ffn_dim: int = 1024,
    num_epochs: int = 4,
) -> int | None:
    """Run a representative training step and return peak VRAM in bytes.

    Returns ``None`` if the probe itself OOMs at this ``num_envs`` (caller
    should pick a smaller probe size). Returns peak ``cuda.max_memory_allocated``
    on success.

    The probe approximates the real training pattern by:
      - Building ``n_concurrent_nets`` copies of the policy (3 for triad mode)
      - Using the correct backbone (transformer vs LSTM)
      - Allocating the rollout buffer with the right state width
      - Running ``num_epochs`` forward+backward passes on each net to match
        PPO's update peak (Adam moments + cached activations across epochs)
    """
    torch_device = torch.device(device)
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=torch_device)

    try:
        env = WarpPokerEnv(num_envs=num_envs, device=device)
        obs = env.reset()
        scalar_dim = int(obs["scalars"].shape[1])
        mask_dim = int(obs["action_mask"].shape[1])

        nets: list[torch.nn.Module] = []
        opts: list[torch.optim.Optimizer] = []
        for _ in range(n_concurrent_nets):
            net = _build_net(
                policy_backbone=policy_backbone,
                scalar_dim=scalar_dim,
                card_embed_dim=card_embed_dim,
                mlp_dim=mlp_dim,
                torso_layers=torso_layers,
                lstm_hidden=lstm_hidden,
                head_layers=head_layers,
                head_dim=head_dim,
                transformer_d_model=transformer_d_model,
                transformer_n_heads=transformer_n_heads,
                transformer_n_layers=transformer_n_layers,
                transformer_ffn_dim=transformer_ffn_dim,
                device=torch_device,
            )
            nets.append(net)
            opts.append(torch.optim.Adam(net.parameters(), lr=3e-4))

        # Rollout buffer (per role in triad; size is per-net's env partition).
        # In triad mode each role owns ~num_envs/n_concurrent_nets envs, but
        # all the per-step storage stays alive throughout the rollout window.
        # We approximate the total buffer footprint by allocating the buffer
        # ONCE at full num_envs (largest role + accounting for all-role peak).
        f32 = torch.float32
        # Use the first net's state width (`lstm_hidden` property is overridden
        # to STATE_WIDTH for the transformer, so this is backbone-correct).
        state_width = int(nets[0].lstm_hidden)  # type: ignore[attr-defined]
        _bufs = [
            torch.empty((rollout_steps, num_envs, 7), device=torch_device, dtype=torch.int32),
            torch.empty((rollout_steps, num_envs, scalar_dim), device=torch_device, dtype=f32),
            torch.empty((rollout_steps, num_envs, mask_dim), device=torch_device, dtype=torch.bool),
            torch.empty((rollout_steps, num_envs), device=torch_device, dtype=torch.int64),
            torch.empty((rollout_steps, num_envs, 1), device=torch_device, dtype=f32),
            torch.empty((rollout_steps, num_envs), device=torch_device, dtype=f32),
            torch.empty((rollout_steps, num_envs), device=torch_device, dtype=f32),
            torch.empty((rollout_steps, num_envs), device=torch_device, dtype=torch.bool),
            torch.empty((rollout_steps, num_envs), device=torch_device, dtype=torch.int64),
            torch.empty((rollout_steps, num_envs, state_width), device=torch_device, dtype=f32),
            torch.empty((rollout_steps, num_envs, state_width), device=torch_device, dtype=f32),
        ]
        if include_eval_env:
            eval_env = WarpPokerEnv(num_envs=min(1024, num_envs), device=device)
            _ = eval_env.reset()

        # Inputs (same shape for every net).
        cards = obs["cards"].to(dtype=torch.int32)
        scalars = obs["scalars"].to(dtype=torch.float32)
        mask = obs["action_mask"].to(dtype=torch.bool)
        action_type = torch.zeros((num_envs,), device=torch_device, dtype=torch.int64)
        raise_bucket = torch.zeros((num_envs,), device=torch_device, dtype=torch.int64)
        returns = torch.zeros((num_envs,), device=torch_device, dtype=torch.float32)
        adv = torch.zeros((num_envs,), device=torch_device, dtype=torch.float32)

        # Run num_epochs forward+backward passes on each net. This matches PPO's
        # update-phase pattern (Adam moments + cached optimizer state across
        # epochs) more faithfully than a single backward.
        for net, opt in zip(nets, opts, strict=True):
            h, c = type(net).init_state(num_envs, state_width, torch_device)  # type: ignore[attr-defined]
            for _epoch in range(max(1, num_epochs)):
                out = net.evaluate_step(  # type: ignore[attr-defined]
                    cards=cards,
                    scalars=scalars,
                    action_mask=mask,
                    action_type=action_type,
                    raise_bucket=raise_bucket,
                    h=h,
                    c=c,
                    terminated=None,
                )
                ratio = torch.exp(out.logprob.clamp(-20.0, 20.0))
                policy_loss = -(ratio * adv).mean()
                value_loss = 0.5 * (out.value - returns).pow(2).mean()
                entropy_bonus = out.entropy.mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        if is_cuda:
            torch.cuda.synchronize(device=torch_device)
            return int(torch.cuda.max_memory_allocated(device=torch_device))
        return 0
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "failed to allocate" in msg:
            return None
        raise
    finally:
        if is_cuda:
            torch.cuda.empty_cache()


def autotune_num_envs(
    *,
    device: str,
    rollout_steps: int,
    include_eval_env: bool,
    n_concurrent_nets: int,
    policy_backbone: str = "lstm",
    card_embed_dim: int = 64,
    mlp_dim: int = 256,
    torso_layers: int = 3,
    lstm_hidden: int = 256,
    head_layers: int = 2,
    head_dim: int | None = None,
    transformer_d_model: int = 256,
    transformer_n_heads: int = 4,
    transformer_n_layers: int = 4,
    transformer_ffn_dim: int = 1024,
    num_epochs: int = 4,
    ceiling: int = 262144,
    safety_fraction: float = 0.85,
    min_envs: int = 1024,
) -> int:
    """Two-point memory calibration to find a safe max ``num_envs``.

    1. Probe at two small sizes (4096 and 8192 by default; backs off on OOM).
    2. Fit ``M(N) = fixed + per_env * N`` from the two peak measurements.
    3. Solve ``M(N) <= safety_fraction * total_vram`` for largest N.
    4. Round to a multiple of 256 (matches the rest of the codebase's grain).

    Returns the recommended ``num_envs``.
    """
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        # No GPU to autotune for; just return ceiling.
        return ceiling

    torch_device = torch.device(device)
    _, total = torch.cuda.mem_get_info(torch_device)
    total_int = int(total)
    budget = int(total_int * float(safety_fraction))

    common_kwargs: dict[str, Any] = {
        "device": device,
        "rollout_steps": rollout_steps,
        "include_eval_env": include_eval_env,
        "n_concurrent_nets": n_concurrent_nets,
        "policy_backbone": policy_backbone,
        "card_embed_dim": card_embed_dim,
        "mlp_dim": mlp_dim,
        "torso_layers": torso_layers,
        "lstm_hidden": lstm_hidden,
        "head_layers": head_layers,
        "head_dim": head_dim,
        "transformer_d_model": transformer_d_model,
        "transformer_n_heads": transformer_n_heads,
        "transformer_n_layers": transformer_n_layers,
        "transformer_ffn_dim": transformer_ffn_dim,
        "num_epochs": num_epochs,
    }

    # Pick two probe sizes that comfortably fit on any modern GPU. If a probe
    # OOMs we halve it; this can only happen on a very small GPU + huge
    # model config, in which case we'll converge to something tiny.
    probe_sizes = [4096, 8192]
    measurements: list[tuple[int, int]] = []  # (num_envs, peak_bytes)
    for size in probe_sizes:
        cur = size
        while cur >= 256:
            t0 = time.perf_counter()
            peak = probe_peak(num_envs=cur, **common_kwargs)
            dt = time.perf_counter() - t0
            if peak is None:
                cur //= 2
                print(f"[autotune] probe(num_envs={cur * 2}) OOM in {dt:.1f}s, halving")
                continue
            measurements.append((cur, peak))
            print(f"[autotune] probe(num_envs={cur:>6}) peak={_fmt_bytes(peak)} ({dt:.1f}s)")
            break
        if cur < 256:
            raise RuntimeError("[autotune] could not fit even the smallest probe")

    if len(measurements) < 2 or measurements[0][0] == measurements[1][0]:
        # Degenerate (both probes collapsed to the same size after OOM
        # bailouts). Fall back to the smallest probed size as recommendation.
        rec = max(min_envs, (measurements[0][0] // 256) * 256)
        print(f"[autotune] degenerate calibration; using {rec}")
        return rec

    # Linear fit: peak = fixed + per_env * N
    (n_a, p_a), (n_b, p_b) = measurements
    if n_a > n_b:
        n_a, n_b, p_a, p_b = n_b, n_a, p_b, p_a
    per_env = (p_b - p_a) / float(n_b - n_a)
    fixed = p_a - per_env * n_a

    print(
        f"[autotune] memory model: M(N) = {_fmt_bytes(fixed)} + "
        f"{_fmt_bytes(per_env)}/env  (total VRAM={_fmt_bytes(total_int)}, "
        f"budget={_fmt_bytes(budget)} @ {safety_fraction:.0%})"
    )

    if per_env <= 0:
        # Pathological (e.g., fixed overhead dominates and measurement noise
        # makes per_env look negative). Be conservative.
        print("[autotune] per_env is non-positive; falling back to largest probe")
        rec = max(min_envs, (measurements[-1][0] // 256) * 256)
        return rec

    n_max_raw = (budget - fixed) / per_env
    n_max = max(min_envs, int(min(ceiling, n_max_raw)))
    rec = max(min_envs, (n_max // 256) * 256)
    print(f"[autotune] -> num_envs={rec} (raw solve={n_max_raw:.0f}, ceiling={ceiling})")
    return rec


# --- legacy API kept for backwards compatibility (used by older callers) ---


def _try_alloc(  # noqa: PLR0913
    *,
    device: str,
    num_envs: int,
    rollout_steps: int,
    include_eval_env: bool,
    card_embed_dim: int,
    mlp_dim: int,
    torso_layers: int,
    lstm_hidden: int,
    head_layers: int,
    head_dim: int | None,
) -> bool:
    """Legacy single-net OK/fail probe. Prefer ``probe_peak``+``autotune_num_envs``."""
    peak = probe_peak(
        device=device,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        include_eval_env=include_eval_env,
        n_concurrent_nets=1,
        policy_backbone="lstm",
        card_embed_dim=card_embed_dim,
        mlp_dim=mlp_dim,
        torso_layers=torso_layers,
        lstm_hidden=lstm_hidden,
        head_layers=head_layers,
        head_dim=head_dim,
        num_epochs=1,
    )
    return peak is not None


def main() -> int:
    p = argparse.ArgumentParser(description="Probe max safe pokergpu training config")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--include-eval-env", action="store_true")
    p.add_argument("--policy-backbone", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--n-concurrent-nets", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--card-embed-dim", type=int, default=64)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--torso-layers", type=int, default=3)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--head-layers", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=0, help="0 means null/default")
    p.add_argument("--transformer-d-model", type=int, default=256)
    p.add_argument("--transformer-n-heads", type=int, default=4)
    p.add_argument("--transformer-n-layers", type=int, default=4)
    p.add_argument("--transformer-ffn-dim", type=int, default=1024)
    p.add_argument("--ceiling", type=int, default=262144)
    p.add_argument("--safety-fraction", type=float, default=0.85)
    args = p.parse_args()

    head_dim = None if args.head_dim == 0 else int(args.head_dim)
    device = args.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    t0 = time.perf_counter()
    rec = autotune_num_envs(
        device=device,
        rollout_steps=int(args.rollout_steps),
        include_eval_env=bool(args.include_eval_env),
        n_concurrent_nets=int(args.n_concurrent_nets),
        policy_backbone=str(args.policy_backbone),
        card_embed_dim=int(args.card_embed_dim),
        mlp_dim=int(args.mlp_dim),
        torso_layers=int(args.torso_layers),
        lstm_hidden=int(args.lstm_hidden),
        head_layers=int(args.head_layers),
        head_dim=head_dim,
        transformer_d_model=int(args.transformer_d_model),
        transformer_n_heads=int(args.transformer_n_heads),
        transformer_n_layers=int(args.transformer_n_layers),
        transformer_ffn_dim=int(args.transformer_ffn_dim),
        num_epochs=int(args.num_epochs),
        ceiling=int(args.ceiling),
        safety_fraction=float(args.safety_fraction),
    )
    dt = time.perf_counter() - t0
    print(f"\nrecommended_num_envs={rec}  elapsed_s={dt:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
