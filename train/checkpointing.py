from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CheckpointState:
    env_steps: int
    opt_steps: int
    update: int


def _to_cpu_byte_tensor(x) -> torch.ByteTensor:
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.dtype != torch.uint8:
            t = t.to(dtype=torch.uint8)
        return t.to(device=torch.device("cpu"), non_blocking=False).contiguous()  # type: ignore[return-value]
    if isinstance(x, bytes | bytearray):
        return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.uint8)
    raise TypeError(f"Unsupported RNG state type: {type(x)}")


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    # PyTorch doesn't automatically move optimizer state tensors to the model device.
    for st in optimizer.state.values():
        if not isinstance(st, dict):
            continue
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device=device)


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    state: CheckpointState,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state": {
            "env_steps": int(state.env_steps),
            "opt_steps": int(state.opt_steps),
            "update": int(state.update),
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(payload, p)
    return p


def load_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> CheckpointState:
    # Always load checkpoints onto CPU first:
    # - torch RNG state must be a CPU ByteTensor for `torch.set_rng_state()`
    # - optimizer state can be moved to the target device after loading
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)

    rng = payload.get("rng") or {}
    if "torch" in rng and rng["torch"] is not None:
        torch.set_rng_state(_to_cpu_byte_tensor(rng["torch"]))
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        cuda_states = rng["cuda"]
        if isinstance(cuda_states, list | tuple):
            torch.cuda.set_rng_state_all([_to_cpu_byte_tensor(s) for s in cuda_states])
        else:
            torch.cuda.set_rng_state_all([_to_cpu_byte_tensor(cuda_states)])

    s = payload.get("state") or {}
    return CheckpointState(
        env_steps=int(s.get("env_steps", 0)),
        opt_steps=int(s.get("opt_steps", 0)),
        update=int(s.get("update", 0)),
    )
