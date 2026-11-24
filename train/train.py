"""Hydra training entrypoint for River (self-play, league, and AlphaStar triad)."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

# Allow script execution (`uv run python train/train.py`) without shadowing the `train` package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if not sys.path or sys.path[0] != _repo_root_str:
    sys.path.insert(0, _repo_root_str)

from train.buffer import RolloutBuffer  # noqa: E402
from train.config import TrainConfig, TrainResult  # noqa: E402
from train.policy_factory import PolicyNet, load_snapshot_model  # noqa: E402
from train.policy_factory import make_policy as _make_policy  # noqa: E402
from train.policy_factory import push_transformer_token as _push_transformer_token  # noqa: E402
from train.policy_factory import stage_from_scalars as _stage_from_scalars  # noqa: E402
from train.single_runner import _run_impl  # noqa: E402
from train.triad_runner import _run_impl_triad  # noqa: E402

try:
    import pynvml

    _HAS_PYNVML = True
except ModuleNotFoundError:
    pynvml = None
    _HAS_PYNVML = False

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    _HAS_HYDRA = True
except ModuleNotFoundError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]
    ConfigStore = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
    _HAS_HYDRA = False


def run(cfg: TrainConfig) -> TrainResult:
    use_nvml = _HAS_PYNVML and cfg.device.startswith("cuda")
    if use_nvml:
        try:
            pynvml.nvmlInit()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] pynvml.nvmlInit() failed ({e!r}); disabling NVML metrics")
            use_nvml = False
    try:
        if cfg.triad_enabled:
            return _run_impl_triad(cfg)
        return _run_impl(cfg)
    finally:
        if use_nvml:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()


def _autotune_num_envs(cfg: TrainConfig) -> TrainConfig:
    """Calibrate the largest safe `num_envs` for the target device and config."""
    from train.autotune import autotune_num_envs

    n_nets = 3 if cfg.triad_enabled else 1
    print(
        f"[autotune] calibrating on {cfg.device} "
        f"(backbone={cfg.policy_backbone}, n_concurrent_nets={n_nets}, "
        f"ceiling={cfg.num_envs}) ..."
    )
    best = autotune_num_envs(
        device=cfg.device,
        rollout_steps=cfg.rollout_steps,
        include_eval_env=cfg.eval_every > 0,
        n_concurrent_nets=n_nets,
        policy_backbone=cfg.policy_backbone,
        card_embed_dim=cfg.card_embed_dim,
        mlp_dim=cfg.mlp_dim,
        torso_layers=cfg.torso_layers,
        lstm_hidden=cfg.lstm_hidden,
        head_layers=cfg.head_layers,
        head_dim=cfg.head_dim,
        transformer_d_model=cfg.transformer_d_model,
        transformer_n_heads=cfg.transformer_n_heads,
        transformer_n_layers=cfg.transformer_n_layers,
        transformer_ffn_dim=cfg.transformer_ffn_dim,
        num_epochs=cfg.num_epochs,
        ceiling=cfg.num_envs,
    )
    new_minibatch = max(256, (min(cfg.minibatch_size, best) // 256) * 256)
    print(f"[autotune] -> num_envs={best}  minibatch_size={new_minibatch}")
    return dataclasses.replace(cfg, num_envs=best, minibatch_size=new_minibatch, autotune=False)


def main() -> int:
    if not _HAS_HYDRA:  # pragma: no cover
        raise ModuleNotFoundError("hydra-core is not installed")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    ConfigStore.instance().store(name="train_config", node=TrainConfig)
    conf_dir = str((_REPO_ROOT / "conf").resolve())

    @hydra.main(version_base=None, config_path=conf_dir, config_name="train")
    def _hydra_main(cfg_dict) -> int:
        cfg_obj = TrainConfig(**OmegaConf.to_container(cfg_dict, resolve=True))  # type: ignore[arg-type]
        if cfg_obj.autotune:
            cfg_obj = _autotune_num_envs(cfg_obj)
        result = run(cfg_obj)

        payload = {
            "meta": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "torch": getattr(torch, "__version__", "unknown"),
            },
            "result": asdict(result),
            "hydra": OmegaConf.to_container(cfg_dict, resolve=True),  # type: ignore[arg-type]
        }

        Path("results.json").write_text(json.dumps(payload, indent=2) + "\n")
        (_REPO_ROOT / "train/latest_train.json").write_text(json.dumps(payload, indent=2) + "\n")

        print("Wrote: results.json")
        print(f"overall_SPS={result.overall_sps:.2f} mean_loss={result.mean_loss:.6f}")
        if result.eval:
            print(f"eval={result.eval}")
        return 0

    return _hydra_main()  # type: ignore[func-returns-value]


__all__ = [
    "PolicyNet",
    "RolloutBuffer",
    "TrainConfig",
    "TrainResult",
    "_make_policy",
    "_push_transformer_token",
    "_run_impl",
    "_run_impl_triad",
    "_stage_from_scalars",
    "load_snapshot_model",
    "main",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
