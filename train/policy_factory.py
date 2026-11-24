"""Policy construction, snapshot loading, and transformer history helpers."""

from __future__ import annotations

import torch

from gpu_poker import constants as c
from gpu_poker.policy import PokerPolicyNet
from gpu_poker.policy_transformer import (
    PokerTransformerPolicyNet,
    push_action_token_packed,
)
from train.config import TrainConfig

PolicyNet = PokerPolicyNet | PokerTransformerPolicyNet


def make_policy(cfg: TrainConfig, scalar_dim: int, device: torch.device) -> PolicyNet:
    """Construct the configured policy backbone (LSTM or Transformer)."""
    backbone = (cfg.policy_backbone or "lstm").lower()
    if backbone == "lstm":
        return PokerPolicyNet(
            scalar_dim=scalar_dim,
            card_embed_dim=cfg.card_embed_dim,
            mlp_dim=cfg.mlp_dim,
            torso_layers=cfg.torso_layers,
            lstm_hidden=cfg.lstm_hidden,
            head_layers=cfg.head_layers,
            head_dim=cfg.head_dim,
        ).to(device=device)
    if backbone == "transformer":
        return PokerTransformerPolicyNet(
            scalar_dim=scalar_dim,
            card_embed_dim=cfg.card_embed_dim,
            d_model=cfg.transformer_d_model,
            n_heads=cfg.transformer_n_heads,
            n_layers=cfg.transformer_n_layers,
            ffn_dim=cfg.transformer_ffn_dim,
        ).to(device=device)
    raise ValueError(f"Unknown policy_backbone: {cfg.policy_backbone!r}")


def load_snapshot_model(
    path: str, device: torch.device, cfg: TrainConfig, scalar_dim: int
) -> PolicyNet:
    """Load a snapshot model, stripping any torch.compile `_orig_mod.` prefix."""
    ckpt = torch.load(path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}

    net = make_policy(cfg, scalar_dim, device)
    try:
        net.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load snapshot at {path}; the architecture likely "
            f"changed since this snapshot was saved. Clear stale snapshots "
            f"from the league pool and resume."
        ) from e
    net.eval()
    return net


def push_transformer_token(
    h0: torch.Tensor,
    h1: torch.Tensor,
    *,
    player_id: torch.Tensor,
    action_type: torch.Tensor,
    raise_bucket: torch.Tensor,
    stage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one public-action token to both seats' packed transformer state."""
    bucket = torch.where(
        action_type.eq(c.ACTION_RAISE),
        raise_bucket.to(dtype=torch.int32),
        torch.zeros_like(raise_bucket, dtype=torch.int32),
    )
    pid_i32 = player_id.to(dtype=torch.int32)
    h0_new = push_action_token_packed(
        h0, player_id=pid_i32, action_type=action_type, raise_bucket=bucket, stage=stage
    )
    h1_new = push_action_token_packed(
        h1, player_id=pid_i32, action_type=action_type, raise_bucket=bucket, stage=stage
    )
    return h0_new, h1_new


def stage_from_scalars(scalars: torch.Tensor) -> torch.Tensor:
    """Recover the integer stage index (0..5) from normalized `scalars[..., 7]`."""
    return (scalars[..., 7] * 5.0).round().to(dtype=torch.int32)
