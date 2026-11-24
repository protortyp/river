"""ReBeL-style River Counterfactual Value Network (`RiverCFVNet`).

Predicts pot-normalized Nash counterfactual values `(cfv_oop[1326], cfv_ip[1326])` at
river entry as a **residual over exact check-down equity**, with:
1. Exact `SPR == 0` boundary condition (`delta_cfv == 0` when all-in).
2. Exact zero-sum projection (`r_oop · cfv_oop + r_ip · cfv_ip == 0` identically)
   so depth-limited Turn DCFR stays strictly zero-sum and converges cleanly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from gpu_poker.search import (
    COMBO_C0,
    COMBO_C1,
    NUM_COMBOS,
    RiverHandEvaluator,
    unblocked_combo_indices,
)


def river_hand_strength_pct(
    board: list[int] | np.ndarray,
    evaluator: RiverHandEvaluator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(rank_pct [1326] in [0, 1], valid_mask [1326] in {0, 1})` for a 5-card board."""
    idx = unblocked_combo_indices(board)
    pairs = np.stack([COMBO_C0[idx], COMBO_C1[idx]], axis=1)
    ranks = evaluator.eval_river_combos(pairs, board)
    # Tie-aware percentile rank in [0, 1].
    order = np.argsort(ranks, kind="stable")
    sorted_r = ranks[order]
    n = len(ranks)
    pct_sorted = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_r[j] == sorted_r[i]:
            j += 1
        mid = 0.5 * (i + j - 1) / max(1, n - 1)
        pct_sorted[i:j] = mid
        i = j
    pct = np.zeros(NUM_COMBOS, dtype=np.float32)
    valid = np.zeros(NUM_COMBOS, dtype=np.float32)
    pct[idx[order]] = pct_sorted
    valid[idx] = 1.0
    return pct, valid


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RiverCFVNet(nn.Module):
    """Residual zero-sum River Counterfactual Value Network over all 1,326 combos."""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        in_dim = NUM_COMBOS * 5 + 4
        self.in_proj = nn.Linear(in_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResBlock(self.hidden_dim, dropout=dropout) for _ in range(self.num_blocks)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_head = nn.Linear(self.hidden_dim, NUM_COMBOS * 2)
        # Zero-init final head so freshly initialized net starts at exact check-down equity.
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        *,
        r0: torch.Tensor,
        r1: torch.Tensor,
        check_cfv0: torch.Tensor,
        check_cfv1: torch.Tensor,
        rank_pct: torch.Tensor,
        valid_mask: torch.Tensor,
        spr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict exact zero-sum pot-normalized `(cfv0, cfv1)` of shape `[B, 1326]`."""
        spr_f = spr.to(dtype=r0.dtype).clamp_min(0.0)
        spr_feats = torch.stack(
            [
                torch.log1p(spr_f),
                spr_f / (1.0 + spr_f),
                1.0 / (1.0 + spr_f),
                (spr_f / 10.0).clamp_max(1.0),
            ],
            dim=-1,
        )  # [B, 4]

        x = torch.cat([r0, r1, check_cfv0, check_cfv1, rank_pct, spr_feats], dim=-1)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        delta = self.out_head(self.out_norm(h))
        d0, d1 = delta[:, :NUM_COMBOS], delta[:, NUM_COMBOS:]

        # Exact boundary condition: when SPR == 0 (all-in), river betting delta is 0.
        spr_gate = (spr_f / (spr_f + 0.1)).unsqueeze(-1)
        cfv0 = (check_cfv0 + spr_gate * d0) * valid_mask
        cfv1 = (check_cfv1 + spr_gate * d1) * valid_mask

        # Exact zero-sum projection: r0 · cfv0 + r1 · cfv1 == 0 identically.
        imbalance = (r0 * cfv0).sum(dim=-1) + (r1 * cfv1).sum(dim=-1)
        mass = (r0.sum(dim=-1) + r1.sum(dim=-1)).clamp_min(1e-6)
        shift = (imbalance / mass).unsqueeze(-1) * valid_mask
        cfv0 = cfv0 - shift
        cfv1 = cfv1 - shift
        return cfv0, cfv1


def save_cfv_checkpoint(path: str | Path, net: RiverCFVNet, meta: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_dim": net.hidden_dim,
            "num_blocks": net.num_blocks,
            "model": net.state_dict(),
            "meta": meta or {},
        },
        p,
    )
    return p


def load_cfv_checkpoint(path: str | Path, device: str = "cpu") -> RiverCFVNet:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    net = RiverCFVNet(
        hidden_dim=int(payload.get("hidden_dim", 1024)),
        num_blocks=int(payload.get("num_blocks", 4)),
    )
    net.load_state_dict(payload["model"])
    return net.to(torch.device(device)).eval()
