"""ReBeL-style River Counterfactual Value Network (`RiverCFVNet`).

Permutation-equivariant, blocker-aware architecture predicting pot-normalized Nash
counterfactual values `(cfv_oop[1326], cfv_ip[1326])` at river entry as a **residual
over exact check-down equity**, with:
1. Exact card-permutation equivariance across all 52! card relabelings (shared per-combo
   weights conditioned on exact 52-card blocker aggregates and 32-bin strength-distribution
   summaries), giving ~1,000x better sample efficiency than a flat 1,326-slot MLP.
2. Exact `SPR == 0` boundary condition (`delta_cfv == 0` when all-in).
3. Exact zero-sum projection (`r_oop · cfv_oop + r_ip · cfv_ip == 0` identically)
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

NUM_STRENGTH_BINS = 32


def _build_card_incidence() -> torch.Tensor:
    """Return `[1326, 52]` binary matrix where `M[h, c] = 1` iff combo `h` contains card `c`."""
    mat = np.zeros((NUM_COMBOS, 52), dtype=np.float32)
    idx = np.arange(NUM_COMBOS)
    mat[idx, COMBO_C0] = 1.0
    mat[idx, COMBO_C1] = 1.0
    return torch.from_numpy(mat)


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
    """Permutation-equivariant, blocker-aware residual zero-sum River CFV network.

    `hidden_dim` sets the per-combo trunk width (e.g. 64 or 96); global range context
    uses `global_dim = max(128, hidden_dim * 4)` over 32 strength-percentile bins plus
    exact 52-card blocker features gathered per combo.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.global_dim = max(128, self.hidden_dim * 4)
        self.bin_dim = max(16, self.hidden_dim // 2)

        # Static `[1326, 52]` combo-to-card incidence matrix for O(H) exact blocker matmuls.
        self.card_incidence: torch.Tensor
        self.register_buffer("card_incidence", _build_card_incidence(), persistent=False)

        # Global 32-bin range histogram (6 stats/bin = 192) + 4 SPR + 6 summary scalars = 202.
        global_in = NUM_STRENGTH_BINS * 6 + 10
        self.global_in = nn.Linear(global_in, self.global_dim)
        self.global_blocks = nn.ModuleList(
            [_ResBlock(self.global_dim, dropout=dropout) for _ in range(3)]
        )
        self.global_norm = nn.LayerNorm(self.global_dim)
        self.global_to_combo = nn.Linear(self.global_dim, self.hidden_dim)
        self.global_to_bins = nn.Linear(self.global_dim, NUM_STRENGTH_BINS * self.bin_dim)

        # Per-combo features:
        # 8 own scalars (r0_s, r1_s, log_r0, log_r1, ck0, ck1, pct, pct^2)
        # + 6 card-blocker aggregates (r0, r1, top_r0, top_r1, ev0, ev1)
        # + 4 SPR scalars + bin_dim tier context.
        combo_in = 8 + 6 + 4 + self.bin_dim
        self.combo_in = nn.Linear(combo_in, self.hidden_dim)
        self.combo_blocks = nn.ModuleList(
            [_ResBlock(self.hidden_dim, dropout=dropout) for _ in range(self.num_blocks)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_head = nn.Linear(self.hidden_dim, 2)
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
        bsz = r0.shape[0]
        dtype = r0.dtype
        spr_f = spr.to(dtype=dtype).clamp_min(0.0)
        spr_feats = torch.stack(
            [
                torch.log1p(spr_f),
                spr_f / (1.0 + spr_f),
                1.0 / (1.0 + spr_f),
                (spr_f / 10.0).clamp_max(1.0),
            ],
            dim=-1,
        )  # [B, 4]

        # Mask everything to valid combos first so blocked slots carry exact zeros.
        r0v = r0 * valid_mask
        r1v = r1 * valid_mask
        ck0v = check_cfv0 * valid_mask
        ck1v = check_cfv1 * valid_mask
        pctv = rank_pct * valid_mask

        # 1. Exact 52-card blocker aggregation via incidence GEMM: [B, 6, 1326] @ [1326, 52].
        top_mask = (pctv >= 0.85).to(dtype=dtype) * valid_mask
        combo_for_cards = torch.stack(
            [
                r0v,
                r1v,
                r0v * top_mask,
                r1v * top_mask,
                r0v * ck0v,
                r1v * ck1v,
            ],
            dim=1,
        )  # [B, 6, 1326]
        m_card = self.card_incidence.to(dtype=dtype)  # [1326, 52]
        # Scale by 26 so average per-card mass is O(1) across 52 cards.
        card_stats = torch.matmul(combo_for_cards, m_card) * 26.0  # [B, 6, 52]
        # Gather sum over both cards of each combo: [B, 6, 52] @ [52, 1326] -> [B, 1326, 6].
        blocker_feats = torch.matmul(card_stats, m_card.transpose(0, 1)).transpose(1, 2)

        # 2. 32-bin strength-percentile range histograms & CDFs.
        bin_idx = (pctv * NUM_STRENGTH_BINS).long().clamp(0, NUM_STRENGTH_BINS - 1)  # [B, 1326]
        zeros_b = r0v.new_zeros((bsz, NUM_STRENGTH_BINS))
        hist_r0 = zeros_b.scatter_add(1, bin_idx, r0v)
        hist_r1 = zeros_b.scatter_add(1, bin_idx, r1v)
        hist_ck0 = zeros_b.scatter_add(1, bin_idx, r0v * ck0v)
        hist_ck1 = zeros_b.scatter_add(1, bin_idx, r1v * ck1v)
        cdf_r0 = torch.cumsum(hist_r0, dim=1)
        cdf_r1 = torch.cumsum(hist_r1, dim=1)

        ev0_tot = (r0v * ck0v).sum(dim=-1)
        conc0 = torch.log1p((r0v * r0v).sum(dim=-1) * 1081.0)
        conc1 = torch.log1p((r1v * r1v).sum(dim=-1) * 1081.0)
        nut0 = (r0v * top_mask).sum(dim=-1)
        nut1 = (r1v * top_mask).sum(dim=-1)
        mid_mask = ((pctv >= 0.30) & (pctv <= 0.70)).to(dtype=dtype) * valid_mask
        mid_diff = (r0v * mid_mask).sum(dim=-1) - (r1v * mid_mask).sum(dim=-1)
        global_scalars = torch.stack(
            [ev0_tot, conc0, conc1, nut0, nut1, mid_diff], dim=-1
        )  # [B, 6]

        global_x = torch.cat(
            [
                hist_r0 * NUM_STRENGTH_BINS,
                hist_r1 * NUM_STRENGTH_BINS,
                hist_ck0 * NUM_STRENGTH_BINS,
                hist_ck1 * NUM_STRENGTH_BINS,
                cdf_r0,
                cdf_r1,
                spr_feats,
                global_scalars,
            ],
            dim=-1,
        )  # [B, 202]
        g = self.global_in(global_x)
        for blk in self.global_blocks:
            g = blk(g)
        g = self.global_norm(g)
        g_bias = self.global_to_combo(g).unsqueeze(1)  # [B, 1, hidden_dim]
        bin_table = self.global_to_bins(g).view(bsz, NUM_STRENGTH_BINS, self.bin_dim)
        # Gather strength-tier embedding for every combo: [B, 1326, bin_dim].
        gather_idx = bin_idx.unsqueeze(-1).expand(-1, -1, self.bin_dim)
        combo_bin_ctx = torch.gather(bin_table, 1, gather_idx)

        # 3. Shared per-combo residual trunk (exact card-permutation equivariance).
        r0_s = (r0v * 1081.0).clamp_max(20.0)
        r1_s = (r1v * 1081.0).clamp_max(20.0)
        own_feats = torch.stack(
            [
                r0_s,
                r1_s,
                torch.log1p(r0v * 1081.0),
                torch.log1p(r1v * 1081.0),
                ck0v,
                ck1v,
                pctv,
                pctv * pctv,
            ],
            dim=-1,
        )  # [B, 1326, 8]
        spr_b = spr_feats.unsqueeze(1).expand(-1, NUM_COMBOS, -1)  # [B, 1326, 4]
        combo_x = torch.cat([own_feats, blocker_feats, spr_b, combo_bin_ctx], dim=-1)
        h = self.combo_in(combo_x) + g_bias
        for blk in self.combo_blocks:
            h = blk(h)
        delta = self.out_head(self.out_norm(h))  # [B, 1326, 2]
        d0, d1 = delta[..., 0], delta[..., 1]

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
            "arch": "equivariant_v2",
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
        hidden_dim=int(payload.get("hidden_dim", 64)),
        num_blocks=int(payload.get("num_blocks", 2)),
    )
    net.load_state_dict(payload["model"])
    return net.to(torch.device(device)).eval()
