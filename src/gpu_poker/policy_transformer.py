"""Transformer backbone for the HUNL policy.

Drop-in alternative to PokerPolicyNet (LSTM) with:
- Suit-isomorphism + hole-card-order canonicalization (48x symmetry reduction),
  applied inside the model so PyTorch, ONNX, browser, and Slumbot inference
  share identical canonical inputs with zero external encoder changes.
- Factored card-token embeddings (card + rank + suit + slot) attending jointly
  with the current scalar state and up to MAX_TOKENS=32 public action tokens.
- Optional training-only omniscient critic (CTDE): when `opp_cards` is passed
  during training, the value head conditions on both hands to cut advantage
  variance while the actor remains strictly decentralized.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as fn

from gpu_poker import constants as constants
from gpu_poker.policy import NUM_RAISE_BUCKETS

MAX_TOKENS: int = 32
TOKEN_FIELDS: int = 4
# Packed-state layout for backbone-agnostic RolloutBuffer storage:
#   [:MAX_TOKENS*TOKEN_FIELDS] = flattened token table (1-indexed small ints as fp32)
#   [MAX_TOKENS*TOKEN_FIELDS]  = current sequence length
STATE_WIDTH: int = MAX_TOKENS * TOKEN_FIELDS + 1

NUM_CARD_SLOTS: int = 7

PAD_IDX: int = 0
PLAYER_VOCAB = 2 + 1
ACTION_VOCAB = constants.NUM_ACTIONS + 1
BUCKET_VOCAB = NUM_RAISE_BUCKETS + 1
STAGE_VOCAB = constants.NUM_STAGES + 1


@dataclass(frozen=True)
class PolicyOutput:
    action_type: torch.Tensor
    raise_bucket: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    h: torch.Tensor
    c: torch.Tensor
    decentralized_value: torch.Tensor | None = None
    opp_card_loss: torch.Tensor | None = None


def canonicalize_cards_with_map(cards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize hole-card order and relabel suits by first appearance on `[B, 7]`.

    1. Sort hole cards (`cards[:, :2]`) descending by rank (tie-break by card id).
    2. Relabel suits `0..3` in order of first appearance across `[hole0, hole1, board0..4]`.
    3. Preserve unrevealed cards (`-1`).

    Returns `(canon_cards [B, 7] int64, canon_suit_map [B, 4] int64)`.
    Pure branchless tensor ops for ONNX opset-17 export.
    """
    c = cards.to(dtype=torch.int64)
    h0 = c[:, 0]
    h1 = c[:, 1]
    neg_one = torch.full_like(h0, -1)
    r0 = torch.where(h0 >= 0, h0 % 13, neg_one)
    r1 = torch.where(h1 >= 0, h1 % 13, neg_one)
    swap = (r1 > r0) | ((r1 == r0) & (h1 > h0))
    new_h0 = torch.where(swap, h1, h0)
    new_h1 = torch.where(swap, h0, h1)
    c = torch.cat([new_h0.unsqueeze(1), new_h1.unsqueeze(1), c[:, 2:]], dim=1)

    valid = c >= 0
    ranks = torch.where(valid, c % 13, torch.zeros_like(c))
    suits = torch.where(valid, c // 13, torch.full_like(c, -1))

    pos = torch.arange(NUM_CARD_SLOTS, device=c.device, dtype=torch.int64).view(
        1, NUM_CARD_SLOTS, 1
    )
    suit_ids = torch.arange(4, device=c.device, dtype=torch.int64).view(1, 1, 4)
    matches = valid.unsqueeze(2) & (suits.unsqueeze(2) == suit_ids)
    # Distinct sentinels 7+s ensure unseen suits get a bijective tie-free ranking.
    sentinel = (NUM_CARD_SLOTS + suit_ids).expand(c.shape[0], NUM_CARD_SLOTS, 4)
    first_pos = torch.where(matches, pos.expand_as(matches), sentinel).min(dim=1).values

    canon_map = (first_pos.unsqueeze(2) > first_pos.unsqueeze(1)).sum(dim=2)
    safe_suits = suits.clamp(min=0, max=3)
    canon_suits = torch.gather(canon_map, dim=1, index=safe_suits)
    canon_cards = canon_suits * 13 + ranks
    return torch.where(valid, canon_cards, torch.full_like(c, -1)), canon_map


def canonicalize_cards(cards: torch.Tensor) -> torch.Tensor:
    """Canonicalize hole-card order and suits on `[B, 7]` cards in `[-1, 51]`."""
    canon_cards, _ = canonicalize_cards_with_map(cards)
    return canon_cards


def apply_suit_map(cards: torch.Tensor, canon_map: torch.Tensor) -> torch.Tensor:
    """Apply a hero-derived `[B, 4]` suit map to arbitrary `[B, K]` cards (e.g. opp hole cards)."""
    c = cards.to(dtype=torch.int64)
    valid = c >= 0
    ranks = torch.where(valid, c % 13, torch.zeros_like(c))
    suits = torch.where(valid, c // 13, torch.zeros_like(c)).clamp(min=0, max=3)
    canon_suits = torch.gather(canon_map, dim=1, index=suits)
    canon_cards = canon_suits * 13 + ranks
    return torch.where(valid, canon_cards, torch.full_like(c, -1))


def _sinusoidal_pos_enc(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal positional encodings, [seq_len, d_model] fp32."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d_model // 2, device=device, dtype=torch.float32)
    div = torch.exp(-(i * 2.0) * (torch.log(torch.tensor(10000.0, device=device)) / d_model))
    angles = pos * div
    pe = torch.zeros((seq_len, d_model), device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class _SDPAEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder block using native Flash/cuDNN SDPA in PyTorch
    and explicit matmul-softmax during ONNX opset-17 export.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.linear1 = nn.Linear(self.d_model, int(ffn_dim))
        self.linear2 = nn.Linear(int(ffn_dim), self.d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        b, s, d = h.shape
        qkv = self.in_proj(h).reshape(b, s, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        if torch.onnx.is_in_onnx_export():
            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
            bias = torch.where(
                key_padding_mask,
                torch.full_like(key_padding_mask, -1e9, dtype=scores.dtype),
                torch.zeros_like(key_padding_mask, dtype=scores.dtype),
            ).view(b, 1, 1, s)
            weights = torch.softmax(scores + bias, dim=-1)
            attn_out = torch.matmul(weights, v)
        else:
            # PyTorch SDPA bool mask: True = attend, False = masked out.
            keep_mask = (~key_padding_mask).view(b, 1, 1, s)
            attn_out = fn.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=keep_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        attn_out = attn_out.transpose(1, 2).reshape(b, s, d)
        x = x + self.out_proj(attn_out)
        x = x + self.linear2(fn.gelu(self.linear1(self.norm2(x))))
        return x


class _SDPAEncoderStack(nn.Module):
    """Stack of `_SDPAEncoderLayer` blocks kept as a dedicated submodule for `torch.compile`."""

    def __init__(self, *, n_layers: int, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _SDPAEncoderLayer(
                    d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


class PokerTransformerPolicyNet(nn.Module):
    """Encoder-only Transformer policy with suit-canonical card tokens and CTDE critic."""

    pos_enc: torch.Tensor

    def __init__(
        self,
        *,
        scalar_dim: int,
        card_embed_dim: int = 32,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.card_embed_dim = int(card_embed_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ffn_dim = int(ffn_dim)
        self.max_tokens = int(max_tokens)

        # ---- Public-action history token embeddings ----
        self.player_emb = nn.Embedding(PLAYER_VOCAB, self.d_model, padding_idx=PAD_IDX)
        self.action_emb = nn.Embedding(ACTION_VOCAB, self.d_model, padding_idx=PAD_IDX)
        self.bucket_emb = nn.Embedding(BUCKET_VOCAB, self.d_model, padding_idx=PAD_IDX)
        self.stage_emb = nn.Embedding(STAGE_VOCAB, self.d_model, padding_idx=PAD_IDX)

        # ---- Factored card-token embeddings (card + rank + suit + slot) ----
        # 0 is PAD (-1 card); real cards map via +1.
        self.card_embed = nn.Embedding(53, self.d_model, padding_idx=0)
        self.rank_embed = nn.Embedding(14, self.d_model, padding_idx=0)
        self.suit_embed = nn.Embedding(5, self.d_model, padding_idx=0)
        self.slot_embed = nn.Embedding(NUM_CARD_SLOTS, self.d_model)

        # ---- Current scalar-state query token ----
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.cur_type_emb = nn.Parameter(torch.zeros(self.d_model))

        # ---- SDPA Transformer encoder ----
        self.blocks: nn.Module = _SDPAEncoderStack(
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_dim,
            dropout=dropout,
        )

        # Sequence: [1 scalar query token] + [7 card tokens] + [max_tokens history tokens].
        pe = _sinusoidal_pos_enc(
            1 + NUM_CARD_SLOTS + self.max_tokens, self.d_model, torch.device("cpu")
        )
        self.register_buffer("pos_enc", pe, persistent=False)

        # ---- Actor & decentralized value heads ----
        self.action_head = nn.Linear(self.d_model, constants.NUM_ACTIONS)
        self.value_head = nn.Linear(self.d_model, 1)
        self.raise_head = nn.Linear(self.d_model, NUM_RAISE_BUCKETS)

        # ---- Training-only omniscient critic (CTDE) ----
        # Opponent hole cards cross-attend to [latent, hero cards, board cards] before value MLP
        # so the critic directly evaluates made hands, draws, and blockers on the current board.
        self.omni_norm_q = nn.LayerNorm(self.d_model)
        self.omni_norm_kv = nn.LayerNorm(self.d_model)
        self.omni_cross_attn = nn.MultiheadAttention(
            self.d_model, self.n_heads, dropout=dropout, batch_first=True
        )
        self.omniscient_value_head = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        # Training-only auxiliary head predicting canonicalized opponent hole cards from the
        # decentralized actor latent so the actor learns implicit Bayesian range reading.
        self.opp_card_head = nn.Linear(self.d_model, constants.NUM_CARDS)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Remap legacy nn.TransformerEncoder keys (`encoder.layers.*`) and fill newly added
        # training-only CTDE/auxiliary head keys so older checkpoints load cleanly.
        legacy_prefix = f"{prefix}encoder.layers."
        remapped = False
        for k in list(state_dict.keys()):
            if k.startswith(legacy_prefix):
                remapped = True
                rest = k[len(legacy_prefix) :]
                if ".self_attn.in_proj_weight" in rest:
                    new_rest = rest.replace(".self_attn.in_proj_weight", ".in_proj.weight")
                elif ".self_attn.in_proj_bias" in rest:
                    new_rest = rest.replace(".self_attn.in_proj_bias", ".in_proj.bias")
                elif ".self_attn.out_proj." in rest:
                    new_rest = rest.replace(".self_attn.out_proj.", ".out_proj.")
                else:
                    new_rest = rest
                state_dict[f"{prefix}blocks.layers.{new_rest}"] = state_dict.pop(k)

        own = self.state_dict()
        if remapped:
            for suffix in (
                "omni_norm_q.weight",
                "omni_norm_q.bias",
                "omni_norm_kv.weight",
                "omni_norm_kv.bias",
                "omni_cross_attn.in_proj_weight",
                "omni_cross_attn.in_proj_bias",
                "omni_cross_attn.out_proj.weight",
                "omni_cross_attn.out_proj.bias",
            ):
                full_k = f"{prefix}{suffix}"
                if full_k not in state_dict and suffix in own:
                    state_dict[full_k] = own[suffix].clone()
        for suffix in ("opp_card_head.weight", "opp_card_head.bias"):
            full_k = f"{prefix}{suffix}"
            if full_k not in state_dict and suffix in own:
                state_dict[full_k] = own[suffix].clone()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @property
    def lstm_hidden(self) -> int:
        return STATE_WIDTH

    @staticmethod
    def init_state(
        batch: int, hidden: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros((1, batch, STATE_WIDTH), device=device, dtype=torch.float32)
        c = torch.zeros((1, batch, STATE_WIDTH), device=device, dtype=torch.float32)
        return h, c

    @staticmethod
    def init_raw_state(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.zeros((batch, MAX_TOKENS, TOKEN_FIELDS), device=device, dtype=torch.int32)
        lengths = torch.zeros((batch,), device=device, dtype=torch.int32)
        return tokens, lengths

    @staticmethod
    def unpack_state(h_packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = h_packed[0] if h_packed.ndim == 3 else h_packed
        batch_size = x.shape[0]
        tok_flat = x[:, : MAX_TOKENS * TOKEN_FIELDS]
        lengths = x[:, -1].round().to(dtype=torch.int32)
        tokens = (
            tok_flat.round().to(dtype=torch.int32).reshape(batch_size, MAX_TOKENS, TOKEN_FIELDS)
        )
        return tokens, lengths

    @staticmethod
    def pack_state(tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        tok_flat = tokens.to(dtype=torch.float32).reshape(batch_size, MAX_TOKENS * TOKEN_FIELDS)
        len_col = lengths.to(dtype=torch.float32).reshape(batch_size, 1)
        packed = torch.cat([tok_flat, len_col], dim=-1)
        return packed.unsqueeze(0)

    def _embed_history(self, tokens: torch.Tensor) -> torch.Tensor:
        t_i64 = tokens.to(dtype=torch.int64).clamp_min(0)
        return (
            self.player_emb(t_i64[..., 0])
            + self.action_emb(t_i64[..., 1])
            + self.bucket_emb(t_i64[..., 2])
            + self.stage_emb(t_i64[..., 3])
        )

    def _embed_cards(self, canon_cards: torch.Tensor) -> torch.Tensor:
        """Factored embedding of `[B, K]` canonical cards in `[-1, 51]` -> `[B, K, D]`."""
        valid = canon_cards >= 0
        card_idx = torch.where(valid, canon_cards + 1, torch.zeros_like(canon_cards)).clamp(0, 52)
        rank_idx = torch.where(valid, (canon_cards % 13) + 1, torch.zeros_like(canon_cards)).clamp(
            0, 13
        )
        suit_idx = torch.where(valid, (canon_cards // 13) + 1, torch.zeros_like(canon_cards)).clamp(
            0, 4
        )
        return self.card_embed(card_idx) + self.rank_embed(rank_idx) + self.suit_embed(suit_idx)

    def _encode_with_cards(
        self,
        *,
        cards: torch.Tensor,
        scalars: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode obs + history -> `(latent [B,D], card_out [B,7,D], card_pad [B,7], canon_map)`."""
        batch_size = cards.shape[0]
        device = cards.device

        canon_cards, canon_map = canonicalize_cards_with_map(cards)
        cur_tok = (self.scalar_mlp(scalars) + self.cur_type_emb).unsqueeze(1)  # [B, 1, D]

        slot_ids = torch.arange(NUM_CARD_SLOTS, device=device, dtype=torch.int64).unsqueeze(0)
        card_toks = self._embed_cards(canon_cards) + self.slot_embed(slot_ids)  # [B, 7, D]

        # Trim trailing all-PAD history slots outside ONNX export: because masked keys have
        # zero attention weight and only position 0 is read, trimming is bit-for-bit lossless
        # while cutting attention/FFN FLOPs and activation memory by ~2-3x.
        if torch.onnx.is_in_onnx_export():
            active_hist = self.max_tokens
        else:
            max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
            active_hist = max(1, min(self.max_tokens, max_len))

        hist_toks = self._embed_history(tokens[:, :active_hist, :])  # [B, active_hist, D]
        seq_len = 1 + NUM_CARD_SLOTS + active_hist

        x = torch.cat([cur_tok, card_toks, hist_toks], dim=1)
        x = x + self.pos_enc[:seq_len].to(x.dtype).unsqueeze(0)

        # Position 0 (query) is always valid; unrevealed board cards (-1) and empty
        # history slots are masked out.
        cur_pad = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        card_pad = canon_cards < 0
        pos = torch.arange(active_hist, device=device).unsqueeze(0)
        hist_pad = pos >= lengths.to(dtype=torch.int64).unsqueeze(1)
        key_padding_mask = torch.cat([cur_pad, card_pad, hist_pad], dim=1)

        y = self.blocks(x, key_padding_mask)
        return y[:, 0, :], y[:, 1 : 1 + NUM_CARD_SLOTS, :], card_pad, canon_map

    def encode_state(
        self,
        *,
        cards: torch.Tensor,
        scalars: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observation + action history into `[B, D]` latent and `[B, 4]` suit map."""
        latent, _, _, canon_map = self._encode_with_cards(
            cards=cards, scalars=scalars, tokens=tokens, lengths=lengths
        )
        return latent, canon_map

    def _masked_action_logits(
        self, logits: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = action_mask.to(dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[-1] != constants.NUM_ACTIONS:
            raise ValueError(
                f"Expected action_mask [batch_size, {constants.NUM_ACTIONS}], "
                f"got {tuple(mask.shape)}"
            )
        mask_sum = mask.sum(dim=-1)
        mask = mask.clone()
        mask[:, constants.ACTION_FOLD] = mask[:, constants.ACTION_FOLD] | (mask_sum == 0)
        neg_inf = float(torch.finfo(logits.dtype).min)
        return torch.where(mask, logits, neg_inf)

    def _compute_values(
        self,
        latent: torch.Tensor,
        card_out: torch.Tensor,
        card_pad: torch.Tensor,
        canon_map: torch.Tensor,
        opp_cards: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dec_val = self.value_head(latent).squeeze(-1)
        if opp_cards is None:
            return dec_val, None
        opp_canon = apply_suit_map(opp_cards, canon_map)
        # Canonicalize opponent hole-card order (descending rank) for symmetry.
        h0 = opp_canon[:, 0]
        h1 = opp_canon[:, 1]
        neg_one = torch.full_like(h0, -1)
        r0 = torch.where(h0 >= 0, h0 % 13, neg_one)
        r1 = torch.where(h1 >= 0, h1 % 13, neg_one)
        swap = (r1 > r0) | ((r1 == r0) & (h1 > h0))
        opp_sorted = torch.stack(
            [torch.where(swap, h1, h0), torch.where(swap, h0, h1)],
            dim=1,
        )
        opp_toks = self._embed_cards(opp_sorted)  # [B, 2, D]
        kv = torch.cat([latent.unsqueeze(1), card_out], dim=1)  # [B, 8, D]
        kv_pad = torch.cat(
            [torch.zeros((latent.shape[0], 1), device=latent.device, dtype=torch.bool), card_pad],
            dim=1,
        )
        attn_ctx, _ = self.omni_cross_attn(
            self.omni_norm_q(opp_toks),
            self.omni_norm_kv(kv),
            self.omni_norm_kv(kv),
            key_padding_mask=kv_pad,
            need_weights=False,
        )
        opp_feat = (opp_toks + attn_ctx).reshape(latent.shape[0], -1)  # [B, 2*D]
        omni_in = torch.cat([latent, opp_feat], dim=-1)
        omni_val = self.omniscient_value_head(omni_in).squeeze(-1)
        return omni_val, dec_val

    def _compute_opp_card_loss(
        self,
        latent: torch.Tensor,
        canon_map: torch.Tensor,
        opp_cards: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if opp_cards is None:
            return None
        opp_canon = apply_suit_map(opp_cards, canon_map)
        valid_rows = (opp_canon[:, 0] >= 0) & (opp_canon[:, 1] >= 0)
        if not valid_rows.any():
            return None
        logits = self.opp_card_head(latent[valid_rows]).to(dtype=torch.float32)
        c0 = opp_canon[valid_rows, 0].clamp(min=0, max=constants.NUM_CARDS - 1)
        c1 = opp_canon[valid_rows, 1].clamp(min=0, max=constants.NUM_CARDS - 1)
        target = torch.zeros_like(logits)
        row_idx = torch.arange(logits.shape[0], device=logits.device)
        target[row_idx, c0] = 1.0
        target[row_idx, c1] = 1.0
        return fn.binary_cross_entropy_with_logits(logits, target)

    def _forward_latent(
        self,
        *,
        cards: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        terminated: torch.Tensor | None,
        opp_cards: torch.Tensor | None = None,
    ) -> tuple[
        Categorical,
        Categorical,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        if terminated is not None:
            reset = terminated.to(dtype=torch.bool)
            if reset.any():
                mask = reset.view(-1, 1, 1).expand_as(tokens)
                tokens = torch.where(mask, torch.zeros_like(tokens), tokens)
                lengths = torch.where(reset, torch.zeros_like(lengths), lengths)

        latent, card_out, card_pad, canon_map = self._encode_with_cards(
            cards=cards, scalars=scalars, tokens=tokens, lengths=lengths
        )

        masked_logits = self._masked_action_logits(self.action_head(latent), action_mask)
        act_dist = Categorical(logits=masked_logits)

        raise_logits = self.raise_head(latent)
        raise_dist = Categorical(logits=raise_logits.to(dtype=torch.float32))

        value, dec_value = self._compute_values(latent, card_out, card_pad, canon_map, opp_cards)
        return act_dist, raise_dist, value, dec_value, tokens, lengths

    def _coerce_input_state(
        self, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if h.dtype.is_floating_point and h.ndim == 3 and h.shape[-1] == STATE_WIDTH:
            tokens, lengths = self.unpack_state(h)
            return tokens, lengths, True
        tokens = h
        lengths = c.reshape(-1).to(dtype=torch.int32) if c.ndim > 1 else c
        return tokens, lengths, False

    def forward_step(
        self,
        *,
        cards: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        terminated: torch.Tensor | None = None,
        deterministic: bool = False,
        opp_cards: torch.Tensor | None = None,
    ) -> PolicyOutput:
        tokens, lengths, is_packed = self._coerce_input_state(h, c)

        act_dist, raise_dist, value, dec_value, tokens2, lengths2 = self._forward_latent(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            tokens=tokens,
            lengths=lengths,
            terminated=terminated,
            opp_cards=opp_cards,
        )

        if deterministic:
            action_type = act_dist.logits.argmax(dim=-1).to(dtype=torch.int64)
            raise_bucket = raise_dist.logits.argmax(dim=-1).to(dtype=torch.int64)
        else:
            action_type = act_dist.sample()
            raise_bucket = raise_dist.sample()

        action_logprob = act_dist.log_prob(action_type)
        raise_logprob = raise_dist.log_prob(raise_bucket)
        is_raise = action_type.eq(constants.ACTION_RAISE)
        logprob = action_logprob + torch.where(
            is_raise, raise_logprob, torch.zeros_like(raise_logprob)
        )
        entropy = act_dist.entropy() + torch.where(
            is_raise, raise_dist.entropy(), torch.zeros_like(raise_logprob)
        )

        if is_packed:
            h_out = self.pack_state(tokens2, lengths2)
            c_out = c
        else:
            h_out = tokens2
            c_out = lengths2

        return PolicyOutput(
            action_type=action_type,
            raise_bucket=raise_bucket,
            logprob=logprob,
            entropy=entropy,
            value=value,
            h=h_out,
            c=c_out,
            decentralized_value=dec_value,
        )

    def evaluate_step(
        self,
        *,
        cards: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        action_type: torch.Tensor,
        raise_bucket: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        terminated: torch.Tensor | None = None,
        opp_cards: torch.Tensor | None = None,
    ) -> PolicyOutput:
        tokens, lengths, is_packed = self._coerce_input_state(h, c)

        if terminated is not None:
            reset = terminated.to(dtype=torch.bool)
            if reset.any():
                mask = reset.view(-1, 1, 1).expand_as(tokens)
                tokens = torch.where(mask, torch.zeros_like(tokens), tokens)
                lengths = torch.where(reset, torch.zeros_like(lengths), lengths)

        latent, card_out, card_pad, canon_map = self._encode_with_cards(
            cards=cards, scalars=scalars, tokens=tokens, lengths=lengths
        )
        masked_logits = self._masked_action_logits(self.action_head(latent), action_mask)
        act_dist = Categorical(logits=masked_logits)
        raise_dist = Categorical(logits=self.raise_head(latent).to(dtype=torch.float32))
        value, dec_value = self._compute_values(latent, card_out, card_pad, canon_map, opp_cards)
        opp_card_loss = self._compute_opp_card_loss(latent, canon_map, opp_cards)
        tokens2, lengths2 = tokens, lengths

        action_type_i64 = action_type.to(dtype=torch.int64)
        action_logprob = act_dist.log_prob(action_type_i64)
        entropy = act_dist.entropy()

        raise_bucket_i64 = raise_bucket.to(dtype=torch.int64)
        if raise_bucket_i64.ndim > 1:
            raise_bucket_i64 = raise_bucket_i64.squeeze(-1)
        raise_bucket_i64 = raise_bucket_i64.clamp(min=0, max=NUM_RAISE_BUCKETS - 1)
        raise_logprob = raise_dist.log_prob(raise_bucket_i64)
        raise_entropy = raise_dist.entropy()

        is_raise = action_type_i64.eq(constants.ACTION_RAISE)
        logprob = action_logprob + torch.where(
            is_raise, raise_logprob, torch.zeros_like(raise_logprob)
        )
        entropy = entropy + torch.where(is_raise, raise_entropy, torch.zeros_like(raise_entropy))

        if is_packed:
            h_out = self.pack_state(tokens2, lengths2)
            c_out = c
        else:
            h_out = tokens2
            c_out = lengths2

        return PolicyOutput(
            action_type=action_type_i64,
            raise_bucket=raise_bucket_i64,
            logprob=logprob,
            entropy=entropy,
            value=value,
            h=h_out,
            c=c_out,
            decentralized_value=dec_value,
            opp_card_loss=opp_card_loss,
        )


def push_token(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    player_id: torch.Tensor,
    action_type: torch.Tensor,
    raise_bucket: torch.Tensor,
    stage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one 1-indexed public-action token per env (capped at MAX_TOKENS)."""
    batch_size = tokens.shape[0]
    device = tokens.device
    lengths_i32 = lengths.to(dtype=torch.int32, device=device)
    write_pos = lengths_i32.clamp(min=0, max=MAX_TOKENS - 1).to(dtype=torch.int64)
    in_range = lengths_i32 < MAX_TOKENS

    pid = (player_id.to(dtype=torch.int64, device=device) + 1).clamp(min=0, max=PLAYER_VOCAB - 1)
    act = (action_type.to(dtype=torch.int64, device=device) + 1).clamp(min=0, max=ACTION_VOCAB - 1)
    bkt = (raise_bucket.to(dtype=torch.int64, device=device) + 1).clamp(min=0, max=BUCKET_VOCAB - 1)
    stg = (stage.to(dtype=torch.int64, device=device) + 1).clamp(min=0, max=STAGE_VOCAB - 1)
    tok_row = torch.stack([pid, act, bkt, stg], dim=-1).to(dtype=tokens.dtype)

    batch_idx = torch.arange(batch_size, device=device, dtype=torch.int64)
    cur = tokens[batch_idx, write_pos]
    new = torch.where(in_range.view(-1, 1), tok_row, cur)
    tokens = tokens.clone()
    tokens[batch_idx, write_pos] = new

    lengths = (lengths_i32 + in_range.to(dtype=torch.int32)).clamp(min=0, max=MAX_TOKENS)
    return tokens, lengths


def reset_terminated(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    terminated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero per-env state for envs whose `terminated` flag is True."""
    reset = terminated.to(dtype=torch.bool, device=tokens.device)
    if not reset.any():
        return tokens, lengths
    tokens = torch.where(reset.view(-1, 1, 1).expand_as(tokens), torch.zeros_like(tokens), tokens)
    lengths = torch.where(
        reset, torch.zeros_like(lengths.to(dtype=torch.int32)), lengths.to(dtype=torch.int32)
    )
    return tokens, lengths


def push_action_token_packed(
    packed_state: torch.Tensor,
    *,
    player_id: torch.Tensor,
    action_type: torch.Tensor,
    raise_bucket: torch.Tensor,
    stage: torch.Tensor,
) -> torch.Tensor:
    """Append one public-action token to every env's packed `[1, B, STATE_WIDTH]` state."""
    tokens, lengths = PokerTransformerPolicyNet.unpack_state(packed_state)
    tokens, lengths = push_token(
        tokens,
        lengths,
        player_id=player_id,
        action_type=action_type,
        raise_bucket=raise_bucket,
        stage=stage,
    )
    return PokerTransformerPolicyNet.pack_state(tokens, lengths)
