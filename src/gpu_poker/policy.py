from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from gpu_poker import constants as constants

# Discrete raise-size buckets. Each entry is a *pot fraction*; sentinel values
# `MIN_RAISE_BUCKET` and `ALL_IN_BUCKET` map to min_raise and max_raise (all-in)
# respectively. This matches the ReBeL / Slumbot / Libratus action abstraction
# (Brown et al. 2020, Brown & Sandholm 2018) and is a prerequisite for any
# future CFR-style search at decision time.
MIN_RAISE_BUCKET = -1.0
ALL_IN_BUCKET = -2.0
RAISE_BUCKET_FRACTIONS: tuple[float, ...] = (
    MIN_RAISE_BUCKET,  # bucket 0: min-raise
    0.5,  # bucket 1: 0.5 * pot
    0.75,  # bucket 2: 0.75 * pot
    1.0,  # bucket 3: 1.0 * pot
    1.5,  # bucket 4: 1.5 * pot
    2.5,  # bucket 5: 2.5 * pot
    ALL_IN_BUCKET,  # bucket 6: all-in
)
NUM_RAISE_BUCKETS = len(RAISE_BUCKET_FRACTIONS)


@dataclass(frozen=True)
class PolicyOutput:
    action_type: torch.Tensor  # [B] int64
    raise_bucket: torch.Tensor  # [B] int64 in [0, NUM_RAISE_BUCKETS)
    logprob: torch.Tensor  # [B] float32
    entropy: torch.Tensor  # [B] float32
    value: torch.Tensor  # [B] float32
    h: torch.Tensor  # [1, B, H]
    c: torch.Tensor  # [1, B, H]


class PokerPolicyNet(nn.Module):
    """
    Minimal actor-critic with LSTM for poker.

    Inputs are the env outputs:
    - cards: int32 tensor [B, 7] values in [-1, 51]
    - scalars: float32 tensor [B, S]
    - action_mask: bool tensor [B, 4]
    - terminated: optional bool tensor [B] used to reset RNN state

    Outputs:
    - action_type categorical over 4 actions (masked)
    - raise_frac ~ Beta(alpha,beta) in [0,1] (used only when action_type==RAISE)
    - value estimate
    """

    def __init__(
        self,
        *,
        scalar_dim: int,
        card_embed_dim: int = 32,
        mlp_dim: int = 128,
        torso_layers: int = 1,
        lstm_hidden: int = 128,
        head_layers: int = 0,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.card_embed_dim = int(card_embed_dim)
        self.mlp_dim = int(mlp_dim)
        self.torso_layers = int(torso_layers)
        self.lstm_hidden = int(lstm_hidden)
        self.head_layers = int(head_layers)
        self.head_dim = int(head_dim) if head_dim is not None else None

        # Map [-1,51] -> [0,52] using (+1) offset; 0 is PAD.
        self.card_embed = nn.Embedding(num_embeddings=53, embedding_dim=self.card_embed_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
        )

        in_dim = self.mlp_dim + 7 * self.card_embed_dim
        if self.torso_layers <= 0:
            raise ValueError(f"torso_layers must be >= 1, got {self.torso_layers}")
        torso_layers_: list[nn.Module] = [nn.Linear(in_dim, self.mlp_dim), nn.ReLU()]
        for _ in range(self.torso_layers - 1):
            torso_layers_.extend([nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU()])
        self.pre_lstm = nn.Sequential(*torso_layers_)

        self.lstm = nn.LSTM(input_size=self.mlp_dim, hidden_size=self.lstm_hidden, batch_first=True)

        if self.head_layers < 0:
            raise ValueError(f"head_layers must be >= 0, got {self.head_layers}")
        head_dim_i = self.lstm_hidden if self.head_dim is None else int(self.head_dim)
        if self.head_layers == 0:
            self.post_lstm = nn.Identity()
            post_dim = self.lstm_hidden
        else:
            head_layers_: list[nn.Module] = [nn.Linear(self.lstm_hidden, head_dim_i), nn.ReLU()]
            for _ in range(self.head_layers - 1):
                head_layers_.extend([nn.Linear(head_dim_i, head_dim_i), nn.ReLU()])
            self.post_lstm = nn.Sequential(*head_layers_)
            post_dim = head_dim_i

        self.action_head = nn.Linear(post_dim, constants.NUM_ACTIONS)
        self.value_head = nn.Linear(post_dim, 1)
        # Discrete raise-size logits over NUM_RAISE_BUCKETS pot-fraction buckets
        # (see RAISE_BUCKET_FRACTIONS above). A categorical over a small discrete
        # set is what every modern HUNL bot uses; it also keeps the action space
        # finite so we can layer CFR-style search on top later.
        self.raise_head = nn.Linear(post_dim, NUM_RAISE_BUCKETS)

    @staticmethod
    def init_state(
        batch: int, hidden: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros((1, batch, hidden), device=device)
        c_ = torch.zeros((1, batch, hidden), device=device)
        return h, c_

    def _masked_action_logits(
        self, logits: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        # logits: [B,4], mask: [B,4] bool
        # Use scalar directly (CUDA graph compatible - no tensor creation)
        neg_inf = float(torch.finfo(logits.dtype).min)
        return torch.where(action_mask, logits, neg_inf)

    def _raise_dist(self, raise_logits: torch.Tensor) -> Categorical:
        # raise_logits: [B, NUM_RAISE_BUCKETS]. Force fp32 so sampling and
        # log_prob are bit-stable across mixed-precision boundaries.
        return Categorical(logits=raise_logits.to(dtype=torch.float32))

    def _forward_latent(
        self,
        *,
        cards: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        terminated: torch.Tensor | None,
    ) -> tuple[Categorical, Categorical, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_state = h
        c_state = c
        if terminated is not None:
            reset = terminated.to(dtype=torch.bool).view(1, -1, 1)
            h_state = torch.where(reset, torch.zeros_like(h_state), h_state)
            c_state = torch.where(reset, torch.zeros_like(c_state), c_state)

        cards_i64 = cards.to(dtype=torch.int64)
        cards_idx = (cards_i64 + 1).clamp_(0, 52)
        card_emb = self.card_embed(cards_idx)  # [B,7,E]
        card_feat = card_emb.reshape(card_emb.shape[0], -1)  # [B,7E]

        scal_feat = self.scalar_mlp(scalars)
        x = torch.cat([card_feat, scal_feat], dim=-1)
        x = self.pre_lstm(x).unsqueeze(1)  # [B,1,D]

        y, (h2, c2) = self.lstm(x, (h_state, c_state))  # y: [B,1,H]
        y = y.squeeze(1)  # [B,H]
        y = self.post_lstm(y)

        logits = self.action_head(y)

        # Defensive: env should always have at least one legal action (fold),
        # but avoid NaNs if an all-false mask sneaks in.
        mask = action_mask.to(dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[-1] != constants.NUM_ACTIONS:
            raise ValueError(
                f"Expected action_mask [B,{constants.NUM_ACTIONS}], got {tuple(mask.shape)}"
            )

        # Ensure fold is always valid when mask is empty (avoids NaN in softmax).
        # Done unconditionally to be CUDA graph compatible (no .any() CPU sync).
        mask_sum = mask.sum(dim=-1)
        mask = mask.clone()
        mask[:, constants.ACTION_FOLD] = mask[:, constants.ACTION_FOLD] | (mask_sum == 0)

        masked_logits = self._masked_action_logits(logits, mask)
        act_dist = Categorical(logits=masked_logits)

        raise_logits = self.raise_head(y)
        raise_dist = self._raise_dist(raise_logits)

        value = self.value_head(y).squeeze(-1)
        return act_dist, raise_dist, value, h2, c2

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
    ) -> PolicyOutput:
        """
        Single-step forward for a batch of environments.
        """
        act_dist, raise_dist, value, h2, c2 = self._forward_latent(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=h,
            c=c,
            terminated=terminated,
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

        entropy = act_dist.entropy()
        raise_entropy = raise_dist.entropy()
        entropy = entropy + torch.where(is_raise, raise_entropy, torch.zeros_like(raise_entropy))

        return PolicyOutput(
            action_type=action_type,
            raise_bucket=raise_bucket,
            logprob=logprob,
            entropy=entropy,
            value=value,
            h=h2,
            c=c2,
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
    ) -> PolicyOutput:
        """
        Computes logprob/value for provided actions (no sampling).
        """
        act_dist, raise_dist, value, h2, c2 = self._forward_latent(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=h,
            c=c,
            terminated=terminated,
        )

        action_type_i64 = action_type.to(dtype=torch.int64)
        action_logprob = act_dist.log_prob(action_type_i64)
        entropy = act_dist.entropy()

        # raise_bucket is the stored discrete bucket index in [0, NUM_RAISE_BUCKETS).
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

        return PolicyOutput(
            action_type=action_type_i64,
            raise_bucket=raise_bucket_i64,
            logprob=logprob,
            entropy=entropy,
            value=value,
            h=h2,
            c=c2,
        )


def raise_frac_to_amount(
    *, raise_frac: torch.Tensor, min_raise: torch.Tensor, max_raise: torch.Tensor
) -> torch.Tensor:
    """
    Legacy helper kept for scripted-bot rollouts. Maps a continuous fraction
    in [0, 1] to a raise delta within [min_raise, max_raise]. Trained policies
    use the discrete-bucket pipeline via raise_bucket_to_amount.
    """
    raise_frac = raise_frac.squeeze(-1).to(dtype=torch.float32)
    min_i = min_raise.to(dtype=torch.int32)
    max_i = max_raise.to(dtype=torch.int32)
    all_in_only = max_i < min_i
    amount_f = min_i.to(dtype=torch.float32) + raise_frac * (max_i - min_i).to(dtype=torch.float32)
    amount_i = amount_f.round().to(dtype=torch.int32)
    amount_i = torch.where(all_in_only, max_i, amount_i)
    lo = torch.minimum(min_i, max_i)
    hi = torch.maximum(min_i, max_i)
    return amount_i.clamp(min=lo, max=hi)


def raise_bucket_to_amount(
    *,
    raise_bucket: torch.Tensor,
    pot_total: torch.Tensor,
    min_raise: torch.Tensor,
    max_raise: torch.Tensor,
) -> torch.Tensor:
    """
    Map a discrete raise bucket index to a raise delta (chips), clamped to
    [min_raise, max_raise]. Per-bucket pot fractions are defined by
    RAISE_BUCKET_FRACTIONS; the special MIN_RAISE_BUCKET and ALL_IN_BUCKET
    sentinels map to min_raise and max_raise respectively.

    Args:
        raise_bucket: [B] int64 bucket indices (or [B, 1] which gets squeezed).
        pot_total: [B] int32 — total chips committed (pot + both round bets).
        min_raise, max_raise: [B] int32 raise-delta bounds from the env kernels.

    Returns:
        [B] int32 raise delta in chips.
    """
    bucket = raise_bucket.squeeze(-1) if raise_bucket.ndim > 1 else raise_bucket
    bucket = bucket.to(dtype=torch.int64)
    pot_f = pot_total.to(dtype=torch.float32)
    min_i = min_raise.to(dtype=torch.int32)
    max_i = max_raise.to(dtype=torch.int32)

    # Build a per-row target delta by indexing the fraction table.
    fractions = torch.tensor(
        RAISE_BUCKET_FRACTIONS, dtype=torch.float32, device=raise_bucket.device
    )
    frac = fractions[bucket.clamp_(0, NUM_RAISE_BUCKETS - 1)]

    pot_amt = (frac * pot_f).round().to(dtype=torch.int32)
    amount_i = pot_amt
    amount_i = torch.where(frac.eq(MIN_RAISE_BUCKET), min_i, amount_i)
    amount_i = torch.where(frac.eq(ALL_IN_BUCKET), max_i, amount_i)

    # In some all-in situations max_raise < min_raise; the only legal raise is
    # the all-in delta == max_raise.
    all_in_only = max_i < min_i
    amount_i = torch.where(all_in_only, max_i, amount_i)

    lo = torch.minimum(min_i, max_i)
    hi = torch.maximum(min_i, max_i)
    return amount_i.clamp(min=lo, max=hi)
