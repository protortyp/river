"""Transformer backbone for the HUNL policy.

Drop-in alternative to PokerPolicyNet (LSTM). Maintains a per-env token
buffer of public action tokens within the current hand; on each step the
agent's current observation (private cards + scalars) is appended as the
final "query" token and the encoder's output at that last position drives
the action / raise / value heads.

Tokens are 4-tuples (player_id, action_type, raise_bucket, stage), each
embedded separately and summed. The buffer is reset across hand boundaries
the same way the LSTM (h, c) tuple is — by zeroing `lengths` and `tokens`
when `terminated[i]` is True.

I/O surface matches PokerPolicyNet's `forward_step` / `evaluate_step` so
train.py can switch backbones with a config flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from gpu_poker import constants as constants
from gpu_poker.policy import NUM_RAISE_BUCKETS

# Hand history is at most ~20 public actions in HUNL with our bucket grid
# (4 betting rounds, ≤5 raises per round, alternating actors). 32 gives plenty
# of slack while keeping the per-env memory bounded.
MAX_TOKENS: int = 32
# Number of int fields per public-action token: (player, action, bucket, stage).
TOKEN_FIELDS: int = 4
# Packed-state layout for backbone-agnostic RolloutBuffer storage:
#   [:MAX_TOKENS*TOKEN_FIELDS] = flattened token table (int values as fp32,
#                                round-trip is exact at our vocab sizes ≤ 8)
#   [MAX_TOKENS*TOKEN_FIELDS]  = current sequence length
# Stored as a `[1, B, STATE_WIDTH]` fp32 tensor so it slots into the existing
# train.py `h` buffer without changing buffer shapes or reshape arithmetic.
STATE_WIDTH: int = MAX_TOKENS * TOKEN_FIELDS + 1

# Sentinel values written into the per-env token table for empty slots. Picked
# so that the embedding lookup at PAD_IDX produces a fixed pad embedding which
# the attention mask will then zero out.
PAD_IDX: int = 0
# Embedding tables use index 0 as PAD; real values are stored at idx+1.
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
    # Keep the same field names as the LSTM PolicyOutput so train.py can stay
    # backbone-agnostic. `h` carries the updated token table [batch_size, MAX_TOKENS, 4]
    # and `c` carries the per-env length [B] (broadcast to a [1, B, 1] shape
    # so RolloutBuffer.h/c storage works without per-step custom code).
    h: torch.Tensor
    c: torch.Tensor


def _sinusoidal_pos_enc(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal positional encodings, [seq_len, d_model] fp32."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d_model // 2, device=device, dtype=torch.float32)
    div = torch.exp(-(i * 2.0) * (torch.log(torch.tensor(10000.0, device=device)) / d_model))
    angles = pos * div  # [seq_len, d_model/2]
    pe = torch.zeros((seq_len, d_model), device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class PokerTransformerPolicyNet(nn.Module):
    """Encoder-only transformer policy. See module docstring.

    Same I/O contract as `PokerPolicyNet`:
      forward_step(cards, scalars, action_mask, h, c, terminated, deterministic)
      evaluate_step(cards, scalars, action_mask, action_type, raise_bucket, h, c, terminated)

    Where `h` is the int32 token table [batch_size, MAX_TOKENS, 4] and `c` is the int32
    length vector reshaped to [1, B, 1] (so it fits the existing
    RolloutBuffer.h/c float32 storage without code changes — see
    `pack_state`/`unpack_state`).
    """

    # Registered buffer (see __init__: register_buffer("pos_enc", ...)). Declared
    # here so type checkers resolve `self.pos_enc` as a Tensor rather than the
    # `Tensor | Module` union that nn.Module.__getattr__ falls back to.
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

        # ---- Token embeddings (history tokens) ----
        self.player_emb = nn.Embedding(PLAYER_VOCAB, self.d_model, padding_idx=PAD_IDX)
        self.action_emb = nn.Embedding(ACTION_VOCAB, self.d_model, padding_idx=PAD_IDX)
        self.bucket_emb = nn.Embedding(BUCKET_VOCAB, self.d_model, padding_idx=PAD_IDX)
        self.stage_emb = nn.Embedding(STAGE_VOCAB, self.d_model, padding_idx=PAD_IDX)

        # ---- Current-state token (private cards + scalars + action mask) ----
        # Card embedding: [-1, 51] -> [0, 52] (PAD=0).
        self.card_embed = nn.Embedding(53, self.card_embed_dim)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.card_proj = nn.Linear(7 * self.card_embed_dim, self.d_model)
        # A learned "this is the current-state token" type embedding, so the
        # transformer can distinguish it from action-history tokens.
        self.cur_type_emb = nn.Parameter(torch.zeros(self.d_model))

        # ---- Transformer encoder ----
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: with norm_first=True, PyTorch's
        # TransformerEncoder cannot use the NestedTensor fast path anyway
        # ("self.use_nested_tensor is False because encoder_layer.norm_first
        # was True" warning) and the runtime branch deciding so on every
        # forward call has nonzero cost. Setting this explicit silences
        # the warning AND skips the eligibility check.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.n_layers, enable_nested_tensor=False
        )

        # Positional encoding for max_tokens history slots + 1 current-state
        # token at the end.
        pe = _sinusoidal_pos_enc(self.max_tokens + 1, self.d_model, torch.device("cpu"))
        self.register_buffer("pos_enc", pe, persistent=False)

        # ---- Heads ----
        self.action_head = nn.Linear(self.d_model, constants.NUM_ACTIONS)
        self.value_head = nn.Linear(self.d_model, 1)
        self.raise_head = nn.Linear(self.d_model, NUM_RAISE_BUCKETS)

    # train.py uses `net.lstm_hidden` as the trailing state-dim everywhere
    # (buffer alloc, h/c init, PPO reshape). Returning STATE_WIDTH here makes
    # the train loop backbone-agnostic when h is treated as a packed state.
    @property
    def lstm_hidden(self) -> int:
        return STATE_WIDTH

    # ------------------------------------------------------------------ state

    @staticmethod
    def init_state(
        batch: int, hidden: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match PokerPolicyNet.init_state signature for backbone-agnostic
        callers (train.py). `hidden` is ignored — the state shape is fixed
        by STATE_WIDTH. Returns the *packed* state representation:
            h: [1, batch, STATE_WIDTH] fp32  (tokens + length packed)
            c: [1, batch, STATE_WIDTH] fp32  (unused; preserved for symmetry)
        """
        h = torch.zeros((1, batch, STATE_WIDTH), device=device, dtype=torch.float32)
        c = torch.zeros((1, batch, STATE_WIDTH), device=device, dtype=torch.float32)
        return h, c

    @staticmethod
    def init_raw_state(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw (unpacked) state for direct test use. Returns:
        tokens:  int32 [batch_size, MAX_TOKENS, 4]
        lengths: int32 [B]
        """
        tokens = torch.zeros((batch, MAX_TOKENS, TOKEN_FIELDS), device=device, dtype=torch.int32)
        lengths = torch.zeros((batch,), device=device, dtype=torch.int32)
        return tokens, lengths

    # -------------------------------------------------------- pack/unpack API

    @staticmethod
    def unpack_state(h_packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[1, B, STATE_WIDTH] fp32 -> (tokens int32 [batch_size, MAX_TOKENS, 4],
        lengths int32 [B]). Values are tiny ints so fp32 round-trip is exact;
        we round-to-nearest defensively in case earlier casts introduced
        sub-ULP drift.
        """
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
        """(tokens [batch_size, MAX_TOKENS, 4] int, lengths [B] int) -> [1, B, STATE_WIDTH] fp32."""
        batch_size = tokens.shape[0]
        tok_flat = tokens.to(dtype=torch.float32).reshape(batch_size, MAX_TOKENS * TOKEN_FIELDS)
        len_col = lengths.to(dtype=torch.float32).reshape(batch_size, 1)
        packed = torch.cat([tok_flat, len_col], dim=-1)
        return packed.unsqueeze(0)

    # -------------------------------------------------------------- internals

    def _embed_history(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: int32 [batch_size, T, 4] -> emb fp32 [batch_size, T, D]. PAD rows -> zero."""
        # tokens stores 1-indexed values (0 = PAD), see push_token().
        t_i64 = tokens.to(dtype=torch.int64).clamp_min(0)
        e = (
            self.player_emb(t_i64[..., 0])
            + self.action_emb(t_i64[..., 1])
            + self.bucket_emb(t_i64[..., 2])
            + self.stage_emb(t_i64[..., 3])
        )
        return e

    def _build_current_token(self, cards: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """[batch_size, D] embedding of the agent's current observation."""
        cards_i64 = cards.to(dtype=torch.int64)
        cards_idx = (cards_i64 + 1).clamp_(0, 52)
        card_emb = self.card_embed(cards_idx)  # [batch_size, 7, E]
        card_feat = card_emb.reshape(card_emb.shape[0], -1)
        card_tok = self.card_proj(card_feat)
        scal_tok = self.scalar_mlp(scalars)
        return card_tok + scal_tok + self.cur_type_emb

    def _forward_latent(
        self,
        *,
        cards: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        terminated: torch.Tensor | None,
    ) -> tuple[Categorical, Categorical, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply terminal resets to the state we were handed *before* using it,
        # mirroring the LSTM backbone: a terminated env's prior hand should not
        # leak tokens into the new hand.
        if terminated is not None:
            reset = terminated.to(dtype=torch.bool)
            if reset.any():
                # Zero out tokens and lengths for terminated envs.
                mask = reset.view(-1, 1, 1).expand_as(tokens)
                tokens = torch.where(mask, torch.zeros_like(tokens), tokens)
                lengths = torch.where(reset, torch.zeros_like(lengths), lengths)

        batch_size = cards.shape[0]
        device = cards.device

        # Layout: position 0 is the current-state ("query") token, positions
        # 1..MAX_TOKENS are the public action history (oldest -> newest).
        # Putting the always-valid current token at index 0 ensures every
        # position has at least one non-padded key to attend to — without
        # that, padded history rows can end up with a fully-masked attention
        # row and produce NaN logits that poison subsequent layers.
        # We deliberately use no causal mask: the encoder runs once per
        # decision point and we only read the position-0 output. History
        # tokens being able to attend to each other freely is fine because
        # we never consume their outputs.
        hist_emb = self._embed_history(tokens)  # [batch_size, MAX_TOKENS, D]
        cur_tok = self._build_current_token(cards, scalars).unsqueeze(1)  # [batch_size, 1, D]
        x = torch.cat([cur_tok, hist_emb], dim=1)  # [batch_size, 1+MAX_TOKENS, D]
        x = x + self.pos_enc.to(x.dtype).unsqueeze(0)

        # Key-padding mask: position 0 (current token) always valid; history
        # positions valid iff (history_slot < lengths[b]).
        pos = torch.arange(self.max_tokens, device=device).unsqueeze(0)  # [1, MAX_TOKENS]
        hist_pad = pos >= lengths.to(dtype=torch.int64).unsqueeze(1)  # [batch_size, MAX_TOKENS]
        cur_pad = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        key_padding_mask = torch.cat([cur_pad, hist_pad], dim=1)  # [batch_size, 1+MAX_TOKENS]

        y = self.encoder(x, mask=None, src_key_padding_mask=key_padding_mask)
        last = y[:, 0, :]  # [batch_size, D] — output at the current-state token position

        # ---- action / value / raise heads (same logic as PokerPolicyNet) ----
        logits = self.action_head(last)
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
        masked_logits = torch.where(mask, logits, neg_inf)
        act_dist = Categorical(logits=masked_logits)

        raise_logits = self.raise_head(last)
        raise_dist = Categorical(logits=raise_logits.to(dtype=torch.float32))

        value = self.value_head(last).squeeze(-1)

        # Return the *input* (post-reset) tokens/lengths as the "next state".
        # The caller is responsible for appending the action token after the
        # env step actually happens (see push_token()).
        return act_dist, raise_dist, value, tokens, lengths

    # ----------------------------------------------------------------- public

    def _coerce_input_state(
        self, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Accept either a packed [1, B, STATE_WIDTH] h (train.py path) or a
        raw (tokens, lengths) pair (unit-test / standalone path). Returns
        `(tokens, lengths, is_packed)`. `c` is consumed only when `h` is the
        raw tokens tensor and `c` carries the raw lengths.
        """
        if h.dtype.is_floating_point and h.ndim == 3 and h.shape[-1] == STATE_WIDTH:
            tokens, lengths = self.unpack_state(h)
            return tokens, lengths, True
        # Raw path: h is int tokens [batch_size, MAX_TOKENS, 4]; c is lengths [B] or
        # [1, B, 1]-ish (we just flatten to [B]).
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
    ) -> PolicyOutput:
        tokens, lengths, is_packed = self._coerce_input_state(h, c)

        act_dist, raise_dist, value, tokens2, lengths2 = self._forward_latent(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            tokens=tokens,
            lengths=lengths,
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
        entropy = act_dist.entropy() + torch.where(
            is_raise, raise_dist.entropy(), torch.zeros_like(raise_logprob)
        )

        if is_packed:
            h_out = self.pack_state(tokens2, lengths2)
            c_out = c  # passed through unchanged; not used by this backbone.
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
        tokens, lengths, is_packed = self._coerce_input_state(h, c)

        act_dist, raise_dist, value, tokens2, lengths2 = self._forward_latent(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            tokens=tokens,
            lengths=lengths,
            terminated=terminated,
        )

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
    """Append one public-action token per env to the per-env buffer.

    Embedding tables use index 0 = PAD, so we store *1-indexed* values
    (`value + 1`). Empty slots stay at zero and embed to the pad row.

    Out-of-bounds writes (when an env's history is already full) are silently
    dropped — at MAX_TOKENS=32 this should never fire in HUNL but it keeps the
    kernel branch-free.

    All inputs are int (any width); returns (tokens, lengths) on the same
    device as `tokens`.
    """
    batch_size = tokens.shape[0]
    device = tokens.device
    lengths_i32 = lengths.to(dtype=torch.int32, device=device)
    write_pos = lengths_i32.clamp(min=0, max=MAX_TOKENS - 1).to(dtype=torch.int64)
    in_range = lengths_i32 < MAX_TOKENS  # bool [B]

    # Build the 4-tuple (1-indexed for embedding pad).
    pid = (player_id.to(dtype=torch.int64, device=device) + 1).clamp(min=0, max=PLAYER_VOCAB - 1)
    act = (action_type.to(dtype=torch.int64, device=device) + 1).clamp(min=0, max=ACTION_VOCAB - 1)
    bkt = (raise_bucket.to(dtype=torch.int64, device=device) + 1).clamp(min=0, max=BUCKET_VOCAB - 1)
    stg = (stage.to(dtype=torch.int64, device=device) + 1).clamp(min=0, max=STAGE_VOCAB - 1)
    tok_row = torch.stack([pid, act, bkt, stg], dim=-1).to(dtype=tokens.dtype)  # [batch_size, 4]

    # Only write rows where in_range is True. Use a where on the destination
    # slice to keep the op shape-stable for graph capture.
    batch_idx = torch.arange(batch_size, device=device, dtype=torch.int64)
    cur = tokens[batch_idx, write_pos]  # [batch_size, 4]
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
    """In-place-ish append of one public-action token to every env's packed
    state buffer. Used by train.py after `env.step()` to record the action
    that was just taken — pushed to BOTH seats' state buffers because the
    action is public.

    Args:
        packed_state: [1, B, STATE_WIDTH] fp32 packed transformer state.
        player_id, action_type, raise_bucket, stage: [B] int (any width).
            `raise_bucket` is ignored for non-RAISE actions but must be a
            valid bucket index (the embedding still gets summed in; using
            bucket 0 for non-RAISE actions is the convention).

    Returns:
        New packed_state [1, B, STATE_WIDTH] fp32 with the token appended
        and length incremented (capped at MAX_TOKENS).
    """
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
