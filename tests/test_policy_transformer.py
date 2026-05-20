import torch

from gpu_poker import constants
from gpu_poker.policy import NUM_RAISE_BUCKETS
from gpu_poker.policy_transformer import (
    MAX_TOKENS,
    STATE_WIDTH,
    PokerTransformerPolicyNet,
    push_action_token_packed,
    push_token,
    reset_terminated,
)


def _make_net(scalar_dim: int = 17) -> PokerTransformerPolicyNet:
    return PokerTransformerPolicyNet(
        scalar_dim=scalar_dim,
        d_model=64,  # tiny for fast CPU tests
        n_heads=4,
        n_layers=2,
        ffn_dim=128,
        max_tokens=MAX_TOKENS,
    )


def test_transformer_shapes_and_masking_cpu():
    batch = 16
    scalar_dim = 17
    net = _make_net(scalar_dim)

    cards = torch.full((batch, 7), -1, dtype=torch.int32)
    scalars = torch.zeros((batch, scalar_dim), dtype=torch.float32)
    # Only CALL is legal.
    action_mask = torch.zeros((batch, constants.NUM_ACTIONS), dtype=torch.bool)
    action_mask[:, constants.ACTION_CALL] = True

    tokens, lengths = PokerTransformerPolicyNet.init_raw_state(batch, torch.device("cpu"))
    out = net.forward_step(
        cards=cards,
        scalars=scalars,
        action_mask=action_mask,
        h=tokens,
        c=lengths,
        terminated=torch.zeros((batch,), dtype=torch.bool),
        deterministic=False,
    )

    assert out.action_type.shape == (batch,)
    assert out.raise_bucket.shape == (batch,)
    assert out.logprob.shape == (batch,)
    assert out.entropy.shape == (batch,)
    assert out.value.shape == (batch,)
    assert out.h.shape == tokens.shape
    assert out.c.shape == lengths.shape
    # Masking forces CALL.
    assert torch.all(out.action_type.eq(constants.ACTION_CALL))


def test_transformer_ppo_ratio_identity_with_frozen_weights():
    """evaluate_step must reproduce forward_step's logprob exactly when called
    with the same inputs/state. This is the same invariant as the LSTM test —
    any drift means the PPO ratio is silently biased.
    """
    torch.manual_seed(0)
    batch = 32
    scalar_dim = 17
    net = _make_net(scalar_dim)
    net.eval()

    cards = torch.randint(-1, 52, (batch, 7), dtype=torch.int32)
    scalars = torch.randn((batch, scalar_dim), dtype=torch.float32)
    action_mask = torch.ones((batch, constants.NUM_ACTIONS), dtype=torch.bool)

    tokens, lengths = PokerTransformerPolicyNet.init_raw_state(batch, torch.device("cpu"))
    # Fill some history so the test exercises non-empty token attention.
    for step in range(4):
        tokens, lengths = push_token(
            tokens,
            lengths,
            player_id=torch.randint(0, 2, (batch,), dtype=torch.int32),
            action_type=torch.randint(0, constants.NUM_ACTIONS, (batch,), dtype=torch.int32),
            raise_bucket=torch.randint(0, NUM_RAISE_BUCKETS, (batch,), dtype=torch.int32),
            stage=torch.full((batch,), step % 4, dtype=torch.int32),
        )

    with torch.no_grad():
        out = net.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=tokens,
            c=lengths,
            deterministic=False,
        )
        eval_out = net.evaluate_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            action_type=out.action_type,
            raise_bucket=out.raise_bucket,
            h=tokens,
            c=lengths,
        )

    # Same logprob, value, entropy bit-for-bit.
    assert torch.allclose(out.logprob, eval_out.logprob, atol=1e-6)
    assert torch.allclose(out.value, eval_out.value, atol=1e-6)
    assert torch.allclose(out.entropy, eval_out.entropy, atol=1e-6)


def test_transformer_terminated_resets_state():
    """terminated[i]=True must zero env i's history before the forward pass.
    Concretely: a forward with stale tokens + terminated should equal a
    forward with empty tokens.
    """
    torch.manual_seed(1)
    batch = 8
    scalar_dim = 17
    net = _make_net(scalar_dim)
    net.eval()

    cards = torch.randint(-1, 52, (batch, 7), dtype=torch.int32)
    scalars = torch.randn((batch, scalar_dim), dtype=torch.float32)
    action_mask = torch.ones((batch, constants.NUM_ACTIONS), dtype=torch.bool)

    stale_tokens, stale_lengths = PokerTransformerPolicyNet.init_raw_state(
        batch, torch.device("cpu")
    )
    for _ in range(3):
        stale_tokens, stale_lengths = push_token(
            stale_tokens,
            stale_lengths,
            player_id=torch.randint(0, 2, (batch,), dtype=torch.int32),
            action_type=torch.randint(0, constants.NUM_ACTIONS, (batch,), dtype=torch.int32),
            raise_bucket=torch.randint(0, NUM_RAISE_BUCKETS, (batch,), dtype=torch.int32),
            stage=torch.zeros((batch,), dtype=torch.int32),
        )

    empty_tokens, empty_lengths = PokerTransformerPolicyNet.init_raw_state(
        batch, torch.device("cpu")
    )

    terminated = torch.ones((batch,), dtype=torch.bool)
    with torch.no_grad():
        out_stale = net.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=stale_tokens,
            c=stale_lengths,
            terminated=terminated,
            deterministic=True,
        )
        out_empty = net.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=empty_tokens,
            c=empty_lengths,
            terminated=torch.zeros((batch,), dtype=torch.bool),
            deterministic=True,
        )
    assert torch.allclose(out_stale.value, out_empty.value, atol=1e-6)
    assert torch.equal(out_stale.action_type, out_empty.action_type)


def test_transformer_history_length_changes_output():
    """Sanity: a non-empty history must produce different value/policy than
    an empty history (the transformer is actually using the tokens).
    """
    torch.manual_seed(2)
    batch = 4
    scalar_dim = 17
    net = _make_net(scalar_dim)
    net.eval()

    cards = torch.randint(-1, 52, (batch, 7), dtype=torch.int32)
    scalars = torch.randn((batch, scalar_dim), dtype=torch.float32)
    action_mask = torch.ones((batch, constants.NUM_ACTIONS), dtype=torch.bool)

    empty_tokens, empty_lengths = PokerTransformerPolicyNet.init_raw_state(
        batch, torch.device("cpu")
    )
    filled_tokens, filled_lengths = PokerTransformerPolicyNet.init_raw_state(
        batch, torch.device("cpu")
    )
    for _ in range(5):
        filled_tokens, filled_lengths = push_token(
            filled_tokens,
            filled_lengths,
            player_id=torch.randint(0, 2, (batch,), dtype=torch.int32),
            action_type=torch.randint(0, constants.NUM_ACTIONS, (batch,), dtype=torch.int32),
            raise_bucket=torch.randint(0, NUM_RAISE_BUCKETS, (batch,), dtype=torch.int32),
            stage=torch.zeros((batch,), dtype=torch.int32),
        )

    with torch.no_grad():
        out_e = net.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=empty_tokens,
            c=empty_lengths,
            deterministic=True,
        )
        out_f = net.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=filled_tokens,
            c=filled_lengths,
            deterministic=True,
        )
    # At least one of value or action_type should differ — extremely unlikely
    # both to be unchanged by 5 tokens of history with random init.
    val_diff = (out_e.value - out_f.value).abs().max().item()
    assert val_diff > 1e-5, "transformer ignored history tokens"


def test_push_token_basic():
    """Push fills slots in order and bumps length; PAD slots stay at zero."""
    batch_size = 3
    tokens = torch.zeros((batch_size, MAX_TOKENS, 4), dtype=torch.int32)
    lengths = torch.zeros((batch_size,), dtype=torch.int32)

    tokens, lengths = push_token(
        tokens,
        lengths,
        player_id=torch.tensor([0, 1, 0], dtype=torch.int32),
        action_type=torch.tensor(
            [constants.ACTION_RAISE, constants.ACTION_CALL, constants.ACTION_FOLD],
            dtype=torch.int32,
        ),
        raise_bucket=torch.tensor([3, 0, 0], dtype=torch.int32),
        stage=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    assert lengths.tolist() == [1, 1, 1]
    # 1-indexed storage: player_id=0 stored as 1, action_type=RAISE(3) as 4, etc.
    assert tokens[0, 0].tolist() == [1, 4, 4, 1]
    assert tokens[1, 0].tolist() == [2, 3, 1, 1]
    # Second slot is still PAD.
    assert tokens[0, 1].tolist() == [0, 0, 0, 0]


def test_packed_state_roundtrip_matches_raw():
    """train.py routes state through packed [1, B, STATE_WIDTH] fp32 tensors.
    A forward_step called with packed inputs must produce the same outputs as
    one called with the raw (tokens, lengths) representation.
    """
    torch.manual_seed(3)
    batch = 4
    scalar_dim = 17
    net = _make_net(scalar_dim)
    net.eval()

    cards = torch.randint(-1, 52, (batch, 7), dtype=torch.int32)
    scalars = torch.randn((batch, scalar_dim), dtype=torch.float32)
    action_mask = torch.ones((batch, constants.NUM_ACTIONS), dtype=torch.bool)

    tokens, lengths = PokerTransformerPolicyNet.init_raw_state(batch, torch.device("cpu"))
    for _ in range(3):
        tokens, lengths = push_token(
            tokens,
            lengths,
            player_id=torch.randint(0, 2, (batch,), dtype=torch.int32),
            action_type=torch.randint(0, constants.NUM_ACTIONS, (batch,), dtype=torch.int32),
            raise_bucket=torch.randint(0, NUM_RAISE_BUCKETS, (batch,), dtype=torch.int32),
            stage=torch.zeros((batch,), dtype=torch.int32),
        )

    packed_h = PokerTransformerPolicyNet.pack_state(tokens, lengths)
    assert packed_h.shape == (1, batch, STATE_WIDTH)
    packed_c = torch.zeros_like(packed_h)

    with torch.no_grad():
        out_raw = net.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=tokens,
            c=lengths,
            deterministic=True,
        )
        out_packed = net.forward_step(
            cards=cards,
            scalars=scalars,
            action_mask=action_mask,
            h=packed_h,
            c=packed_c,
            deterministic=True,
        )

    assert torch.allclose(out_raw.value, out_packed.value, atol=1e-6)
    assert torch.equal(out_raw.action_type, out_packed.action_type)

    # Packed h_out must have the right shape so train.py's reshape arithmetic
    # keeps working.
    assert out_packed.h.shape == (1, batch, STATE_WIDTH)


def test_push_action_token_packed_matches_raw_push():
    """The packed-state push helper used by train.py after env.step must
    leave the underlying tokens/lengths in the same state as a raw push.
    """
    torch.manual_seed(4)
    batch = 6
    packed_h, _ = PokerTransformerPolicyNet.init_state(batch, 0, torch.device("cpu"))
    raw_tokens, raw_lengths = PokerTransformerPolicyNet.init_raw_state(batch, torch.device("cpu"))

    pid = torch.randint(0, 2, (batch,), dtype=torch.int32)
    at = torch.randint(0, constants.NUM_ACTIONS, (batch,), dtype=torch.int32)
    rb = torch.randint(0, NUM_RAISE_BUCKETS, (batch,), dtype=torch.int32)
    stg = torch.full((batch,), 2, dtype=torch.int32)

    packed_h = push_action_token_packed(
        packed_h, player_id=pid, action_type=at, raise_bucket=rb, stage=stg
    )
    raw_tokens, raw_lengths = push_token(
        raw_tokens, raw_lengths, player_id=pid, action_type=at, raise_bucket=rb, stage=stg
    )

    unp_tokens, unp_lengths = PokerTransformerPolicyNet.unpack_state(packed_h)
    assert torch.equal(unp_tokens, raw_tokens)
    assert torch.equal(unp_lengths, raw_lengths)


def test_reset_terminated_zeros_state():
    batch_size = 3
    tokens = torch.ones((batch_size, MAX_TOKENS, 4), dtype=torch.int32)
    lengths = torch.full((batch_size,), 5, dtype=torch.int32)
    terminated = torch.tensor([True, False, True], dtype=torch.bool)
    tokens, lengths = reset_terminated(tokens, lengths, terminated)
    assert lengths.tolist() == [0, 5, 0]
    assert torch.all(tokens[0] == 0)
    assert torch.all(tokens[2] == 0)
    assert torch.all(tokens[1] == 1)
