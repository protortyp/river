"""
Unit tests for the "force to legal" illegal-action policy.
"""

import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import actions
from gpu_poker.struct_types import GameState

wp.init()

try:
    DEVICE = "cuda" if wp.is_device_available("cuda") else "cpu"
except RuntimeError:
    DEVICE = "cpu"


def _alloc_state() -> GameState:
    n_envs = 1
    s = GameState()
    s.stage = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.button = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.active_player = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.done = wp.zeros(n_envs, dtype=wp.bool, device=DEVICE)

    s.deck = wp.zeros((n_envs, c.NUM_CARDS), dtype=wp.int32, device=DEVICE)
    s.deck_top = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.hole_cards = wp.zeros((n_envs, c.NUM_PLAYERS, c.HOLE_CARDS), dtype=wp.int32, device=DEVICE)
    s.community_cards = wp.zeros((n_envs, c.COMMUNITY_CARDS), dtype=wp.int32, device=DEVICE)
    s.num_community = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)

    s.stacks = wp.zeros((n_envs, c.NUM_PLAYERS), dtype=wp.int32, device=DEVICE)
    s.bets = wp.zeros((n_envs, c.NUM_PLAYERS), dtype=wp.int32, device=DEVICE)
    s.initial_stacks = wp.zeros((n_envs, c.NUM_PLAYERS), dtype=wp.int32, device=DEVICE)
    s.pot = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)

    s.last_raise = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.num_raises = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.last_aggressor = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.num_actions = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.actions_this_street = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.last_action_type = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.last_action_amount = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.last_action_was_raise = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.rng_state = wp.zeros(n_envs, dtype=wp.uint32, device=DEVICE)
    s.episode_id = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.cfg_starting_stack = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.cfg_small_blind = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    s.cfg_big_blind = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
    return s


class TestIllegalActionPolicy:
    def test_check_when_facing_bet_becomes_call(self):
        s = _alloc_state()
        out_type = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_invalid = wp.zeros(1, dtype=wp.bool, device=DEVICE)

        @wp.kernel
        def k(
            s: GameState,
            out_type: wp.array(dtype=wp.int32),
            out_invalid: wp.array(dtype=wp.bool),
        ):
            tid = wp.tid()
            s.active_player[tid] = 0
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 10
            s.stacks[tid, 0] = 100
            s.stacks[tid, 1] = 100
            s.last_raise[tid] = 10

            packed = actions.force_legal_action(s, tid, 0, c.ACTION_CHECK, 0)
            out_invalid[tid] = actions.unpack_invalid(packed)
            out_type[tid] = actions.unpack_action_type(packed)

        wp.launch(k, dim=1, inputs=[s, out_type, out_invalid])
        assert out_invalid.numpy()[0] == True  # noqa: E712
        assert out_type.numpy()[0] == c.ACTION_CALL

    def test_raise_amount_clamped_to_max(self):
        s = _alloc_state()
        out_amount = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_invalid = wp.zeros(1, dtype=wp.bool, device=DEVICE)

        @wp.kernel
        def k(
            s: GameState,
            out_amount: wp.array(dtype=wp.int32),
            out_invalid: wp.array(dtype=wp.bool),
        ):
            tid = wp.tid()
            s.active_player[tid] = 0

            # Facing 10, both have stacks, but opponent only has 5 behind.
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 10
            s.stacks[tid, 0] = 100
            s.stacks[tid, 1] = 5
            s.last_raise[tid] = 10

            packed = actions.force_legal_action(s, tid, 0, c.ACTION_RAISE, 999)
            out_invalid[tid] = actions.unpack_invalid(packed)
            out_amount[tid] = actions.unpack_amount(packed)

        wp.launch(k, dim=1, inputs=[s, out_amount, out_invalid])

        # Can't raise; should fall back to CALL (amount 0), still marked invalid.
        assert out_invalid.numpy()[0] == True  # noqa: E712
        assert out_amount.numpy()[0] == 0

    def test_undersized_raise_deep_stack_clamps_to_min_not_shove(self):
        """Regression: force_legal_action used to substitute an all-in for any
        sub-min RAISE amount when `all_in_delta <= max_d`, silently turning a
        tiny illegal bet into a stack-pushing action. After the fix, it should
        clamp UP to min_d when min_d is itself a legal size."""
        s = _alloc_state()
        out_type = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_amount = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_invalid = wp.zeros(1, dtype=wp.bool, device=DEVICE)

        @wp.kernel
        def k(
            s: GameState,
            out_type: wp.array(dtype=wp.int32),
            out_amount: wp.array(dtype=wp.int32),
            out_invalid: wp.array(dtype=wp.bool),
        ):
            tid = wp.tid()
            s.active_player[tid] = 0
            # Deep stacks, no bet, big blind = 2.
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 0
            s.stacks[tid, 0] = 100
            s.stacks[tid, 1] = 100
            s.last_raise[tid] = 2
            s.cfg_big_blind[tid] = 2

            # Submit illegal sub-min raise (amount=0).
            packed = actions.force_legal_action(s, tid, 0, c.ACTION_RAISE, 0)
            out_invalid[tid] = actions.unpack_invalid(packed)
            out_type[tid] = actions.unpack_action_type(packed)
            out_amount[tid] = actions.unpack_amount(packed)

        wp.launch(k, dim=1, inputs=[s, out_type, out_amount, out_invalid])
        assert out_invalid.numpy()[0] == True  # noqa: E712
        assert out_type.numpy()[0] == c.ACTION_RAISE
        # Min raise == big blind == 2. MUST NOT be the full 100-stack shove.
        assert out_amount.numpy()[0] == 2

    def test_oversized_raise_with_valid_raise_available_is_marked_invalid_and_clamped(self):
        s = _alloc_state()
        out_type = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_amount = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_invalid = wp.zeros(1, dtype=wp.bool, device=DEVICE)

        @wp.kernel
        def k(
            s: GameState,
            out_type: wp.array(dtype=wp.int32),
            out_amount: wp.array(dtype=wp.int32),
            out_invalid: wp.array(dtype=wp.bool),
        ):
            tid = wp.tid()
            s.active_player[tid] = 0
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 0
            s.stacks[tid, 0] = 100
            s.stacks[tid, 1] = 10
            s.last_raise[tid] = 2
            s.cfg_big_blind[tid] = 2

            packed = actions.force_legal_action(s, tid, 0, c.ACTION_RAISE, 50)
            out_invalid[tid] = actions.unpack_invalid(packed)
            out_type[tid] = actions.unpack_action_type(packed)
            out_amount[tid] = actions.unpack_amount(packed)

        wp.launch(k, dim=1, inputs=[s, out_type, out_amount, out_invalid])

        assert out_invalid.numpy()[0] == True  # noqa: E712
        assert out_type.numpy()[0] == c.ACTION_RAISE
        assert out_amount.numpy()[0] == 10
