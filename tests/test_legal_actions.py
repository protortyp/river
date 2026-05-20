"""
Unit tests for legal action mask / raise bounds logic.
"""

import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import legal_actions
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


class TestLegalActions:
    def test_no_bet_allows_check_and_raise(self):
        s = _alloc_state()
        mask = wp.zeros((1, c.NUM_ACTIONS), dtype=wp.bool, device=DEVICE)
        min_r = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        max_r = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        @wp.kernel
        def k(
            s: GameState,
            mask: wp.array(dtype=wp.bool, ndim=2),
            min_r: wp.array(dtype=wp.int32),
            max_r: wp.array(dtype=wp.int32),
        ):  # noqa: E501
            tid = wp.tid()
            s.active_player[tid] = 0
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 0
            s.stacks[tid, 0] = 100
            s.stacks[tid, 1] = 100
            s.last_raise[tid] = 0
            legal_actions.write_legal_actions(s, tid, mask, min_r, max_r)

        wp.launch(k, dim=1, inputs=[s, mask, min_r, max_r])
        m = mask.numpy()[0]
        assert m[c.ACTION_FOLD] == True  # noqa: E712
        assert m[c.ACTION_CHECK] == True  # noqa: E712
        assert m[c.ACTION_CALL] == False  # noqa: E712
        assert m[c.ACTION_RAISE] == True  # noqa: E712
        assert min_r.numpy()[0] == c.BIG_BLIND
        assert max_r.numpy()[0] == 100

    def test_facing_bet_allows_call_and_raise(self):
        s = _alloc_state()
        mask = wp.zeros((1, c.NUM_ACTIONS), dtype=wp.bool, device=DEVICE)
        min_r = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        max_r = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        @wp.kernel
        def k(
            s: GameState,
            mask: wp.array(dtype=wp.bool, ndim=2),
            min_r: wp.array(dtype=wp.int32),
            max_r: wp.array(dtype=wp.int32),
        ):  # noqa: E501
            tid = wp.tid()
            s.active_player[tid] = 0
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 10
            s.stacks[tid, 0] = 100
            s.stacks[tid, 1] = 100
            s.last_raise[tid] = 10
            legal_actions.write_legal_actions(s, tid, mask, min_r, max_r)

        wp.launch(k, dim=1, inputs=[s, mask, min_r, max_r])
        m = mask.numpy()[0]
        assert m[c.ACTION_CHECK] == False  # noqa: E712
        assert m[c.ACTION_CALL] == True  # noqa: E712
        assert m[c.ACTION_RAISE] == True  # noqa: E712
        assert min_r.numpy()[0] == 10
        assert max_r.numpy()[0] == 90

    def test_opponent_too_short_disables_raise(self):
        s = _alloc_state()
        mask = wp.zeros((1, c.NUM_ACTIONS), dtype=wp.bool, device=DEVICE)
        min_r = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        max_r = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        @wp.kernel
        def k(
            s: GameState,
            mask: wp.array(dtype=wp.bool, ndim=2),
            min_r: wp.array(dtype=wp.int32),
            max_r: wp.array(dtype=wp.int32),
        ):  # noqa: E501
            tid = wp.tid()
            s.active_player[tid] = 0
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 10
            s.stacks[tid, 0] = 100
            s.stacks[tid, 1] = 5
            s.last_raise[tid] = 10
            legal_actions.write_legal_actions(s, tid, mask, min_r, max_r)

        wp.launch(k, dim=1, inputs=[s, mask, min_r, max_r])
        m = mask.numpy()[0]
        assert m[c.ACTION_CALL] == True  # noqa: E712
        assert m[c.ACTION_RAISE] == False  # noqa: E712
        assert max_r.numpy()[0] == 5
