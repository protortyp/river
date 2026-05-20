"""
Unit tests for observation logic (kernels/observations.py).
"""

import os

import numpy as np
import pytest
import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import observations
from gpu_poker.struct_types import GameState

wp.init()

try:
    DEVICE = "cuda" if wp.is_device_available("cuda") else "cpu"
except RuntimeError:
    DEVICE = "cpu"


class TestObservations:
    @pytest.fixture(scope="class")
    def lookup_data(self):
        """Loads Cactus Kev lookup tables."""
        primes_np = np.array(c.RANK_PRIMES, dtype=np.int32)
        primes_wp = wp.from_numpy(primes_np, dtype=wp.int32, device=DEVICE)

        data_path = os.path.join(
            os.path.dirname(__file__), "../src/gpu_poker/lookup_data/hand_ranks.npz"
        )

        if not os.path.exists(data_path):
            pytest.skip("Lookup data not found.")

        with np.load(data_path) as data:
            unsuited_np = data["unsuited_table"]
            flush_np = data["flush_table"]

        unsuited_wp = wp.from_numpy(unsuited_np, dtype=wp.int16, device=DEVICE)
        flush_wp = wp.from_numpy(flush_np, dtype=wp.int16, device=DEVICE)

        return {"primes": primes_wp, "unsuited": unsuited_wp, "flush": flush_wp}

    @pytest.fixture
    def state(self):
        """Creates a single-environment GameState."""
        n_envs = 1
        s = GameState()

        # Allocate required arrays
        s.stage = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        s.active_player = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        s.hole_cards = wp.zeros((n_envs, 2, 2), dtype=wp.int32, device=DEVICE)
        s.community_cards = wp.zeros((n_envs, 5), dtype=wp.int32, device=DEVICE)

        s.stacks = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
        s.bets = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
        s.pot = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)

        # Dummy allocs for safety
        s.button = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        s.done = wp.zeros(n_envs, dtype=wp.bool, device=DEVICE)
        s.deck = wp.zeros((n_envs, 52), dtype=wp.int32, device=DEVICE)
        s.deck_top = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        s.num_community = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        s.initial_stacks = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
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

    def test_card_visibility(self, state):
        """Tests that future cards are masked with -1."""

        @wp.kernel
        def test_vis_kernel(s: GameState, stage: wp.int32, out_cards: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            s.stage[tid] = stage
            s.active_player[tid] = 0

            # Setup dummy cards
            # P0: 10, 11. Board: 20, 21, 22, 23, 24
            s.hole_cards[tid, 0, 0] = 10
            s.hole_cards[tid, 0, 1] = 11
            s.community_cards[tid, 0] = 20
            s.community_cards[tid, 1] = 21
            s.community_cards[tid, 2] = 22
            s.community_cards[tid, 3] = 23
            s.community_cards[tid, 4] = 24

            # Read all 7 slots
            for i in range(7):
                out_cards[i] = observations.get_visible_card(s, tid, i)

        out = wp.zeros(7, dtype=wp.int32, device=DEVICE)

        # 1. Preflop: Only hole cards visible
        wp.launch(test_vis_kernel, dim=1, inputs=[state, c.STAGE_PREFLOP, out])
        res = out.numpy()
        assert res[0] == 10 and res[1] == 11
        assert (res[2:] == c.INVALID_CARD).all()

        # 2. Flop: Hole + 3 board visible
        wp.launch(test_vis_kernel, dim=1, inputs=[state, c.STAGE_FLOP, out])
        res = out.numpy()
        assert (res[0:5] != c.INVALID_CARD).all()
        assert res[5] == c.INVALID_CARD  # Turn
        assert res[6] == c.INVALID_CARD  # River

        # 3. River: All visible
        wp.launch(test_vis_kernel, dim=1, inputs=[state, c.STAGE_RIVER, out])
        res = out.numpy()
        assert (res != c.INVALID_CARD).all()
