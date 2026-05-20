"""
Unit tests for the Poker Game State Machine (state.py).

Tests the full lifecycle of a hand:
1. Initialization (Reset, Blinds, Shuffle)
2. Mechanics (Dealing streets, collecting pots)
3. Rules (Round completion, BB option)
4. Resolution (Showdown math, Payouts, Auto-reset)
"""

import os

import numpy as np
import pytest
import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import state
from gpu_poker.struct_types import GameState

# Initialize Warp
wp.init()

# Device selection
try:
    DEVICE = "cuda" if wp.is_device_available("cuda") else "cpu"
except RuntimeError:
    DEVICE = "cpu"


# Skip tests if running on CPU but needing CUDA-specific features (though state.py is generic)
# We generally want to test on the available device.
class TestState:
    @pytest.fixture(scope="class")
    def lookup_data(self):
        """Loads Cactus Kev lookup tables for showdown testing."""
        primes_np = np.array(c.RANK_PRIMES, dtype=np.int32)
        primes_wp = wp.from_numpy(primes_np, dtype=wp.int32, device=DEVICE)

        # Path relative to this test file
        data_path = os.path.join(
            os.path.dirname(__file__), "../src/gpu_poker/lookup_data/hand_ranks.npz"
        )

        if not os.path.exists(data_path):
            pytest.skip("Lookup data not found. Run generator first.")

        with np.load(data_path) as data:
            unsuited_np = data["unsuited_table"]
            flush_np = data["flush_table"]

        unsuited_wp = wp.from_numpy(unsuited_np, dtype=wp.int16, device=DEVICE)
        flush_wp = wp.from_numpy(flush_np, dtype=wp.int16, device=DEVICE)

        return {"primes": primes_wp, "unsuited": unsuited_wp, "flush": flush_wp}

    @pytest.fixture
    def state_struct(self):
        """Creates a single-environment GameState (SoA layout) on Device."""
        n_envs = 1

        # Allocate all backing arrays
        # Note: We use zeros to ensure clean state
        s = GameState()
        s.stage = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        s.button = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        s.active_player = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        s.done = wp.zeros(n_envs, dtype=wp.bool, device=DEVICE)

        s.deck = wp.zeros((n_envs, c.NUM_CARDS), dtype=wp.int32, device=DEVICE)
        s.deck_top = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)

        s.hole_cards = wp.zeros((n_envs, 2, 2), dtype=wp.int32, device=DEVICE)
        s.community_cards = wp.zeros((n_envs, 5), dtype=wp.int32, device=DEVICE)
        s.num_community = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)

        s.stacks = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
        s.bets = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
        s.initial_stacks = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
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

        # Seed RNG
        wp.copy(s.rng_state, wp.from_numpy(np.array([42], dtype=np.uint32), device=DEVICE))

        return s

    def test_reset_env(self, state_struct):
        """Tests initialization logic: Blinds, Cards, Pointers."""

        @wp.kernel
        def kernel_test_reset(s: GameState):
            tid = wp.tid()
            state.reset_env(s, tid, 1000, 10, 20)

        wp.launch(kernel_test_reset, dim=1, inputs=[state_struct])

        # 1. Check Blinds Posted
        # Heads-up: Button=SB(10), Opponent=BB(20)
        btn = state_struct.button.numpy()[0]
        sb_p = btn
        bb_p = 1 - btn

        bets = state_struct.bets.numpy()[0]
        stacks = state_struct.stacks.numpy()[0]

        assert bets[sb_p] == 10
        assert bets[bb_p] == 20
        assert stacks[sb_p] == 990
        assert stacks[bb_p] == 980

        # 2. Check Cards Dealt
        deck_top = state_struct.deck_top.numpy()[0]
        assert deck_top == 4  # 2 cards per player

        hole_cards = state_struct.hole_cards.numpy()[0]
        # Ensure valid cards (0-51)
        assert (hole_cards >= 0).all() and (hole_cards < 52).all()
        # Ensure no duplicates between players
        p0_cards = set(hole_cards[0])
        p1_cards = set(hole_cards[1])
        assert len(p0_cards.intersection(p1_cards)) == 0

        # 3. Check Stage
        assert state_struct.stage.numpy()[0] == c.STAGE_PREFLOP

    def test_betting_settled_logic(self, state_struct):
        """Tests is_betting_settled including the BB Option edge case."""

        @wp.kernel
        def kernel_test_settled(
            s: GameState,
            stage: wp.int32,
            bet0: wp.int32,
            bet1: wp.int32,
            active: wp.int32,
            is_bb_active: wp.bool,
            num_raises: wp.int32,
            out_result: wp.array(dtype=wp.bool),
        ):
            tid = wp.tid()
            s.stage[tid] = stage
            s.bets[tid, 0] = bet0
            s.bets[tid, 1] = bet1
            s.active_player[tid] = active
            s.num_raises[tid] = num_raises
            s.actions_this_street[tid] = 2
            s.stacks[tid, 0] = 1000
            s.stacks[tid, 1] = 1000

            # Setup button so we know who is BB
            # If is_bb_active is True, we set button opposite to active
            if is_bb_active:
                s.button[tid] = 1 - active
            else:
                s.button[tid] = active  # Active is Button (SB)

            out_result[tid] = state.is_betting_settled(s, tid)

        res = wp.zeros(1, dtype=wp.bool, device=DEVICE)

        # Case 1: Bets Unequal -> False
        wp.launch(
            kernel_test_settled,
            dim=1,
            inputs=[state_struct, c.STAGE_FLOP, 10, 20, 0, False, 0, res],
        )
        assert res.numpy()[0] == False  # noqa: E712

        # Case 2: Bets Equal, Postflop -> True
        wp.launch(
            kernel_test_settled,
            dim=1,
            inputs=[state_struct, c.STAGE_FLOP, 20, 20, 0, False, 1, res],
        )
        assert res.numpy()[0] == True  # noqa: E712

        # Case 3: Preflop, Bets Equal (Limped Pot), BB to act -> False (Option)
        # num_raises = 0 for limped pot.
        wp.launch(
            kernel_test_settled,
            dim=1,
            inputs=[state_struct, c.STAGE_PREFLOP, 20, 20, 0, True, 0, res],
        )
        assert res.numpy()[0] == False  # noqa: E712

        # Case 4: Preflop, Bets Equal (Raised Pot), BB to act -> True (Call ends action)
        # num_raises > 0 for raised pot.
        wp.launch(
            kernel_test_settled,
            dim=1,
            inputs=[state_struct, c.STAGE_PREFLOP, 40, 40, 0, True, 1, res],
        )
        assert res.numpy()[0] == True  # noqa: E712

    def test_deal_street(self, state_struct):
        """Tests dealing community cards."""

        @wp.kernel
        def kernel_test_deal(s: GameState, stage: wp.int32, start_deck_top: wp.int32):
            tid = wp.tid()
            s.stage[tid] = stage
            # Ensure deck is initialized
            for i in range(c.NUM_CARDS):
                s.deck[tid, i] = i

            # Set starting pointers
            s.deck_top[tid] = start_deck_top
            s.num_community[tid] = 0  # This resets community count, logic below handles offset

            # For this test we manually set num_community based on stage to match expectation
            if stage == c.STAGE_TURN:
                s.num_community[tid] = 3

            state.deal_street(s, tid)

        # 1. Flop (3 cards). Starts at 4 (after holes).
        wp.launch(kernel_test_deal, dim=1, inputs=[state_struct, c.STAGE_FLOP, 4])
        assert state_struct.num_community.numpy()[0] == 3
        assert state_struct.deck_top.numpy()[0] == 7  # 4 + 3

        # 2. Turn (1 card). Starts at 7.
        wp.launch(kernel_test_deal, dim=1, inputs=[state_struct, c.STAGE_TURN, 7])

        assert state_struct.num_community.numpy()[0] == 4
        assert state_struct.deck_top.numpy()[0] == 8

    def test_showdown_payout(self, state_struct, lookup_data):
        """Tests reward calculation at showdown."""

        @wp.kernel
        def kernel_test_showdown(
            s: GameState,
            primes: wp.array(dtype=wp.int32),
            unsuited: wp.array(dtype=wp.int16),
            flush: wp.array(dtype=wp.int16),
            out_reward: wp.array(dtype=wp.float32),
        ):
            tid = wp.tid()
            # Setup Pot
            s.pot[tid] = 200
            s.stacks[tid, 0] = 900  # Invested 100
            s.stacks[tid, 1] = 900  # Invested 100
            s.initial_stacks[tid, 0] = 1000
            s.initial_stacks[tid, 1] = 1000

            # Setup Winner P0 (Aces) vs P1 (Kings)
            # Board: 2, 3, 4, 5, 9 (Rainbow)
            # P0: AA
            # P1: KK

            # Helper to set cards... skipping precise card setting for brevity
            # and trusting the evaluator test.
            # We will force the result by mocking the resolve function logic?
            # No, we can't mock inside a kernel.
            # We must set actual cards that result in a win.

            # P0: Ace Spades (51), Ace Hearts (38)
            s.hole_cards[tid, 0, 0] = 51
            s.hole_cards[tid, 0, 1] = 38

            # P1: King Spades (50), King Hearts (37)
            s.hole_cards[tid, 1, 0] = 50
            s.hole_cards[tid, 1, 1] = 37

            # Board: 2c, 3c, 4d, 5d, 9s (0, 1, 15, 16, 46)
            s.community_cards[tid, 0] = 0
            s.community_cards[tid, 1] = 1
            s.community_cards[tid, 2] = 15
            s.community_cards[tid, 3] = 16
            s.community_cards[tid, 4] = 46

            out_reward[tid] = state.resolve_showdown(s, tid, primes, unsuited, flush)

        res = wp.zeros(1, dtype=wp.float32, device=DEVICE)
        wp.launch(
            kernel_test_showdown,
            dim=1,
            inputs=[
                state_struct,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
                res,
            ],
        )

        # P0 wins.
        # Reward = (FinalStack - Initial).
        # FinalStack = 900 + 200 = 1100.
        # Initial = 1000.
        # Reward = 100.
        assert res.numpy()[0] == 100.0

    def test_step_fold(self, state_struct, lookup_data):
        """Tests folding logic."""

        @wp.kernel
        def kernel_test_fold(
            s: GameState,
            primes: wp.array(dtype=wp.int32),
            unsuited: wp.array(dtype=wp.int16),
            flush: wp.array(dtype=wp.int16),
            out_reward: wp.array(dtype=wp.float32),
        ):
            tid = wp.tid()

            # FORCE BUTTON: Set button to 1 so reset_env rotates it to 0.
            # This ensures P0 becomes the active player (SB) after reset.
            s.button[tid] = 1

            # Reset first to get clean state
            state.reset_env(s, tid, 1000, 10, 20)

            # Player 0 (SB) Folds immediately
            # Current Pot: 30 (10 SB + 20 BB)
            # P0 invested 10. P1 invested 20.

            out_reward[tid] = state.step(
                s, tid, c.ACTION_FOLD, 0, primes, unsuited, flush, 1000, 10, 20
            )

        res = wp.zeros(1, dtype=wp.float32, device=DEVICE)
        wp.launch(
            kernel_test_fold,
            dim=1,
            inputs=[
                state_struct,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
                res,
            ],
        )

        # P0 folded. P0 invested 10.
        # Reward should be -10.
        assert res.numpy()[0] == -10.0

        # Check auto-reset happened (stage should be Preflop, pot 0)
        assert state_struct.stage.numpy()[0] == c.STAGE_PREFLOP
        assert state_struct.pot.numpy()[0] == 0

    def test_all_in_runout(self, state_struct, lookup_data):
        """Tests that All-In correctly fast-forwards to showdown."""

        @wp.kernel
        def kernel_test_allin(
            s: GameState,
            primes: wp.array(dtype=wp.int32),
            unsuited: wp.array(dtype=wp.int16),
            flush: wp.array(dtype=wp.int16),
            out_reward: wp.array(dtype=wp.float32),
        ):
            tid = wp.tid()
            state.reset_env(s, tid, 1000, 10, 20)

            # Force P0 to hold Aces (Guarantee win against random garbage usually)
            s.hole_cards[tid, 0, 0] = 51  # As
            s.hole_cards[tid, 0, 1] = 38  # Ah
            # Force P1 to hold 2/3 offsuit
            s.hole_cards[tid, 1, 0] = 0
            s.hole_cards[tid, 1, 1] = 14

            # P0 is active (SB). P0 goes All-In (Raise 990).
            # We call step twice.
            # 1. P0 All-in
            state.step(s, tid, c.ACTION_RAISE, 990, primes, unsuited, flush, 1000, 10, 20)

            # 2. P1 Calls (P1 is now active)
            # P1 has 980 stack. Needs to call 980 more to match P0.
            # Call amount is calculated inside step.
            out_reward[tid] = state.step(
                s, tid, c.ACTION_CALL, 0, primes, unsuited, flush, 1000, 10, 20
            )

        res = wp.zeros(1, dtype=wp.float32, device=DEVICE)
        wp.launch(
            kernel_test_allin,
            dim=1,
            inputs=[
                state_struct,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
                res,
            ],
        )

        # P0 wins entire stack of P1.
        # P0 Reward = +1000.
        assert res.numpy()[0] == 1000.0

        # Ensure done flag was set (though it gets reset immediately)
        # Ideally we'd check if `resolve_showdown` was called. The positive reward proves it.
