"""
Unit tests for poker action logic.

Tests the Game Mechanics layer:
- Chip calculations (get_to_call, execute_bet)
- Action validation (rules for Check/Call/Raise)
- State mutation (apply_action)
"""

import pytest
import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import actions
from gpu_poker.struct_types import GameState

wp.init()

try:
    DEVICE = "cuda" if wp.is_device_available("cuda") else "cpu"
except RuntimeError:
    DEVICE = "cpu"


class TestActions:
    @pytest.fixture
    def state(self):
        """Creates a GameState struct with properly allocated SoA arrays for 1 environment."""
        n_envs = 1

        # Allocate backing arrays (SoA layout)
        stacks = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
        bets = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
        initial_stacks = wp.zeros((n_envs, 2), dtype=wp.int32, device=DEVICE)
        pot = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        last_raise = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        num_raises = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        last_aggressor = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        done = wp.zeros(n_envs, dtype=wp.bool, device=DEVICE)

        # Dummy allocations for other required fields
        stage = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        button = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        active_player = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        deck = wp.zeros((n_envs, 52), dtype=wp.int32, device=DEVICE)
        deck_top = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        hole_cards = wp.zeros((n_envs, 2, 2), dtype=wp.int32, device=DEVICE)
        community_cards = wp.zeros((n_envs, 5), dtype=wp.int32, device=DEVICE)
        num_community = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        num_actions = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        actions_this_street = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        last_action_type = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        last_action_amount = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        last_action_was_raise = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        rng_state = wp.zeros(n_envs, dtype=wp.uint32, device=DEVICE)
        episode_id = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        cfg_starting_stack = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        cfg_small_blind = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)
        cfg_big_blind = wp.zeros(n_envs, dtype=wp.int32, device=DEVICE)

        # Create GameState struct instance
        state = GameState()
        state.stacks = stacks
        state.bets = bets
        state.initial_stacks = initial_stacks
        state.pot = pot
        state.last_raise = last_raise
        state.num_raises = num_raises
        state.last_aggressor = last_aggressor
        state.done = done
        state.stage = stage
        state.button = button
        state.active_player = active_player
        state.deck = deck
        state.deck_top = deck_top
        state.hole_cards = hole_cards
        state.community_cards = community_cards
        state.num_community = num_community
        state.num_actions = num_actions
        state.actions_this_street = actions_this_street
        state.last_action_type = last_action_type
        state.last_action_amount = last_action_amount
        state.last_action_was_raise = last_action_was_raise
        state.rng_state = rng_state
        state.episode_id = episode_id
        state.cfg_starting_stack = cfg_starting_stack
        state.cfg_small_blind = cfg_small_blind
        state.cfg_big_blind = cfg_big_blind

        return state

    def test_betting_math(self, state):
        """
        Tests get_to_call and execute_bet logic.
        """

        @wp.kernel
        def test_math_kernel(
            s: GameState,
            p0_bet: wp.int32,
            p1_bet: wp.int32,
            p0_stack: wp.int32,
            bet_amount: wp.int32,
            out_to_call: wp.array(dtype=wp.int32),
            out_actual_bet: wp.array(dtype=wp.int32),
            out_final_stack: wp.array(dtype=wp.int32),
            out_final_pot: wp.array(dtype=wp.int32),
        ):
            tid = wp.tid()
            # Setup scenario in global arrays
            s.bets[tid, 0] = p0_bet
            s.bets[tid, 1] = p1_bet
            s.stacks[tid, 0] = p0_stack
            s.pot[tid] = p0_bet + p1_bet

            # 1. Test get_to_call
            out_to_call[tid] = actions.get_to_call(s, tid, 0)

            # 2. Test execute_bet (mutates state)
            actual = actions.execute_bet(s, tid, 0, bet_amount)
            out_actual_bet[tid] = actual

            # Capture results
            out_final_stack[tid] = s.stacks[tid, 0]
            out_final_pot[tid] = s.pot[tid]

        # Scenario: P1 bets 50, P0 bets 10. P0 needs to call 40.
        # P0 attempts to bet 100 (more than stack of 80).
        p0_bet, p1_bet = 10, 50
        p0_stack = 80
        bet_attempt = 100

        # Allocations
        out_to_call = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_actual = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_stack = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_pot = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(
            kernel=test_math_kernel,
            dim=1,
            inputs=[
                state,
                p0_bet,
                p1_bet,
                p0_stack,
                bet_attempt,
                out_to_call,
                out_actual,
                out_stack,
                out_pot,
            ],
        )

        # Verification
        # 1. To Call: 50 - 10 = 40
        assert out_to_call.numpy()[0] == 40

        # 2. Actual Bet: Capped at stack (80)
        assert out_actual.numpy()[0] == 80

        # 3. Final Stack: 80 - 80 = 0
        assert out_stack.numpy()[0] == 0

        # 4. Final Pot: Should remain 60 (10+50)
        # execute_bet only updates bets[], not pot.
        # Pot is updated via collect_pot in state.py
        assert out_pot.numpy()[0] == 60

    def test_validation_basics(self, state):
        """
        Tests basic Fold, Check, Call validation.
        """

        @wp.kernel
        def test_basic_valid(
            s: GameState,
            action_type: wp.int32,
            p0_bet: wp.int32,
            p1_bet: wp.int32,
            out_valid: wp.array(dtype=wp.bool),
        ):
            tid = wp.tid()
            s.bets[tid, 0] = p0_bet
            s.bets[tid, 1] = p1_bet
            # Amount 0 for non-raise actions
            out_valid[tid] = actions.validate_action(s, tid, 0, action_type, 0)

        out_valid = wp.zeros(1, dtype=wp.bool, device=DEVICE)

        # 1. Test FOLD (Always valid)
        wp.launch(test_basic_valid, dim=1, inputs=[state, c.ACTION_FOLD, 0, 100, out_valid])
        assert out_valid.numpy()[0] == True  # noqa: E712

        # 2. Test CHECK (Valid if bets equal)
        wp.launch(test_basic_valid, dim=1, inputs=[state, c.ACTION_CHECK, 50, 50, out_valid])
        assert out_valid.numpy()[0] == True  # noqa: E712

        # 3. Test CHECK (Invalid if facing bet)
        wp.launch(test_basic_valid, dim=1, inputs=[state, c.ACTION_CHECK, 50, 100, out_valid])
        assert out_valid.numpy()[0] == False  # noqa: E712

        # 4. Test CALL (Valid if facing bet)
        wp.launch(test_basic_valid, dim=1, inputs=[state, c.ACTION_CALL, 50, 100, out_valid])
        assert out_valid.numpy()[0] == True  # noqa: E712

        # 5. Test CALL (Invalid/Check if bets equal - Strict Rules)
        wp.launch(test_basic_valid, dim=1, inputs=[state, c.ACTION_CALL, 50, 50, out_valid])
        assert out_valid.numpy()[0] == False  # noqa: E712

    def test_validation_raise(self, state):
        """
        Tests Raise validation logic including min-raises and stack sizes.
        """

        @wp.kernel
        def test_raise_valid(
            s: GameState,
            p0_bet: wp.int32,
            p1_bet: wp.int32,
            p0_stack: wp.int32,
            p1_stack: wp.int32,
            last_raise: wp.int32,
            raise_amt: wp.int32,
            out_valid: wp.array(dtype=wp.bool),
        ):
            tid = wp.tid()
            s.bets[tid, 0] = p0_bet
            s.bets[tid, 1] = p1_bet
            s.stacks[tid, 0] = p0_stack
            s.stacks[tid, 1] = p1_stack
            s.last_raise[tid] = last_raise

            out_valid[tid] = actions.validate_action(s, tid, 0, c.ACTION_RAISE, raise_amt)

        out_valid = wp.zeros(1, dtype=wp.bool, device=DEVICE)

        # 1. Valid Raise
        wp.launch(test_raise_valid, dim=1, inputs=[state, 10, 10, 1000, 1000, 10, 20, out_valid])
        assert out_valid.numpy()[0] == True  # noqa: E712

        # 2. Invalid Raise (Too small)
        wp.launch(test_raise_valid, dim=1, inputs=[state, 10, 10, 1000, 1000, 10, 5, out_valid])
        assert out_valid.numpy()[0] == False  # noqa: E712

        # 3. Invalid Raise (Not enough stack)
        wp.launch(test_raise_valid, dim=1, inputs=[state, 10, 10, 15, 1000, 10, 20, out_valid])
        assert out_valid.numpy()[0] == False  # noqa: E712

        # 4. Valid All-In Raise (Exception)
        wp.launch(test_raise_valid, dim=1, inputs=[state, 10, 10, 15, 1000, 20, 15, out_valid])
        assert out_valid.numpy()[0] == True  # noqa: E712

        # 5. Invalid: all-in-for-less that exceeds the opponent's effective stack.
        # p0 bet=0, p1 bet=10 → to_call=10. p0 stack=15. p1 stack=4 (very short).
        # last_raise=10 so min_raise=10, but max_raise (opp can match) = 4.
        # `amount = stack - to_call = 5` was previously accepted by validate_action
        # because it matched the all-in-for-less rule, even though it exceeded
        # the opponent's effective stack cap. The mask-vs-apply contract was
        # broken: the mask said max=4, validate said 5 OK. Now both reject.
        wp.launch(test_raise_valid, dim=1, inputs=[state, 0, 10, 15, 4, 10, 5, out_valid])
        assert out_valid.numpy()[0] == False  # noqa: E712

    def test_short_all_in_does_not_update_last_raise(self, state):
        """
        Standard NLHE: an all-in raise of less than a full raise size does NOT
        reset `last_raise` (it does not reopen action for players already past).
        Only full raises (delta >= prior min) update the reference.
        """

        @wp.kernel
        def k(
            s: GameState,
            out_last_raise_before: wp.array(dtype=wp.int32),
            out_last_raise_after: wp.array(dtype=wp.int32),
            out_actual_raise: wp.array(dtype=wp.int32),
        ):
            tid = wp.tid()
            # Set up: p0 raised by 50; last_raise = 50. p1 now wants to all-in
            # for less than 50 (only 30 in stack + to_call already in bets).
            # We model it directly via apply_action: p1 facing to_call=50 with
            # only 30 remaining stack, RAISE amount=0 would be a short all-in,
            # but apply_action expects amount as the raise delta — for the
            # short-stack case we pass the only legal delta (stack - to_call).
            s.bets[tid, 0] = 50
            s.bets[tid, 1] = 0
            s.stacks[tid, 0] = 950
            s.stacks[tid, 1] = 30  # less than the prior 50 raise + to_call
            s.last_raise[tid] = 50
            s.cfg_big_blind[tid] = 2
            s.last_aggressor[tid] = 0
            s.num_raises[tid] = 1

            out_last_raise_before[tid] = s.last_raise[tid]

            # p1's only legal raise size is stack-to_call but to_call = 50 > stack=30.
            # So p1 can call all-in (CALL action), but a RAISE that goes all-in for
            # less requires we set the state so to_call < stack. Reframe: p0 raised
            # earlier to 50, but to keep this a RAISE path we need bets that allow
            # a raise. Re-setup as: p0 bet 10 (initial raise), to_call=10 for p1,
            # last_raise=10. p1 stack=25, p1 raise delta=15 (= stack-to_call=15)
            # which exceeds prior last_raise=10. NOT a short all-in.
            # We want short: set last_raise=20 (someone raised by 20), p0 bet 20,
            # to_call=20, p1 stack=25 -> stack-to_call=5, which is < min_raise=20.
            s.bets[tid, 0] = 20
            s.bets[tid, 1] = 0
            s.stacks[tid, 0] = 1000
            s.stacks[tid, 1] = 25
            s.last_raise[tid] = 20
            s.last_aggressor[tid] = 0
            s.num_raises[tid] = 1

            # All-in raise of 5 (less than min-raise 20).
            actions.apply_action(s, tid, 1, c.ACTION_RAISE, 5)

            out_last_raise_after[tid] = s.last_raise[tid]
            # Re-derive actual_raise that apply_action would have computed:
            # opp_max_total = 20+1000 = 1020, player_max_total = 0+25 = 25.
            # desired_total clamped to 25. total_bet = 25, actual_raise = 25-20=5.
            out_actual_raise[tid] = 5

        out_before = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_after = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_actual = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        wp.launch(k, dim=1, inputs=[state, out_before, out_after, out_actual])

        # apply_action moved a chip, but because actual_raise (5) < prior
        # min-raise threshold (20), the engine must NOT update last_raise.
        # Previously: last_raise would have been overwritten to 5.
        assert out_after.numpy()[0] == 20, (
            f"last_raise should stay at 20 after all-in-for-less; got {out_after.numpy()[0]}"
        )

    def test_apply_action_state_mutation(self, state):
        """
        Tests that apply_action correctly updates game state variables.
        """

        @wp.kernel
        def test_apply(
            s: GameState,
            out_aggressor: wp.array(dtype=wp.int32),
            out_last_raise: wp.array(dtype=wp.int32),
            out_num_raises: wp.array(dtype=wp.int32),
            out_done: wp.array(dtype=wp.bool),
        ):
            tid = wp.tid()
            # Reset state for test
            s.last_aggressor[tid] = -1
            s.num_raises[tid] = 0
            s.last_raise[tid] = 10
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 0
            s.stacks[tid, 0] = 1000
            s.stacks[tid, 1] = 1000

            # Action: Raise 50
            actions.apply_action(s, tid, 0, c.ACTION_RAISE, 50)

            out_aggressor[0] = s.last_aggressor[tid]
            out_last_raise[0] = s.last_raise[tid]
            out_num_raises[0] = s.num_raises[tid]

            # Action: Fold
            s.done[tid] = False
            actions.apply_action(s, tid, 0, c.ACTION_FOLD, 0)
            out_done[0] = s.done[tid]

        # Outputs
        res_agg = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        res_raise = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        res_num = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        res_done = wp.zeros(1, dtype=wp.bool, device=DEVICE)

        wp.launch(test_apply, dim=1, inputs=[state, res_agg, res_raise, res_num, res_done])

        # Verify Raise Updates
        assert res_agg.numpy()[0] == 0
        assert res_raise.numpy()[0] == 50
        assert res_num.numpy()[0] == 1

        # Verify Fold Updates
        assert res_done.numpy()[0] == True  # noqa: E712

    def test_apply_action_raise_effective_stack_cap_total(self, state):
        """
        Ensures raise capping happens on the resulting total bet, not the action amount.
        """

        @wp.kernel
        def k(
            s: GameState,
            out_bet0: wp.array(dtype=wp.int32),
            out_bet1: wp.array(dtype=wp.int32),
        ):
            tid = wp.tid()
            # Opponent has 20 in already and 100 behind => max total 120.
            s.bets[tid, 1] = 20
            s.stacks[tid, 1] = 100

            # We already have 10 in and 100 behind.
            s.bets[tid, 0] = 10
            s.stacks[tid, 0] = 200

            # Try an absurd raise delta; the resulting total should cap to 120,
            # so our final bet should be 120 (not 130).
            actions.apply_action(s, tid, 0, c.ACTION_RAISE, 1000)

            out_bet0[tid] = s.bets[tid, 0]
            out_bet1[tid] = s.bets[tid, 1]

        out_bet0 = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_bet1 = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        wp.launch(k, dim=1, inputs=[state, out_bet0, out_bet1])

        assert out_bet0.numpy()[0] == 120
        assert out_bet1.numpy()[0] == 20
