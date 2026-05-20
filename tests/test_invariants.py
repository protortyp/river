"""
Invariant-focused tests for the GPU poker kernels.

These tests are designed to catch subtle correctness issues that can still pass
basic flow tests (chip conservation, uncalled bets, split pots, etc.).
"""

import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import observations, state
from gpu_poker.struct_types import GameState

wp.init()

try:
    DEVICE = "cuda" if wp.is_device_available("cuda") else "cpu"
except RuntimeError:
    DEVICE = "cpu"


def _alloc_state(n_envs: int) -> GameState:
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


class TestInvariants:
    def test_chip_conservation_reset_and_simple_round(self):
        """
        Chips should be conserved:
          stacks0 + stacks1 + pot + bets0 + bets1 == 2 * starting_stack
        """
        s = _alloc_state(1)
        starting_stack = 1000
        sb = 1
        bb = 2

        primes = wp.zeros(13, dtype=wp.int32, device=DEVICE)
        unsuited = wp.zeros(1, dtype=wp.int16, device=DEVICE)
        flush = wp.zeros(1, dtype=wp.int16, device=DEVICE)

        out_total0 = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_total1 = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        @wp.kernel
        def kernel(
            s: GameState,
            primes: wp.array(dtype=wp.int32),
            unsuited: wp.array(dtype=wp.int16),
            flush: wp.array(dtype=wp.int16),
            out_total0: wp.array(dtype=wp.int32),
            out_total1: wp.array(dtype=wp.int32),
        ):
            tid = wp.tid()
            state.reset_env(s, tid, starting_stack, sb, bb)

            total0 = (
                s.stacks[tid, 0] + s.stacks[tid, 1] + s.pot[tid] + s.bets[tid, 0] + s.bets[tid, 1]
            )

            # SB calls, BB checks -> advance to flop
            state.step(s, tid, c.ACTION_CALL, 0, primes, unsuited, flush, starting_stack, sb, bb)
            state.step(s, tid, c.ACTION_CHECK, 0, primes, unsuited, flush, starting_stack, sb, bb)

            total1 = (
                s.stacks[tid, 0] + s.stacks[tid, 1] + s.pot[tid] + s.bets[tid, 0] + s.bets[tid, 1]
            )

            out_total0[tid] = total0
            out_total1[tid] = total1

        wp.launch(kernel, dim=1, inputs=[s, primes, unsuited, flush, out_total0, out_total1])

        assert out_total0.numpy()[0] == 2 * starting_stack
        assert out_total1.numpy()[0] == 2 * starting_stack

    def test_collect_pot_returns_uncalled_bet(self):
        s = _alloc_state(1)

        out_pot = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_stack0 = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        out_stack1 = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        @wp.kernel
        def kernel(
            s: GameState,
            out_pot: wp.array(dtype=wp.int32),
            out_s0: wp.array(dtype=wp.int32),
            out_s1: wp.array(dtype=wp.int32),
        ):  # noqa: E501
            tid = wp.tid()
            # Synthetic situation: P0 has bet 100, P1 has bet 60. The extra 40 is uncalled.
            s.stacks[tid, 0] = 900
            s.stacks[tid, 1] = 940
            s.bets[tid, 0] = 100
            s.bets[tid, 1] = 60
            s.pot[tid] = 10

            state.collect_pot(s, tid)

            out_pot[tid] = s.pot[tid]
            out_s0[tid] = s.stacks[tid, 0]
            out_s1[tid] = s.stacks[tid, 1]

        wp.launch(kernel, dim=1, inputs=[s, out_pot, out_stack0, out_stack1])

        # Pot collects only the matched amount: 60 + 60 = 120 (plus existing 10)
        assert out_pot.numpy()[0] == 130
        # P0 gets the uncalled 40 back: 900 -> 940
        assert out_stack0.numpy()[0] == 940
        assert out_stack1.numpy()[0] == 940

    def test_split_pot_reward_even(self):
        s = _alloc_state(1)
        out_reward = wp.zeros(1, dtype=wp.float32, device=DEVICE)

        @wp.kernel
        def kernel(s: GameState, out_reward: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            s.pot[tid] = 4
            s.stacks[tid, 0] = 998
            s.stacks[tid, 1] = 998
            s.initial_stacks[tid, 0] = 1000
            s.initial_stacks[tid, 1] = 1000
            s.button[tid] = 0
            out_reward[tid] = state.calculate_rewards(s, tid, c.INVALID_PLAYER)

        wp.launch(kernel, dim=1, inputs=[s, out_reward])
        assert out_reward.numpy()[0] == 0.0

    def test_split_pot_reward_odd_chip_button(self):
        s = _alloc_state(1)
        out_reward_btn0 = wp.zeros(1, dtype=wp.float32, device=DEVICE)
        out_reward_btn1 = wp.zeros(1, dtype=wp.float32, device=DEVICE)

        @wp.kernel
        def kernel(s: GameState, button: wp.int32, out_reward: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            s.pot[tid] = 5
            s.stacks[tid, 0] = 998
            s.stacks[tid, 1] = 998
            s.initial_stacks[tid, 0] = 1000
            s.initial_stacks[tid, 1] = 1000
            s.button[tid] = button
            out_reward[tid] = state.calculate_rewards(s, tid, c.INVALID_PLAYER)

        wp.launch(kernel, dim=1, inputs=[s, 0, out_reward_btn0])
        wp.launch(kernel, dim=1, inputs=[s, 1, out_reward_btn1])

        assert out_reward_btn0.numpy()[0] == 1.0
        assert out_reward_btn1.numpy()[0] == 0.0

    def test_observation_to_call_stack_zero_safe(self):
        s = _alloc_state(1)
        obs_cards = wp.zeros((1, 7), dtype=wp.int32, device=DEVICE)
        obs_scalars = wp.zeros((1, 17), dtype=wp.float32, device=DEVICE)

        primes = wp.zeros(13, dtype=wp.int32, device=DEVICE)
        unsuited = wp.zeros(1, dtype=wp.int16, device=DEVICE)
        flush = wp.zeros(1, dtype=wp.int16, device=DEVICE)

        @wp.kernel
        def kernel(
            s: GameState,
            obs_cards: wp.array(dtype=wp.int32, ndim=2),
            obs_scalars: wp.array(dtype=wp.float32, ndim=2),
            primes: wp.array(dtype=wp.int32),
            unsuited: wp.array(dtype=wp.int16),
            flush: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            s.active_player[tid] = 0
            s.stage[tid] = c.STAGE_PREFLOP
            s.stacks[tid, 0] = 0
            s.stacks[tid, 1] = 100
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 10
            observations.write_observation(s, tid, obs_cards, obs_scalars, primes, unsuited, flush)

        wp.launch(kernel, dim=1, inputs=[s, obs_cards, obs_scalars, primes, unsuited, flush])
        assert obs_scalars.numpy()[0, 6] == 0.0

    def test_observation_to_call_clipped_when_facing_overbet(self):
        """Regression: obs_scalars[6] = to_call / stack was unclipped, so when
        the actor faces a bet larger than their remaining stack (i.e., must
        call all-in for less), the feature exceeded 1.0 — violating the
        normalization contract documented in AGENTS.md. The fix clamps the
        numerator at the player's stack so the ratio stays in [0, 1]."""
        s = _alloc_state(1)
        obs_cards = wp.zeros((1, 7), dtype=wp.int32, device=DEVICE)
        obs_scalars = wp.zeros((1, 17), dtype=wp.float32, device=DEVICE)

        primes = wp.zeros(13, dtype=wp.int32, device=DEVICE)
        unsuited = wp.zeros(1, dtype=wp.int16, device=DEVICE)
        flush = wp.zeros(1, dtype=wp.int16, device=DEVICE)

        @wp.kernel
        def kernel(
            s: GameState,
            obs_cards: wp.array(dtype=wp.int32, ndim=2),
            obs_scalars: wp.array(dtype=wp.float32, ndim=2),
            primes: wp.array(dtype=wp.int32),
            unsuited: wp.array(dtype=wp.int16),
            flush: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            s.active_player[tid] = 0
            s.stage[tid] = c.STAGE_PREFLOP
            # P0 has 50 in stack, P1 has bet 200 (overbet beyond P0's stack).
            # to_call = 200, but P0 can only commit 50.
            s.stacks[tid, 0] = 50
            s.stacks[tid, 1] = 1000
            s.bets[tid, 0] = 0
            s.bets[tid, 1] = 200
            observations.write_observation(s, tid, obs_cards, obs_scalars, primes, unsuited, flush)

        wp.launch(kernel, dim=1, inputs=[s, obs_cards, obs_scalars, primes, unsuited, flush])
        val = float(obs_scalars.numpy()[0, 6])
        # Before the fix: 200/50 = 4.0. After: clamped to 50/50 = 1.0.
        assert val == 1.0, f"to_call/stack should clip to 1.0, got {val}"
