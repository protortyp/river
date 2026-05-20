"""
Unit tests for the GPU Hand Evaluator.

This tests the logic in `src/gpu_poker/kernels/evaluator.py` by:
1. Loading the pre-calculated Cactus Kev lookup tables into GPU memory.
2. Running kernels that evaluate specific known hands (Royal Flush, Quads,
    etc.).
3. Asserting that the scores follow the correct poker hierarchy.
"""

import os

import numpy as np
import pytest
import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import evaluator

# Ensure Warp is initialized
wp.init()

try:
    HAS_CUDA = wp.is_device_available("cuda")
except RuntimeError:
    HAS_CUDA = False


def card(rank_str: str, suit_str: str) -> int:
    """Helper to create card integers for readable tests.
    Ranks: 2,3,4,5,6,7,8,9,T,J,Q,K,A
    Suits: c,d,h,s
    """
    ranks = "23456789TJQKA"
    suits = "cdhs"

    r = ranks.index(rank_str)
    s = suits.index(suit_str)
    return r + (s * 13)


@pytest.mark.skipif(not HAS_CUDA, reason="Warp CUDA device not available")
class TestEvaluator:
    @pytest.fixture(scope="class")
    def lookup_data(self):
        """
        Loads the lookup tables and primes array into Warp GPU arrays.
        This runs once per test session to avoid reloading 100MB repeatedly.
        """
        # 1. Load Primes
        primes_np = np.array(c.RANK_PRIMES, dtype=np.int32)
        primes_wp = wp.from_numpy(primes_np, dtype=wp.int32, device="cuda")

        # 2. Load Lookup Tables
        # Assuming the generator has run and placed the file in the source tree
        data_path = os.path.join(
            os.path.dirname(__file__), "../src/gpu_poker/lookup_data/hand_ranks.npz"
        )

        if not os.path.exists(data_path):
            pytest.fail(
                f"Lookup data not found at {data_path}. "
                "Run `python -m src.gpu_poker.lookup_data.generator` first."
            )

        with np.load(data_path) as data:
            unsuited_np = data["unsuited_table"]
            flush_np = data["flush_table"]

        # Move to GPU
        unsuited_wp = wp.from_numpy(unsuited_np, dtype=wp.int16, device="cuda")
        flush_wp = wp.from_numpy(flush_np, dtype=wp.int16, device="cuda")

        return {"primes": primes_wp, "unsuited": unsuited_wp, "flush": flush_wp}

    def test_eval_5_hierarchy(self, lookup_data):
        """
        Tests eval_5 against specific hand categories to ensure correct ranking order.
        """

        @wp.kernel
        def test_eval5_kernel(
            cards: wp.array(dtype=wp.int32, ndim=2),  # [N, 5]
            scores: wp.array(dtype=wp.int32),  # [N]
            primes: wp.array(dtype=wp.int32),
            unsuited_lut: wp.array(dtype=wp.int16),
            flush_lut: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            # Unpack for the @wp.func call
            c1 = cards[tid, 0]
            c2 = cards[tid, 1]
            c3 = cards[tid, 2]
            c4 = cards[tid, 3]
            c5 = cards[tid, 4]

            scores[tid] = evaluator.eval_5(c1, c2, c3, c4, c5, flush_lut, unsuited_lut, primes)

        # Define hands (Weakest to Strongest)
        hands = [
            # 0. High Card (7-5-4-3-2 unsuited) - Absolute worst hand
            [card("7", "c"), card("5", "d"), card("4", "h"), card("3", "s"), card("2", "c")],
            # 1. One Pair (2-2-5-4-3)
            [card("2", "c"), card("2", "d"), card("5", "h"), card("4", "s"), card("3", "c")],
            # 2. Two Pair (3-3-2-2-5)
            [card("3", "c"), card("3", "d"), card("2", "h"), card("2", "s"), card("5", "c")],
            # 3. Trips (AAA)
            [card("A", "c"), card("A", "d"), card("A", "h"), card("2", "s"), card("3", "c")],
            # 4. Straight (2-3-4-5-6)
            [card("2", "c"), card("3", "d"), card("4", "h"), card("5", "s"), card("6", "c")],
            # 5. Flush (A-K-Q-J-9 spades) - Beats straight
            [card("A", "s"), card("K", "s"), card("Q", "s"), card("J", "s"), card("9", "s")],
            # 6. Full House (KKK22)
            [card("K", "c"), card("K", "d"), card("K", "h"), card("2", "s"), card("2", "c")],
            # 7. Quads (2222A)
            [card("2", "c"), card("2", "d"), card("2", "h"), card("2", "s"), card("A", "c")],
            # 8. Straight Flush (3-4-5-6-7 hearts)
            [card("3", "h"), card("4", "h"), card("5", "h"), card("6", "h"), card("7", "h")],
            # 9. Royal Flush (Spades) - Absolute best hand
            [card("A", "s"), card("K", "s"), card("Q", "s"), card("J", "s"), card("T", "s")],
        ]

        n = len(hands)
        cards_np = np.array(hands, dtype=np.int32)

        # Warp Allocations
        cards_wp = wp.from_numpy(cards_np, dtype=wp.int32, device="cuda")
        scores_wp = wp.zeros(n, dtype=wp.int32, device="cuda")

        # Launch
        wp.launch(
            kernel=test_eval5_kernel,
            dim=n,
            inputs=[
                cards_wp,
                scores_wp,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
            ],
        )

        scores = scores_wp.numpy()

        print("\n5-Card Evaluation Scores:")
        for i, score in enumerate(scores):
            print(f"Hand {i}: {score}")

        # Verify strict ascending order
        # Since our input list was ordered by strength, the scores must
        # strictly increase
        for i in range(n - 1):
            assert scores[i] < scores[i + 1], (
                f"Hand {i} (Score {scores[i]}) should be "
                f"weaker than Hand {i + 1} (Score {scores[i + 1]})"
            )

    def test_eval_7_combinations(self, lookup_data):
        """
        Tests eval_7 logic: given 7 cards, verify it finds the max 5-card score.
        Uses specific scenarios where the best hand uses different combos of hole/board cards.
        """

        @wp.kernel
        def test_eval7_kernel(
            hole_cards: wp.array(dtype=wp.int32, ndim=2),  # [N, 2]
            board: wp.array(dtype=wp.int32, ndim=2),  # [N, 5]
            scores: wp.array(dtype=wp.int32),  # [N]
            primes: wp.array(dtype=wp.int32),
            unsuited_lut: wp.array(dtype=wp.int16),
            flush_lut: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            # Pass slices to the function
            scores[tid] = evaluator.eval_7(
                hole_cards[tid], board[tid], flush_lut, unsuited_lut, primes
            )

        scenarios = []
        expected_descriptions = []

        # Scenario A: Royal Flush on Board (Playing the board)
        # Hole: 2c, 3c (Trash)
        # Board: As, Ks, Qs, Js, Ts (Royal Flush)
        scenarios.append(
            {
                "hole": [card("2", "c"), card("3", "c")],
                "board": [
                    card("A", "s"),
                    card("K", "s"),
                    card("Q", "s"),
                    card("J", "s"),
                    card("T", "s"),
                ],
            }
        )
        expected_descriptions.append("Board Royal Flush")

        # Scenario B: Hole cards make Quads
        # Hole: 8c, 8d
        # Board: 8h, 8s, A, K, Q
        scenarios.append(
            {
                "hole": [card("8", "c"), card("8", "d")],
                "board": [
                    card("8", "h"),
                    card("8", "s"),
                    card("A", "c"),
                    card("K", "c"),
                    card("Q", "c"),
                ],
            }
        )
        expected_descriptions.append("Pocket Pair Quads")

        # Scenario C: 3-card Flush (1 hole, 4 board) vs Full House (2 hole, 3
        # board)
        # Hole: Ah, As
        # Board: Ad, Kh, Ks, 2h, 3h
        # Best hand: Full House (A-A-A-K-K)
        # Trap: There are 3 hearts (Ah, Kh, 2h, 3h is 4 hearts... wait, need 5
        # for flush).
        # Let's make it a real choice.
        # Board: 2s, 3s, 4s, 5s, Ah
        # Hole: As, Ks
        # Hand 1: As-Ks-5s-4s-3s (Ace-High Flush)
        # Hand 2: As-Ah... nothing.
        # Correct choice: Flush.
        scenarios.append(
            {
                "hole": [card("A", "s"), card("K", "s")],
                "board": [
                    card("2", "s"),
                    card("3", "s"),
                    card("4", "s"),
                    card("7", "s"),
                    card("A", "h"),
                ],
            }
        )
        expected_descriptions.append("Ace High Flush using both hole cards")

        # Prepare Data
        n = len(scenarios)
        hole_np = np.array([s["hole"] for s in scenarios], dtype=np.int32)
        board_np = np.array([s["board"] for s in scenarios], dtype=np.int32)

        # GPU Alloc
        hole_wp = wp.from_numpy(hole_np, dtype=wp.int32, device="cuda")
        board_wp = wp.from_numpy(board_np, dtype=wp.int32, device="cuda")
        scores_wp = wp.zeros(n, dtype=wp.int32, device="cuda")

        # Launch
        wp.launch(
            kernel=test_eval7_kernel,
            dim=n,
            inputs=[
                hole_wp,
                board_wp,
                scores_wp,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
            ],
        )

        scores = scores_wp.numpy()
        print("\n7-Card Evaluation Results:")

        # Verify Scenario A (Royal Flush)
        # Royal Flush score is max possible
        royal_score = scores[0]
        assert royal_score > 7400, f"Royal flush score too low: {royal_score}"
        print(f"Scenario A ({expected_descriptions[0]}): {royal_score} (OK)")

        # Verify Scenario B (Quads)
        quads_score = scores[1]
        # Quads is high, but lower than Straight Flush
        assert 7000 < quads_score < royal_score, f"Quads score unexpected: {quads_score}"
        print(f"Scenario B ({expected_descriptions[1]}): {quads_score} (OK)")

        # Verify Scenario C (Flush)
        flush_score = scores[2]
        # Based on lookup generator:
        # - Flush base is ~5873
        # - Full house base is ~7160
        assert 5873 < flush_score < 7160, f"Flush score unexpected: {flush_score}"
        print(f"Scenario C ({expected_descriptions[2]}): {flush_score} (OK)")
