"""
Cross-validation tests for the GPU Hand Evaluator against the treys library.

This test compares our Cactus Kev implementation against the well-established
treys library to ensure correctness across a large sample of random hands.
"""

import os
import random

import numpy as np
import pytest
import warp as wp
from treys import Card
from treys import Evaluator as TreysEvaluator

from gpu_poker import constants as c
from gpu_poker.kernels import evaluator

wp.init()

try:
    HAS_CUDA = wp.is_device_available("cuda")
except RuntimeError:
    HAS_CUDA = False

DEVICE = "cuda" if HAS_CUDA else "cpu"

# Rank/suit string mappings for treys conversion
RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"  # clubs, diamonds, hearts, spades


def gpu_card_to_treys(gpu_card: int) -> int:
    """Convert GPU card representation to treys Card integer.

    GPU format: card = rank + (suit * 13)
        - rank: 0-12 (2 through A)
        - suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades

    Treys format: Card.new('As') style strings, then internal int representation.
    """
    rank = gpu_card % 13
    suit = gpu_card // 13

    rank_char = RANK_CHARS[rank]
    suit_char = SUIT_CHARS[suit]

    return Card.new(rank_char + suit_char)


def treys_to_gpu_card(treys_card: int) -> int:
    """Convert treys Card integer to GPU card representation."""
    # Get the string representation from treys
    card_str = Card.int_to_str(treys_card)
    rank_char = card_str[0]
    suit_char = card_str[1].lower()

    rank = RANK_CHARS.index(rank_char)
    suit = SUIT_CHARS.index(suit_char)

    return rank + (suit * 13)


def generate_random_hand(n_cards: int, rng: random.Random) -> list[int]:
    """Generate a random hand of n_cards unique GPU card indices."""
    deck = list(range(52))
    rng.shuffle(deck)
    return deck[:n_cards]


@pytest.fixture(scope="module")
def lookup_data():
    """Loads the lookup tables and primes array into Warp GPU arrays."""
    primes_np = np.array(c.RANK_PRIMES, dtype=np.int32)
    primes_wp = wp.from_numpy(primes_np, dtype=wp.int32, device=DEVICE)

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

    unsuited_wp = wp.from_numpy(unsuited_np, dtype=wp.int16, device=DEVICE)
    flush_wp = wp.from_numpy(flush_np, dtype=wp.int16, device=DEVICE)

    return {"primes": primes_wp, "unsuited": unsuited_wp, "flush": flush_wp}


@pytest.fixture(scope="module")
def treys_evaluator():
    """Create a treys Evaluator instance."""
    return TreysEvaluator()


class TestEvaluatorVsTreys:
    """Cross-validation tests comparing GPU evaluator against treys."""

    def test_card_conversion_roundtrip(self):
        """Verify card conversion functions work correctly."""
        for gpu_card in range(52):
            treys_card = gpu_card_to_treys(gpu_card)
            back_to_gpu = treys_to_gpu_card(treys_card)
            assert gpu_card == back_to_gpu, f"Roundtrip failed for card {gpu_card}"

    def test_specific_hands_match(self, lookup_data, treys_evaluator):
        """Test specific known hands match between implementations."""

        @wp.kernel
        def eval5_kernel(
            cards: wp.array(dtype=wp.int32, ndim=2),
            scores: wp.array(dtype=wp.int32),
            primes: wp.array(dtype=wp.int32),
            unsuited_lut: wp.array(dtype=wp.int16),
            flush_lut: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            c1 = cards[tid, 0]
            c2 = cards[tid, 1]
            c3 = cards[tid, 2]
            c4 = cards[tid, 3]
            c5 = cards[tid, 4]
            scores[tid] = evaluator.eval_5(c1, c2, c3, c4, c5, flush_lut, unsuited_lut, primes)

        # Define test hands (GPU card format)
        def card(rank_str: str, suit_str: str) -> int:
            r = RANK_CHARS.index(rank_str)
            s = SUIT_CHARS.index(suit_str)
            return r + (s * 13)

        test_hands = [
            # Royal flush (spades)
            [card("A", "s"), card("K", "s"), card("Q", "s"), card("J", "s"), card("T", "s")],
            # Straight flush (5-high, hearts) - wheel
            [card("A", "h"), card("2", "h"), card("3", "h"), card("4", "h"), card("5", "h")],
            # Four of a kind (aces)
            [card("A", "c"), card("A", "d"), card("A", "h"), card("A", "s"), card("K", "c")],
            # Full house (kings full of twos)
            [card("K", "c"), card("K", "d"), card("K", "h"), card("2", "s"), card("2", "c")],
            # Flush (ace-high, clubs)
            [card("A", "c"), card("J", "c"), card("9", "c"), card("6", "c"), card("3", "c")],
            # Straight (broadway)
            [card("A", "c"), card("K", "d"), card("Q", "h"), card("J", "s"), card("T", "c")],
            # Straight (wheel)
            [card("A", "c"), card("2", "d"), card("3", "h"), card("4", "s"), card("5", "c")],
            # Three of a kind
            [card("7", "c"), card("7", "d"), card("7", "h"), card("A", "s"), card("K", "c")],
            # Two pair
            [card("A", "c"), card("A", "d"), card("K", "h"), card("K", "s"), card("Q", "c")],
            # One pair
            [card("A", "c"), card("A", "d"), card("K", "h"), card("Q", "s"), card("J", "c")],
            # High card
            [card("A", "c"), card("K", "d"), card("Q", "h"), card("J", "s"), card("9", "c")],
        ]

        n = len(test_hands)
        cards_np = np.array(test_hands, dtype=np.int32)
        cards_wp = wp.from_numpy(cards_np, dtype=wp.int32, device=DEVICE)
        scores_wp = wp.zeros(n, dtype=wp.int32, device=DEVICE)

        wp.launch(
            kernel=eval5_kernel,
            dim=n,
            inputs=[
                cards_wp,
                scores_wp,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
            ],
        )

        gpu_scores = scores_wp.numpy()

        # Get treys scores and compare rankings
        treys_scores = []
        for hand in test_hands:
            treys_hand = [gpu_card_to_treys(c) for c in hand]
            # treys uses LOWER score = BETTER hand
            treys_scores.append(treys_evaluator.evaluate([], treys_hand))

        # Verify ordering matches (GPU: higher is better, treys: lower is better)
        for i in range(n - 1):
            gpu_better = gpu_scores[i] > gpu_scores[i + 1]
            treys_better = treys_scores[i] < treys_scores[i + 1]
            assert gpu_better == treys_better, (
                f"Ordering mismatch at hands {i} vs {i + 1}: "
                f"GPU scores {gpu_scores[i]} vs {gpu_scores[i + 1]}, "
                f"Treys scores {treys_scores[i]} vs {treys_scores[i + 1]}"
            )

    def test_random_5card_hands_ordering(self, lookup_data, treys_evaluator):
        """Test that random 5-card hand orderings match between implementations."""

        @wp.kernel
        def eval5_kernel(
            cards: wp.array(dtype=wp.int32, ndim=2),
            scores: wp.array(dtype=wp.int32),
            primes: wp.array(dtype=wp.int32),
            unsuited_lut: wp.array(dtype=wp.int16),
            flush_lut: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            c1 = cards[tid, 0]
            c2 = cards[tid, 1]
            c3 = cards[tid, 2]
            c4 = cards[tid, 3]
            c5 = cards[tid, 4]
            scores[tid] = evaluator.eval_5(c1, c2, c3, c4, c5, flush_lut, unsuited_lut, primes)

        rng = random.Random(42)
        n_hands = 100000

        # Generate random hands
        hands = [generate_random_hand(5, rng) for _ in range(n_hands)]
        cards_np = np.array(hands, dtype=np.int32)
        cards_wp = wp.from_numpy(cards_np, dtype=wp.int32, device=DEVICE)
        scores_wp = wp.zeros(n_hands, dtype=wp.int32, device=DEVICE)

        wp.launch(
            kernel=eval5_kernel,
            dim=n_hands,
            inputs=[
                cards_wp,
                scores_wp,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
            ],
        )

        gpu_scores = scores_wp.numpy()

        # Get treys scores
        treys_scores = []
        for hand in hands:
            treys_hand = [gpu_card_to_treys(c) for c in hand]
            treys_scores.append(treys_evaluator.evaluate([], treys_hand))

        treys_scores = np.array(treys_scores)

        # Compare all pairs: check that relative ordering is consistent
        # Sample random pairs to avoid O(n^2) comparisons
        n_comparisons = 50000
        mismatches = 0
        mismatch_examples = []

        for _ in range(n_comparisons):
            i, j = rng.sample(range(n_hands), 2)

            gpu_cmp = np.sign(gpu_scores[i] - gpu_scores[j])
            # Treys: lower is better, so we negate
            treys_cmp = np.sign(treys_scores[j] - treys_scores[i])

            if gpu_cmp != treys_cmp:
                mismatches += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        (
                            i,
                            j,
                            hands[i],
                            hands[j],
                            gpu_scores[i],
                            gpu_scores[j],
                            treys_scores[i],
                            treys_scores[j],
                        )
                    )

        if mismatch_examples:
            print("\nMismatch examples (5-card):")
            for i, j, h1, h2, gs1, gs2, ts1, ts2 in mismatch_examples:
                h1_str = [Card.int_to_str(gpu_card_to_treys(c)) for c in h1]
                h2_str = [Card.int_to_str(gpu_card_to_treys(c)) for c in h2]
                print(f"  Hand {i}: {h1_str} GPU={gs1} Treys={ts1}")
                print(f"  Hand {j}: {h2_str} GPU={gs2} Treys={ts2}")
                gpu_result = "h1 > h2" if gs1 > gs2 else ("h1 < h2" if gs1 < gs2 else "tie")
                treys_result = "h1 > h2" if ts1 < ts2 else ("h1 < h2" if ts1 > ts2 else "tie")
                print(f"  GPU says: {gpu_result}")
                print(f"  Treys says: {treys_result}")
                print()

        mismatch_rate = mismatches / n_comparisons
        assert mismatch_rate == 0.0, (
            f"Found {mismatches}/{n_comparisons} ordering mismatches ({mismatch_rate:.2%}). "
            "GPU and treys evaluators disagree on hand rankings."
        )

    def test_random_7card_hands_ordering(self, lookup_data, treys_evaluator):
        """Test that random 7-card hand orderings match between implementations."""

        @wp.kernel
        def eval7_kernel(
            hole_cards: wp.array(dtype=wp.int32, ndim=2),
            board: wp.array(dtype=wp.int32, ndim=2),
            scores: wp.array(dtype=wp.int32),
            primes: wp.array(dtype=wp.int32),
            unsuited_lut: wp.array(dtype=wp.int16),
            flush_lut: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            scores[tid] = evaluator.eval_7(
                hole_cards[tid], board[tid], flush_lut, unsuited_lut, primes
            )

        rng = random.Random(123)
        n_hands = 100000

        # Generate random 7-card hands (2 hole + 5 board)
        all_cards = [generate_random_hand(7, rng) for _ in range(n_hands)]
        hole_cards = [[h[0], h[1]] for h in all_cards]
        boards = [[h[2], h[3], h[4], h[5], h[6]] for h in all_cards]

        hole_np = np.array(hole_cards, dtype=np.int32)
        board_np = np.array(boards, dtype=np.int32)

        hole_wp = wp.from_numpy(hole_np, dtype=wp.int32, device=DEVICE)
        board_wp = wp.from_numpy(board_np, dtype=wp.int32, device=DEVICE)
        scores_wp = wp.zeros(n_hands, dtype=wp.int32, device=DEVICE)

        wp.launch(
            kernel=eval7_kernel,
            dim=n_hands,
            inputs=[
                hole_wp,
                board_wp,
                scores_wp,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
            ],
        )

        gpu_scores = scores_wp.numpy()

        # Get treys scores
        treys_scores = []
        for hole, board in zip(hole_cards, boards, strict=False):
            treys_hole = [gpu_card_to_treys(c) for c in hole]
            treys_board = [gpu_card_to_treys(c) for c in board]
            treys_scores.append(treys_evaluator.evaluate(treys_board, treys_hole))

        treys_scores = np.array(treys_scores)

        # Compare random pairs
        n_comparisons = 50000
        mismatches = 0

        for _ in range(n_comparisons):
            i, j = rng.sample(range(n_hands), 2)

            gpu_cmp = np.sign(gpu_scores[i] - gpu_scores[j])
            treys_cmp = np.sign(treys_scores[j] - treys_scores[i])

            if gpu_cmp != treys_cmp:
                mismatches += 1

        mismatch_rate = mismatches / n_comparisons
        assert mismatch_rate == 0.0, (
            f"Found {mismatches}/{n_comparisons} ordering mismatches ({mismatch_rate:.2%}). "
            "GPU and treys evaluators disagree on 7-card hand rankings."
        )

    def test_hand_class_consistency(self, lookup_data, treys_evaluator):
        """Test that hand class (pair, flush, etc.) matches between implementations."""

        @wp.kernel
        def eval5_kernel(
            cards: wp.array(dtype=wp.int32, ndim=2),
            scores: wp.array(dtype=wp.int32),
            primes: wp.array(dtype=wp.int32),
            unsuited_lut: wp.array(dtype=wp.int16),
            flush_lut: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            c1 = cards[tid, 0]
            c2 = cards[tid, 1]
            c3 = cards[tid, 2]
            c4 = cards[tid, 3]
            c5 = cards[tid, 4]
            scores[tid] = evaluator.eval_5(c1, c2, c3, c4, c5, flush_lut, unsuited_lut, primes)

        # Hand class boundaries for OUTPUT scores
        # The generator's internal indices get +1 in output formulas, so boundaries shift
        # Higher score = better hand in our implementation
        pair_min = 1288  # Internal index 1287 + 1
        two_pair_min = 4148  # Internal index 4147 + 1
        trips_min = 5006  # Internal index 5005 + 1
        straight_min = 5867  # Formula: 5863 + rank + 1, min rank=3 (wheel)
        flush_min = 5878  # Formula: 5877 + internal + 1, min internal=0
        full_house_min = 7165  # Internal index 5863 maps to output 7165
        quads_min = 7321  # Internal index 6019 maps to output 7321
        straight_flush_min = 7477  # Formula: 7476 + rank - 2, min rank=3 (wheel)

        def gpu_score_to_class(score: int) -> int:
            """Convert GPU score to hand class (0-8)."""
            if score >= straight_flush_min:
                return 8  # Straight flush
            elif score >= quads_min:
                return 7  # Four of a kind
            elif score >= full_house_min:
                return 6  # Full house
            elif score >= flush_min:
                return 5  # Flush
            elif score >= straight_min:
                return 4  # Straight
            elif score >= trips_min:
                return 3  # Three of a kind
            elif score >= two_pair_min:
                return 2  # Two pair
            elif score >= pair_min:
                return 1  # One pair
            else:
                return 0  # High card

        rng = random.Random(456)
        n_hands = 100000

        hands = [generate_random_hand(5, rng) for _ in range(n_hands)]
        cards_np = np.array(hands, dtype=np.int32)
        cards_wp = wp.from_numpy(cards_np, dtype=wp.int32, device=DEVICE)
        scores_wp = wp.zeros(n_hands, dtype=wp.int32, device=DEVICE)

        wp.launch(
            kernel=eval5_kernel,
            dim=n_hands,
            inputs=[
                cards_wp,
                scores_wp,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
            ],
        )

        gpu_scores = scores_wp.numpy()

        # Treys hand class mapping (from get_rank_class)
        # 1 = Straight Flush, 2 = Four of a Kind, ..., 9 = High Card
        treys_to_our_class = {
            1: 8,  # Straight Flush
            2: 7,  # Four of a Kind
            3: 6,  # Full House
            4: 5,  # Flush
            5: 4,  # Straight
            6: 3,  # Three of a Kind
            7: 2,  # Two Pair
            8: 1,  # One Pair
            9: 0,  # High Card
        }

        class_names = [
            "High Card",
            "Pair",
            "Two Pair",
            "Trips",
            "Straight",
            "Flush",
            "Full House",
            "Quads",
            "Straight Flush",
        ]

        mismatches = 0
        mismatch_examples = []
        for i, hand in enumerate(hands):
            treys_hand = [gpu_card_to_treys(c) for c in hand]
            treys_score = treys_evaluator.evaluate([], treys_hand)
            treys_class = treys_evaluator.get_rank_class(treys_score)

            gpu_class = gpu_score_to_class(gpu_scores[i])
            expected_class = treys_to_our_class[treys_class]

            if gpu_class != expected_class:
                mismatches += 1
                if len(mismatch_examples) < 20:
                    hand_str = [Card.int_to_str(gpu_card_to_treys(c)) for c in hand]
                    mismatch_examples.append(
                        (hand_str, gpu_scores[i], gpu_class, expected_class, treys_score)
                    )

        if mismatch_examples:
            print("\nHand class mismatch examples:")
            for hand_str, gpu_score, gpu_class, expected_class, treys_score in mismatch_examples:
                print(
                    f"  {hand_str}: GPU score={gpu_score} class={class_names[gpu_class]}, "
                    f"expected={class_names[expected_class]} (treys score={treys_score})"
                )

        assert mismatches == 0, (
            f"Found {mismatches}/{n_hands} hand class mismatches. "
            "GPU evaluator assigns wrong hand categories."
        )

    def test_tie_detection(self, lookup_data, treys_evaluator):
        """Test that identical hands produce identical scores (ties detected correctly)."""

        @wp.kernel
        def eval5_kernel(
            cards: wp.array(dtype=wp.int32, ndim=2),
            scores: wp.array(dtype=wp.int32),
            primes: wp.array(dtype=wp.int32),
            unsuited_lut: wp.array(dtype=wp.int16),
            flush_lut: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            c1 = cards[tid, 0]
            c2 = cards[tid, 1]
            c3 = cards[tid, 2]
            c4 = cards[tid, 3]
            c5 = cards[tid, 4]
            scores[tid] = evaluator.eval_5(c1, c2, c3, c4, c5, flush_lut, unsuited_lut, primes)

        def card(rank_str: str, suit_str: str) -> int:
            r = RANK_CHARS.index(rank_str)
            s = SUIT_CHARS.index(suit_str)
            return r + (s * 13)

        # Hands that should tie (same ranks, different suits where suit doesn't matter)
        tie_pairs = [
            # Same high card hand, different suits
            (
                [card("A", "c"), card("K", "d"), card("Q", "h"), card("J", "s"), card("9", "c")],
                [card("A", "d"), card("K", "h"), card("Q", "s"), card("J", "c"), card("9", "d")],
            ),
            # Same pair
            (
                [card("A", "c"), card("A", "d"), card("K", "h"), card("Q", "s"), card("J", "c")],
                [card("A", "h"), card("A", "s"), card("K", "c"), card("Q", "d"), card("J", "h")],
            ),
            # Same two pair
            (
                [card("A", "c"), card("A", "d"), card("K", "h"), card("K", "s"), card("Q", "c")],
                [card("A", "h"), card("A", "s"), card("K", "c"), card("K", "d"), card("Q", "h")],
            ),
            # Same straight (not a flush)
            (
                [card("A", "c"), card("K", "d"), card("Q", "h"), card("J", "s"), card("T", "c")],
                [card("A", "d"), card("K", "h"), card("Q", "s"), card("J", "c"), card("T", "d")],
            ),
        ]

        all_hands = [h for pair in tie_pairs for h in pair]
        n = len(all_hands)

        cards_np = np.array(all_hands, dtype=np.int32)
        cards_wp = wp.from_numpy(cards_np, dtype=wp.int32, device=DEVICE)
        scores_wp = wp.zeros(n, dtype=wp.int32, device=DEVICE)

        wp.launch(
            kernel=eval5_kernel,
            dim=n,
            inputs=[
                cards_wp,
                scores_wp,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
            ],
        )

        gpu_scores = scores_wp.numpy()

        # Check each pair ties
        for i in range(0, n, 2):
            assert gpu_scores[i] == gpu_scores[i + 1], (
                f"Expected tie for hands {i} and {i + 1}, "
                f"but got scores {gpu_scores[i]} vs {gpu_scores[i + 1]}"
            )

        # Also verify with treys
        for i in range(0, n, 2):
            treys_hand1 = [gpu_card_to_treys(c) for c in all_hands[i]]
            treys_hand2 = [gpu_card_to_treys(c) for c in all_hands[i + 1]]
            treys_score1 = treys_evaluator.evaluate([], treys_hand1)
            treys_score2 = treys_evaluator.evaluate([], treys_hand2)
            assert treys_score1 == treys_score2, (
                f"Treys also expected tie for hands {i} and {i + 1}"
            )

    def test_edge_cases(self, lookup_data, treys_evaluator):
        """Test edge cases like wheel straight, steel wheel, etc."""

        @wp.kernel
        def eval5_kernel(
            cards: wp.array(dtype=wp.int32, ndim=2),
            scores: wp.array(dtype=wp.int32),
            primes: wp.array(dtype=wp.int32),
            unsuited_lut: wp.array(dtype=wp.int16),
            flush_lut: wp.array(dtype=wp.int16),
        ):
            tid = wp.tid()
            c1 = cards[tid, 0]
            c2 = cards[tid, 1]
            c3 = cards[tid, 2]
            c4 = cards[tid, 3]
            c5 = cards[tid, 4]
            scores[tid] = evaluator.eval_5(c1, c2, c3, c4, c5, flush_lut, unsuited_lut, primes)

        def card(rank_str: str, suit_str: str) -> int:
            r = RANK_CHARS.index(rank_str)
            s = SUIT_CHARS.index(suit_str)
            return r + (s * 13)

        # Edge cases
        edge_cases = [
            # Wheel (5-high straight)
            [card("A", "c"), card("2", "d"), card("3", "h"), card("4", "s"), card("5", "c")],
            # Steel wheel (5-high straight flush)
            [card("A", "h"), card("2", "h"), card("3", "h"), card("4", "h"), card("5", "h")],
            # 6-high straight (should beat wheel)
            [card("2", "c"), card("3", "d"), card("4", "h"), card("5", "s"), card("6", "c")],
            # 6-high straight flush (should beat steel wheel)
            [card("2", "h"), card("3", "h"), card("4", "h"), card("5", "h"), card("6", "h")],
            # Royal flush
            [card("A", "s"), card("K", "s"), card("Q", "s"), card("J", "s"), card("T", "s")],
        ]

        n = len(edge_cases)
        cards_np = np.array(edge_cases, dtype=np.int32)
        cards_wp = wp.from_numpy(cards_np, dtype=wp.int32, device=DEVICE)
        scores_wp = wp.zeros(n, dtype=wp.int32, device=DEVICE)

        wp.launch(
            kernel=eval5_kernel,
            dim=n,
            inputs=[
                cards_wp,
                scores_wp,
                lookup_data["primes"],
                lookup_data["unsuited"],
                lookup_data["flush"],
            ],
        )

        gpu_scores = scores_wp.numpy()

        # Get treys scores
        treys_scores = []
        for hand in edge_cases:
            treys_hand = [gpu_card_to_treys(c) for c in hand]
            treys_scores.append(treys_evaluator.evaluate([], treys_hand))

        # Verify specific orderings
        # Wheel (0) < 6-high straight (2)
        assert gpu_scores[0] < gpu_scores[2], "6-high straight should beat wheel"
        assert treys_scores[0] > treys_scores[2], "Treys: 6-high straight should beat wheel"

        # Steel wheel (1) < 6-high straight flush (3)
        assert gpu_scores[1] < gpu_scores[3], "6-high SF should beat steel wheel"
        assert treys_scores[1] > treys_scores[3], "Treys: 6-high SF should beat steel wheel"

        # 6-high straight flush (3) < Royal flush (4)
        assert gpu_scores[3] < gpu_scores[4], "Royal flush should beat 6-high SF"
        assert treys_scores[3] > treys_scores[4], "Treys: Royal flush should beat 6-high SF"

        # Steel wheel (1) > Wheel (0) - straight flush beats straight
        assert gpu_scores[1] > gpu_scores[0], "Steel wheel (SF) should beat wheel (straight)"
        assert treys_scores[1] < treys_scores[0], "Treys: Steel wheel should beat wheel"
