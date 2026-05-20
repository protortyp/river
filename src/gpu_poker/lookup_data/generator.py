"""
Generates hand ranking lookup tables for the GPU poker environment.

This script pre-calculates the strength of every possible 5-card poker hand
and stores the results in two lookup tables, optimized for GPU access:

1.  unsuited_table: Maps a prime-product of card ranks to a hand strength score.
    Used for all non-flush hands (pairs, straights, high card, etc.).
    This is a large, sparse array.

2.  flush_table: Maps a bitmask of card ranks to a hand strength score.
    Used only when a flush is detected. This is a small, dense array.

This approach allows for a branchless, O(1) hand evaluation on the GPU.

To run this script, ensure you are in the project root directory:
    python -m src.gpu_poker.lookup_data.generator
"""

import argparse
import itertools
import os
from collections import Counter

import numpy as np
from src.gpu_poker.constants import RANK_PRIMES

# ==============================================================================
# Hand Strength Constants & Rank Definitions
# ==============================================================================

# Card Ranks (0=Two, ..., 12=Ace)
RANKS = range(13)

# Max prime product for 5 cards: A,A,A,A,K => 41*41*41*41*37
# We add a buffer for safety.
MAX_PRIME_PRODUCT = 300000000  # Adjusted for K,K,K,K,A and other high hands

# Max rank bitmask: 2^13 for 13 ranks
MAX_RANK_BITMASK = 1 << 13

# Hand categories, ordered by strength (higher is better)
# These are the base scores before considering kickers.
# Offsets calculated based on cumulative counts of previous categories.
# Block sizes in _hand_scores:
# 1. Distinct Ranks: 1287 (indices 0-1286) - Used for High Card & Flush
# 2. Pair: 2860 (indices 1287-4146)
# 3. Two Pair: 858 (indices 4147-5004)
# 4. Trips: 858 (indices 5005-5862)
# 5. Full House: 156 (indices 5863-6018)
# 6. Quads: 156 (indices 6019-6174)

HIGH_CARD_BASE = 0
PAIR_BASE = 1287  # After 1287 distinct rank combos
TWO_PAIR_BASE = 4147  # 1287 + 2860
TRIPS_BASE = 5005  # 4147 + 858
STRAIGHT_BASE = 5863  # 5005 + 858
# Straights are scored as STRAIGHT_BASE + rank + 1 where rank is 3 (wheel) to 12 (broadway)
# This gives scores 5867-5876, so flush must start at 5877
FLUSH_BASE = 5877  # 5863 + 14 (accounts for straight scoring formula)
FULL_HOUSE_BASE = 7164  # 5877 + 1287 flushes
QUADS_BASE = 7320  # 7164 + 156 full houses
STRAIGHT_FLUSH_BASE = 7476  # 7320 + 156 quads

# Start indices in _hand_scores for relative offset calculation
IDX_START_PAIR = 1287
IDX_START_2PAIR = 4147
IDX_START_TRIPS = 5005
IDX_START_FH = 5863
IDX_START_QUADS = 6019


def _rank_combos(n):
    return itertools.combinations(RANKS, n)


class HandGenerator:
    """
    Generates and evaluates 5-card poker hands to build lookup tables.
    """

    def __init__(self):
        # These tables will be populated.
        self.unsuited_table = np.zeros(MAX_PRIME_PRODUCT, dtype=np.int16)
        self.flush_table = np.zeros(MAX_RANK_BITMASK, dtype=np.int16)

        # Internal scorekeeping to rank hands relative to each other.
        self._hand_scores = self._generate_lexographical_scores()

    def _generate_lexographical_scores(self) -> dict:
        """
        Creates a dictionary mapping hand patterns to their rank score.

        Hands are assigned scores in poker strength order within each category.
        Lower index = weaker hand (for our higher-is-better convention).

        For each category, we use a custom sort key that extracts the structural
        components (pair rank, trip rank, kickers, etc.) and sorts by poker
        strength order.
        """
        scores = {}

        # Helper to extract pair rank from a key
        def get_pair_rank(key):
            counts = Counter(key)
            return [r for r, c in counts.items() if c == 2][0]

        # Helper to extract kickers (ranks that appear once) sorted descending
        def get_kickers(key):
            counts = Counter(key)
            return tuple(sorted([r for r, c in counts.items() if c == 1], reverse=True))

        # Helper to extract trip rank from a key
        def get_trip_rank(key):
            counts = Counter(key)
            return [r for r, c in counts.items() if c == 3][0]

        # Helper to extract quad rank from a key
        def get_quad_rank(key):
            counts = Counter(key)
            return [r for r, c in counts.items() if c == 4][0]

        # 1. High Card (5 distinct ranks)
        # Lexicographic sort on descending tuple IS correct for high cards
        high_card_keys = []
        for ranks in itertools.combinations(RANKS, 5):
            key = tuple(sorted(ranks, reverse=True))
            high_card_keys.append(key)
        high_card_keys.sort()  # Lex sort = poker strength order (weakest first)
        for key in high_card_keys:
            scores[key] = len(scores)

        # 2. One Pair
        # Sort by: (pair_rank, kickers) - pair rank is most important
        pair_keys = []
        for pair_rank in RANKS:
            for kickers in itertools.combinations([r for r in RANKS if r != pair_rank], 3):
                ranks = (pair_rank, pair_rank) + tuple(kickers)
                key = tuple(sorted(ranks, reverse=True))
                pair_keys.append(key)
        pair_keys.sort(key=lambda k: (get_pair_rank(k), get_kickers(k)))
        for key in pair_keys:
            scores[key] = len(scores)

        # 3. Two Pair
        # Sort by: (higher_pair, lower_pair, kicker)
        def two_pair_sort_key(key):
            counts = Counter(key)
            pairs = sorted([r for r, c in counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in counts.items() if c == 1][0]
            return (pairs[0], pairs[1], kicker)

        two_pair_keys = []
        for pair_ranks in itertools.combinations(RANKS, 2):
            for kicker in RANKS:
                if kicker not in pair_ranks:
                    ranks = (pair_ranks[0], pair_ranks[0], pair_ranks[1], pair_ranks[1], kicker)
                    key = tuple(sorted(ranks, reverse=True))
                    two_pair_keys.append(key)
        two_pair_keys.sort(key=two_pair_sort_key)
        for key in two_pair_keys:
            scores[key] = len(scores)

        # 4. Three of a Kind
        # Sort by: (trip_rank, kickers)
        trips_keys = []
        for trip_rank in RANKS:
            for kickers in itertools.combinations([r for r in RANKS if r != trip_rank], 2):
                ranks = (trip_rank, trip_rank, trip_rank) + tuple(kickers)
                key = tuple(sorted(ranks, reverse=True))
                trips_keys.append(key)
        trips_keys.sort(key=lambda k: (get_trip_rank(k), get_kickers(k)))
        for key in trips_keys:
            scores[key] = len(scores)

        # 5. Full House
        # Sort by: (trip_rank, pair_rank)
        full_house_keys = []
        for trip_rank in RANKS:
            for pair_rank in RANKS:
                if pair_rank != trip_rank:
                    ranks = (trip_rank, trip_rank, trip_rank, pair_rank, pair_rank)
                    key = tuple(sorted(ranks, reverse=True))
                    full_house_keys.append(key)
        full_house_keys.sort(key=lambda k: (get_trip_rank(k), get_pair_rank(k)))
        for key in full_house_keys:
            scores[key] = len(scores)

        # 6. Four of a Kind
        # Sort by: (quad_rank, kicker)
        quads_keys = []
        for quad_rank in RANKS:
            for kicker in RANKS:
                if kicker != quad_rank:
                    ranks = (quad_rank, quad_rank, quad_rank, quad_rank, kicker)
                    key = tuple(sorted(ranks, reverse=True))
                    quads_keys.append(key)
        quads_keys.sort(key=lambda k: (get_quad_rank(k), get_kickers(k)[0]))
        for key in quads_keys:
            scores[key] = len(scores)

        return scores

    def evaluate_hand(self, ranks: tuple[int, ...], is_flush: bool) -> int:
        """
        Calculates the strength score for a 5-card hand.

        Args:
            ranks: A tuple of 5 card ranks (0-12).
            is_flush: True if all cards have the same suit.

        Returns:
            An integer score from 1 to 7462.
        """
        # Straights are a special case. Ace-low straight: A,2,3,4,5
        is_straight = len(set(ranks)) == 5 and (max(ranks) - min(ranks) == 4)
        is_ace_low_straight = set(ranks) == {12, 0, 1, 2, 3}  # A, 2, 3, 4, 5

        ranks = (3, 2, 1, 0, 12) if is_ace_low_straight else tuple(sorted(ranks, reverse=True))

        is_straight = is_straight or is_ace_low_straight

        if is_straight and is_flush:
            # Straight flush: score is based on highest rank
            # Ace-low is rank 3 (the '5'), high card is max(ranks)
            rank = 3 if is_ace_low_straight else max(ranks)
            return STRAIGHT_FLUSH_BASE + rank - 2

        if is_flush:
            # Lexographical score of the 5 ranks + flush base score
            return FLUSH_BASE + self._hand_scores[ranks] + 1

        if is_straight:
            # Ace-low is rank 3, otherwise max rank
            rank = 3 if is_ace_low_straight else max(ranks)
            return STRAIGHT_BASE + rank + 1

        # Count rank occurrences for pairs, trips, etc.
        counts = Counter(ranks)
        rank_counts = tuple(sorted(counts.values(), reverse=True))

        if rank_counts == (4, 1):
            return QUADS_BASE + (self._hand_scores[ranks] - IDX_START_QUADS) + 1
        if rank_counts == (3, 2):
            return FULL_HOUSE_BASE + (self._hand_scores[ranks] - IDX_START_FH) + 1
        if rank_counts == (3, 1, 1):
            return TRIPS_BASE + (self._hand_scores[ranks] - IDX_START_TRIPS) + 1
        if rank_counts == (2, 2, 1):
            return TWO_PAIR_BASE + (self._hand_scores[ranks] - IDX_START_2PAIR) + 1
        if rank_counts == (2, 1, 1, 1):
            return PAIR_BASE + (self._hand_scores[ranks] - IDX_START_PAIR) + 1

        # High card (uses first block of indices starting at 0)
        return HIGH_CARD_BASE + (self._hand_scores[ranks] - 0) + 1

    def generate_tables(self):
        """
        Iterates through all hand combinations and populates the lookup tables.
        """
        print("Generating lookup tables...")

        # 1. Generate flush_table (including straight flushes)
        # Iterate through all 1287 combinations of 5 distinct ranks for a flush.
        print("Populating flush_table...")
        for ranks_tuple in _rank_combos(5):
            bitmask = sum(1 << r for r in ranks_tuple)
            score = self.evaluate_hand(ranks_tuple, is_flush=True)
            self.flush_table[bitmask] = score

        # 2. Generate unsuited_table
        # This loop is more complex as it must cover all non-flush hands.
        # We iterate through hand structures (quads, full house, etc.)
        print("Populating unsuited_table (this will take a while)...")

        # This is a large iteration space. We generate all unique 5-rank multisets.
        for ranks_tuple in itertools.combinations_with_replacement(RANKS, 5):
            prime_product = 1
            for r in ranks_tuple:
                prime_product *= RANK_PRIMES[r]

            try:
                # We evaluate it as a non-flush hand.
                score = self.evaluate_hand(ranks_tuple, is_flush=False)
            except KeyError:
                # This combination (e.g., 5 of a kind) is impossible in a
                # real game, so we don't need a score for it.
                continue

            # Some combinations_with_replacement can be flushes (if all ranks unique)
            # but we are only populating the unsuited table here.
            # The GPU logic will check for flush first, so this is safe.
            if self.unsuited_table[prime_product] == 0:
                self.unsuited_table[prime_product] = score

    def save_tables(self, output_path: str):
        """Saves the generated tables to a compressed NPZ file."""
        print(f"Saving tables to {output_path}...")
        np.savez_compressed(
            output_path,
            unsuited_table=self.unsuited_table,
            flush_table=self.flush_table,
        )
        print("Done.")


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(description="Generate poker hand lookup tables.")
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "hand_ranks.npz"),
        help="Path to save the output .npz file.",
    )
    args = parser.parse_args()

    generator = HandGenerator()
    generator.generate_tables()
    generator.save_tables(args.output_path)

    # --- Verification Step ---
    print("\nVerifying generated tables...")
    unsuited_table = generator.unsuited_table
    flush_table = generator.flush_table

    # 1. Test Full House (K,K,K,2,2)
    # Ranks: 11, 11, 11, 0, 0
    # Primes: 37*37*37 * 2*2
    kkk22_prod = RANK_PRIMES[11] ** 3 * RANK_PRIMES[0] ** 2
    kkk22_score = unsuited_table[kkk22_prod]
    assert kkk22_score > FULL_HOUSE_BASE, f"Full house score ({kkk22_score}) is wrong!"
    print(f"OK: Full House (KKK22) score: {kkk22_score}")

    # 2. Test Ace-high flush (not straight)
    # Ranks: A,K,Q,J,9 -> 12,11,10,9,7
    akqj9_mask = (1 << 12) | (1 << 11) | (1 << 10) | (1 << 9) | (1 << 7)
    akqj9_score = flush_table[akqj9_mask]
    assert FLUSH_BASE < akqj9_score <= FULL_HOUSE_BASE
    print(f"OK: Ace-high flush score: {akqj9_score} (Max Flush range: {FULL_HOUSE_BASE})")

    # 3. Test Royal Flush
    # Ranks: A,K,Q,J,T -> 12,11,10,9,8
    royal_mask = (1 << 12) | (1 << 11) | (1 << 10) | (1 << 9) | (1 << 8)
    royal_score = flush_table[royal_mask]
    # Max score is for Royal Flush, based on our ranking
    assert royal_score == STRAIGHT_FLUSH_BASE + 10, "Royal flush score is wrong!"
    print(f"OK: Royal Flush score: {royal_score}")

    # 4. Test a high card hand
    # Ranks: A,K,Q,J,9 -> 12,11,10,9,7
    akqj9_prod = (
        RANK_PRIMES[12] * RANK_PRIMES[11] * RANK_PRIMES[10] * RANK_PRIMES[9] * RANK_PRIMES[7]
    )
    akqj9_unsuited_score = unsuited_table[akqj9_prod]
    assert HIGH_CARD_BASE < akqj9_unsuited_score < PAIR_BASE
    print(f"OK: Ace-high (unsuited) score: {akqj9_unsuited_score}")

    # 5. Test Broadway straight (A-K-Q-J-T unsuited)
    # Ranks: 12,11,10,9,8
    broadway_prod = (
        RANK_PRIMES[12] * RANK_PRIMES[11] * RANK_PRIMES[10] * RANK_PRIMES[9] * RANK_PRIMES[8]
    )
    broadway_score = unsuited_table[broadway_prod]
    assert STRAIGHT_BASE < broadway_score < FLUSH_BASE, (
        f"Broadway straight score ({broadway_score}) should be between "
        f"STRAIGHT_BASE ({STRAIGHT_BASE}) and FLUSH_BASE ({FLUSH_BASE})"
    )
    print(f"OK: Broadway straight score: {broadway_score}")

    # 6. Test Wheel straight (A-2-3-4-5 unsuited)
    # Ranks: 12,0,1,2,3
    wheel_prod = RANK_PRIMES[12] * RANK_PRIMES[0] * RANK_PRIMES[1] * RANK_PRIMES[2] * RANK_PRIMES[3]
    wheel_score = unsuited_table[wheel_prod]
    assert STRAIGHT_BASE < wheel_score < broadway_score, (
        f"Wheel score ({wheel_score}) should be less than broadway ({broadway_score})"
    )
    print(f"OK: Wheel straight score: {wheel_score}")

    print("\nVerification complete. Tables appear to be generated correctly.")


if __name__ == "__main__":
    main()
