"""
Unit tests for card utility kernels.

Since Warp functions (@wp.func) cannot be called directly from Python,
we define wrapper kernels (@wp.kernel) here to invoke them and verify results.
"""

import numpy as np
import pytest
import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import card_utils

# Initialize Warp once
wp.init()


class TestCardUtils:
    @pytest.fixture(scope="class")
    def primes_lut(self):
        """Fixture to provide the primes lookup table on device."""
        primes = np.array(c.RANK_PRIMES, dtype=np.int32)
        return wp.from_numpy(primes, dtype=wp.int32)

    def test_stateless_conversions(self, primes_lut):
        """
        Test get_rank, get_suit, get_rank_bitmask, and get_prime.
        """

        # Define a wrapper kernel to call the device functions
        @wp.kernel
        def test_conversions_kernel(
            card_ids: wp.array(dtype=wp.int32),
            primes: wp.array(dtype=wp.int32),
            out_rank: wp.array(dtype=wp.int32),
            out_suit: wp.array(dtype=wp.int32),
            out_mask: wp.array(dtype=wp.int32),
            out_prime: wp.array(dtype=wp.int32),
        ):
            tid = wp.tid()
            c = card_ids[tid]

            out_rank[tid] = card_utils.get_rank(c)
            out_suit[tid] = card_utils.get_suit(c)
            out_mask[tid] = card_utils.get_rank_bitmask(c)
            out_prime[tid] = card_utils.get_prime(c, primes)

        # Test Data:
        # 0:  2 of Clubs (Rank 0, Suit 0, Prime 2)
        # 12: Ace of Clubs (Rank 12, Suit 0, Prime 41)
        # 13: 2 of Diamonds (Rank 0, Suit 1, Prime 2)
        # 51: Ace of Spades (Rank 12, Suit 3, Prime 41)
        # -5: Invalid Card
        cards_input = np.array([0, 12, 13, 51, -5], dtype=np.int32)
        n = len(cards_input)

        # Allocate memory
        card_ids_wp = wp.from_numpy(cards_input, dtype=wp.int32)
        out_rank = wp.zeros(n, dtype=wp.int32)
        out_suit = wp.zeros(n, dtype=wp.int32)
        out_mask = wp.zeros(n, dtype=wp.int32)
        out_prime = wp.zeros(n, dtype=wp.int32)

        # Launch kernel
        wp.launch(
            kernel=test_conversions_kernel,
            dim=n,
            inputs=[card_ids_wp, primes_lut, out_rank, out_suit, out_mask, out_prime],
        )

        # Copy back to CPU
        ranks = out_rank.numpy()
        suits = out_suit.numpy()
        masks = out_mask.numpy()
        primes = out_prime.numpy()

        # 1. Check Ranks (0-12)
        # -5 should return -1
        np.testing.assert_equal(ranks, [0, 12, 0, 12, -1])

        # 2. Check Suits (0-3)
        np.testing.assert_equal(suits, [0, 0, 1, 3, -1])

        # 3. Check Bitmasks (1 << rank)
        # 2 -> 1<<0 = 1
        # A -> 1<<12 = 4096
        # Invalid -> 0
        np.testing.assert_equal(masks, [1, 4096, 1, 4096, 0])

        # 4. Check Primes
        # 2 -> 2
        # A -> 41
        # Invalid -> 1 (Identity)
        expected_primes = [2, 41, 2, 41, 1]
        np.testing.assert_equal(primes, expected_primes)

    def test_deck_operations(self):
        """
        Test init_deck and shuffle_deck.
        """

        @wp.kernel
        def test_deck_kernel(
            deck: wp.array(dtype=wp.int32),
            seed: wp.uint32,
            mode: wp.int32,  # 0=init, 1=shuffle
        ):
            # We call the functions on a single thread to verify logic
            if mode == 0:
                card_utils.init_deck(deck)
            else:
                card_utils.shuffle_deck(deck, seed)

        # Setup
        deck_size = c.NUM_CARDS
        deck_wp = wp.zeros(deck_size, dtype=wp.int32)

        # 1. Test Initialization
        wp.launch(kernel=test_deck_kernel, dim=1, inputs=[deck_wp, wp.uint32(0), 0])

        deck_np = deck_wp.numpy()
        expected = np.arange(52, dtype=np.int32)

        # Verify exact 0..51 sequence
        np.testing.assert_equal(deck_np, expected)

        # 2. Test Shuffling
        # Run shuffle kernel
        seed = 42
        wp.launch(kernel=test_deck_kernel, dim=1, inputs=[deck_wp, wp.uint32(seed), 1])

        shuffled = deck_wp.numpy()

        # Property A: Integrity (Conservation of cards)
        # Sorting the shuffled deck must result in 0..51
        shuffled_sorted = np.sort(shuffled)
        np.testing.assert_equal(shuffled_sorted, expected)

        # Property B: Derangement (Order changed)
        # It is statistically impossible for a shuffled deck to match ordered deck exactly
        assert not np.array_equal(shuffled, expected)

        # 3. Test Determinism
        # Re-initialize
        wp.launch(kernel=test_deck_kernel, dim=1, inputs=[deck_wp, wp.uint32(0), 0])
        # Shuffle with SAME seed
        wp.launch(kernel=test_deck_kernel, dim=1, inputs=[deck_wp, wp.uint32(seed), 1])

        shuffled_again = deck_wp.numpy()
        np.testing.assert_equal(shuffled, shuffled_again)
