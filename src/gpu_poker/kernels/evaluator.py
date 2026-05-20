import warp as wp

from gpu_poker.kernels import card_utils


@wp.func
def eval_5(
    c1: wp.int32,
    c2: wp.int32,
    c3: wp.int32,
    c4: wp.int32,
    c5: wp.int32,
    flush_lut: wp.array(dtype=wp.int16),
    unsuited_lut: wp.array(dtype=wp.int16),
    primes: wp.array(dtype=wp.int32),
) -> wp.int32:
    # 1. Check for Flush
    s1 = card_utils.get_suit(c1)
    s2 = card_utils.get_suit(c2)
    s3 = card_utils.get_suit(c3)
    s4 = card_utils.get_suit(c4)
    s5 = card_utils.get_suit(c5)

    is_flush = (s1 == s2) and (s2 == s3) and (s3 == s4) and (s4 == s5)

    if is_flush:
        # Calculate Rank Bitmask
        # OR operations to combine bits
        mask = (
            card_utils.get_rank_bitmask(c1)
            | card_utils.get_rank_bitmask(c2)
            | card_utils.get_rank_bitmask(c3)
            | card_utils.get_rank_bitmask(c4)
            | card_utils.get_rank_bitmask(c5)
        )
        return wp.int32(flush_lut[mask])
    else:
        # Calculate Prime Product
        # Note: Using int32 for product.
        # Max theoretical product is ~104M (Quad Aces + King), fits in int32
        # (2B).
        p = (
            card_utils.get_prime(c1, primes)
            * card_utils.get_prime(c2, primes)
            * card_utils.get_prime(c3, primes)
            * card_utils.get_prime(c4, primes)
            * card_utils.get_prime(c5, primes)
        )
        return wp.int32(unsuited_lut[p])


@wp.func
def eval_7(
    hole_cards: wp.array(dtype=wp.int32),  # [2]
    board: wp.array(dtype=wp.int32),  # [5]
    flush_lut: wp.array(dtype=wp.int16),
    unsuited_lut: wp.array(dtype=wp.int16),
    primes: wp.array(dtype=wp.int32),
) -> wp.int32:
    # Unpack cards for readability/performance in the loops
    h0 = hole_cards[0]
    h1 = hole_cards[1]
    b0 = board[0]
    b1 = board[1]
    b2 = board[2]
    b3 = board[3]
    b4 = board[4]

    # Max score tracker
    best_score = 0

    # There are exactly 21 combinations of 5 cards from 7.
    # We unroll them manually or use a structured set of calls.
    # Since we can't easily loop over combinations in Warp without recursion or
    # stack, hardcoding the 21 calls is the fastest and most GPU-friendly way
    # (no registers used for loops).

    # 1. Use both hole cards (5 board cards choose 3) -> 10 combos
    best_score = wp.max(best_score, eval_5(h0, h1, b0, b1, b2, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b0, b1, b3, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b0, b1, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b0, b2, b3, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b0, b2, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b0, b3, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b1, b2, b3, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b1, b2, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b1, b3, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, h1, b2, b3, b4, flush_lut, unsuited_lut, primes))

    # 2. Use first hole card (5 board cards choose 4) -> 5 combos
    best_score = wp.max(best_score, eval_5(h0, b0, b1, b2, b3, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, b0, b1, b2, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, b0, b1, b3, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, b0, b2, b3, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h0, b1, b2, b3, b4, flush_lut, unsuited_lut, primes))

    # 3. Use second hole card (5 board cards choose 4) -> 5 combos
    best_score = wp.max(best_score, eval_5(h1, b0, b1, b2, b3, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h1, b0, b1, b2, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h1, b0, b1, b3, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h1, b0, b2, b3, b4, flush_lut, unsuited_lut, primes))
    best_score = wp.max(best_score, eval_5(h1, b1, b2, b3, b4, flush_lut, unsuited_lut, primes))

    # 4. Play the board (5 board cards choose 5) -> 1 combo
    best_score = wp.max(best_score, eval_5(b0, b1, b2, b3, b4, flush_lut, unsuited_lut, primes))

    return best_score
