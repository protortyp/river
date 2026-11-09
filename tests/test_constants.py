"""Invariants that other modules rely on, rather than the constant values themselves."""

from gpu_poker import constants


def test_deck_is_ranks_times_suits():
    assert constants.NUM_RANKS * constants.NUM_SUITS == constants.NUM_CARDS


def test_rank_primes_are_distinct_primes_in_ascending_order():
    def is_prime(n):
        return n >= 2 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))

    primes = list(constants.RANK_PRIMES)
    assert len(primes) == constants.NUM_RANKS
    assert all(is_prime(p) for p in primes)
    assert primes == sorted(set(primes))


def test_sentinels_lie_outside_valid_ranges():
    assert constants.INVALID_CARD not in range(constants.NUM_CARDS)
    assert constants.INVALID_ACTION not in range(constants.NUM_ACTIONS)
    assert constants.INVALID_PLAYER not in range(constants.NUM_PLAYERS)
