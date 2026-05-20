import warp as wp

from gpu_poker import constants as c


@wp.func
def get_rank(card_idx: wp.int32) -> wp.int32:
    if card_idx < 0:
        return -1  # Safety for invalid cards
    return card_idx % c.NUM_RANKS


@wp.func
def get_suit(card_idx: wp.int32) -> wp.int32:
    if card_idx < 0:
        return -1
    return card_idx // c.NUM_RANKS


@wp.func
def get_rank_bitmask(card_idx: wp.int32) -> wp.int32:
    """Returns 1 << rank for bitwise hand evaluation."""
    r = get_rank(card_idx)
    if r < 0:
        return 0
    return 1 << r


@wp.func
def get_prime(card_idx: wp.int32, primes: wp.array(dtype=wp.int32)) -> wp.int32:
    """
    Returns the prime number associated with the card rank.
    Requires the primes lookup array to be passed in.
    """
    r = get_rank(card_idx)
    if r < 0:
        return 1  # Identity for multiplication
    return primes[r]


@wp.func
def swap(arr: wp.array(dtype=wp.int32), i: wp.int32, j: wp.int32):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp


@wp.func
def init_deck(deck: wp.array(dtype=wp.int32)):
    """Resets a deck to sorted order 0..51."""
    for i in range(c.NUM_CARDS):
        deck[i] = i


@wp.func
def shuffle_deck(deck: wp.array(dtype=wp.int32), rng_state: wp.uint32):
    """
    In-place Fisher-Yates shuffle.
    Requires a thread-local RNG seed/state.
    """
    state = wp.rand_init(wp.int32(rng_state))
    for i in range(c.NUM_CARDS - 1, 0, -1):
        j = wp.randi(state, 0, i + 1)
        swap(deck, i, j)
