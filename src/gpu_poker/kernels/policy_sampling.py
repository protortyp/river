import warp as wp

from gpu_poker import constants as c


@wp.func
def _count_legal(mask_row: wp.array(dtype=wp.bool)) -> wp.int32:
    cnt = 0
    for i in range(c.NUM_ACTIONS):
        if mask_row[i]:
            cnt += 1
    return wp.int32(cnt)


@wp.func
def _select_nth_legal(mask_row: wp.array(dtype=wp.bool), n: wp.int32) -> wp.int32:
    idx = 0
    for i in range(c.NUM_ACTIONS):
        if mask_row[i]:
            if idx == n:
                return wp.int32(i)
            idx += 1
    return wp.int32(c.ACTION_FOLD)


@wp.kernel
def sample_actions_kernel(
    rng_state: wp.array(dtype=wp.uint32),
    action_mask: wp.array(dtype=wp.bool, ndim=2),  # [N, 4]
    min_raise: wp.array(dtype=wp.int32),  # [N]
    max_raise: wp.array(dtype=wp.int32),  # [N]
    out_action_type: wp.array(dtype=wp.int32),  # [N]
    out_amount: wp.array(dtype=wp.int32),  # [N] raise delta in chips
):
    """
    Samples a legal `action_type` uniformly from the `action_mask` and samples a
    raise delta when `ACTION_RAISE` is selected.

    This is meant for benchmarking / smoke training loops to avoid host-side
    sampling overhead.
    """
    i = wp.tid()
    seed = rng_state[i]
    r = wp.rand_init(wp.int32(seed))

    # Sample action type uniformly among legal actions.
    cnt = _count_legal(action_mask[i])
    # Fallback: fold (should not happen; fold is always legal).
    if cnt <= 0:
        out_action_type[i] = c.ACTION_FOLD
        out_amount[i] = 0
    else:
        pick = wp.int32(wp.randi(r, 0, cnt))
        a = _select_nth_legal(action_mask[i], pick)
        out_action_type[i] = a

        if a == c.ACTION_RAISE:
            lo = min_raise[i]
            hi = max_raise[i]
            amt = 0
            if hi > 0:
                amt = wp.int32(wp.randi(r, lo, hi + 1)) if hi >= lo else hi
            out_amount[i] = amt
        else:
            out_amount[i] = 0

    rng_state[i] = seed * wp.uint32(1664525) + wp.uint32(1013904223)
