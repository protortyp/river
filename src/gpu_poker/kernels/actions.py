import warp as wp

from gpu_poker import constants as c
from gpu_poker.struct_types import GameState


@wp.func
def get_opponent(player_idx: wp.int32) -> wp.int32:
    return 1 - player_idx


@wp.func
def get_to_call(state: GameState, env_idx: wp.int32, player_idx: wp.int32) -> wp.int32:
    """Calculates amount needed to call to match the highest bet."""
    opp = get_opponent(player_idx)
    # Ensure we don't return negative if logic is slightly off (safety)
    diff = state.bets[env_idx, opp] - state.bets[env_idx, player_idx]
    if diff < 0:
        return 0
    return diff


@wp.func
def execute_bet(
    state: GameState, env_idx: wp.int32, player_idx: wp.int32, amount: wp.int32
) -> wp.int32:
    """
    Moves chips from stack to pot/bets.
    Caps amount at available stack (All-in logic).
    Returns actual amount bet.
    """
    actual_amount = amount
    stack = state.stacks[env_idx, player_idx]

    # All-in logic: cap at stack
    if actual_amount >= stack:
        actual_amount = stack

    # Mutate state
    state.stacks[env_idx, player_idx] = stack - actual_amount
    state.bets[env_idx, player_idx] = state.bets[env_idx, player_idx] + actual_amount
    # Pot is updated in collect_pot (state.py), not here.

    return actual_amount


@wp.func
def validate_action(
    state: GameState,
    env_idx: wp.int32,
    player_idx: wp.int32,
    action_type: wp.int32,
    amount: wp.int32,
) -> wp.bool:
    """
    Checks if an action is valid.
    """
    to_call = get_to_call(state, env_idx, player_idx)
    stack = state.stacks[env_idx, player_idx]

    # FOLD: Always valid (unless already terminated)
    if action_type == c.ACTION_FOLD:
        return True

    # CHECK: Valid only if to_call is 0
    if action_type == c.ACTION_CHECK:
        return to_call == 0

    # CALL: Valid if to_call > 0
    # Note: In standard rules, calling more than stack is an All-In, which is valid.
    if action_type == c.ACTION_CALL:
        # If to_call is 0, 'Call' is usually treated as 'Check' or invalid.
        # Strict interpretation: Call means match bet.
        return to_call > 0

    # RAISE: Complex validation
    if action_type == c.ACTION_RAISE:
        # 1. Must have enough chips to cover the call + raise amount
        if stack <= to_call:
            return False  # Cannot raise, only call/fold (all-in call)

        # 2. Check if player has enough chips for call + raise
        total_needed = to_call + amount
        if stack < total_needed:
            return False  # Not enough chips for this raise size

        # 3. Raise size must be valid
        # 'amount' here usually implies the amount ADDED on top of the call
        # Min raise is usually the size of the previous raise (or BB)
        min_raise = state.last_raise[env_idx]
        bb = state.cfg_big_blind[env_idx]
        if bb <= 0:
            bb = c.BIG_BLIND
        if min_raise < bb:
            min_raise = bb

        if amount < min_raise and amount != (stack - to_call):
            return False
        # All-in-for-less is the only sub-min exception; it still must respect
        # the effective-stack cap (opponent's max total). For full raises, also
        # enforce the upper bound here.
        max_raise = _max_raise_delta(state, env_idx, player_idx)
        return amount <= max_raise

    return False


@wp.func
def _min_raise_delta(state: GameState, env_idx: wp.int32) -> wp.int32:
    m = state.last_raise[env_idx]
    bb = state.cfg_big_blind[env_idx]
    if bb <= 0:
        bb = c.BIG_BLIND
    if m < bb:
        m = bb
    return m


@wp.func
def _max_raise_delta(state: GameState, env_idx: wp.int32, player: wp.int32) -> wp.int32:
    opp = 1 - player
    to_call = get_to_call(state, env_idx, player)
    player_stack = state.stacks[env_idx, player]
    current_bet = state.bets[env_idx, player]

    if player_stack <= to_call:
        return 0

    opp_max_total = state.bets[env_idx, opp] + state.stacks[env_idx, opp]
    player_max_total = current_bet + player_stack

    max_total_allowed = opp_max_total
    if player_max_total < max_total_allowed:
        max_total_allowed = player_max_total

    max_delta = max_total_allowed - (current_bet + to_call)
    if max_delta < 0:
        max_delta = 0
    return max_delta


@wp.func
def _pack_action(invalid: wp.int32, action_type: wp.int32, amount: wp.int32) -> wp.int64:
    # Layout: [invalid:1bit | type:31bits | amount:32bits]
    inv = wp.uint64(invalid & 1)
    t = wp.uint64(action_type & 0x7FFFFFFF)
    a = wp.uint64(wp.uint32(amount))
    packed = (inv << wp.uint64(63)) | (t << wp.uint64(32)) | a
    return wp.int64(packed)


@wp.func
def unpack_invalid(packed: wp.int64) -> wp.bool:
    u = wp.uint64(packed)
    return ((u >> wp.uint64(63)) & wp.uint64(1)) == wp.uint64(1)


@wp.func
def unpack_action_type(packed: wp.int64) -> wp.int32:
    u = wp.uint64(packed)
    return wp.int32((u >> wp.uint64(32)) & wp.uint64(0x7FFFFFFF))


@wp.func
def unpack_amount(packed: wp.int64) -> wp.int32:
    u = wp.uint64(packed)
    return wp.int32(wp.uint32(u & wp.uint64(0xFFFFFFFF)))


@wp.func
def force_legal_action(
    state: GameState,
    env_idx: wp.int32,
    player_idx: wp.int32,
    action_type: wp.int32,
    amount: wp.int32,
) -> wp.int64:
    """
    Illegal-action policy: force to a legal action without leaving the GPU.

    - CHECK when facing bet => CALL
    - CALL when no bet => CHECK
    - RAISE gets clamped into a legal raise delta if possible, else CALL/CHECK
    - Unknown action => FOLD

    Returns a packed int64 containing (invalid_flag, sanitized_type, sanitized_amount).
    """
    # Fast-path: in-range and already valid.
    if (
        action_type >= 0
        and action_type < c.NUM_ACTIONS
        and validate_action(state, env_idx, player_idx, action_type, amount)
    ):
        return _pack_action(0, action_type, amount)

    invalid = wp.int32(1)
    to_call = get_to_call(state, env_idx, player_idx)
    player_stack = state.stacks[env_idx, player_idx]

    # Normalize common mismatches.
    if action_type == c.ACTION_CHECK:
        if to_call > 0:
            return _pack_action(invalid, c.ACTION_CALL, 0)
        return _pack_action(invalid, c.ACTION_CHECK, 0)

    if action_type == c.ACTION_CALL:
        if to_call == 0:
            return _pack_action(invalid, c.ACTION_CHECK, 0)
        return _pack_action(invalid, c.ACTION_CALL, 0)

    if action_type == c.ACTION_RAISE:
        # Can't raise if we can't even cover the call.
        if player_stack <= to_call:
            if to_call > 0:
                return _pack_action(invalid, c.ACTION_CALL, 0)
            return _pack_action(invalid, c.ACTION_CHECK, 0)

        min_d = _min_raise_delta(state, env_idx)
        max_d = _max_raise_delta(state, env_idx, player_idx)
        all_in_delta = player_stack - to_call

        # If opponent can't match any raise (or we can't), fall back.
        if max_d <= 0:
            if to_call > 0:
                return _pack_action(invalid, c.ACTION_CALL, 0)
            return _pack_action(invalid, c.ACTION_CHECK, 0)

        # If we can't make a legal raise size (and we're not all-in), raising is not allowed.
        if max_d < min_d:
            if all_in_delta > 0 and all_in_delta <= max_d:
                return _pack_action(invalid, c.ACTION_RAISE, all_in_delta)
            if to_call > 0:
                return _pack_action(invalid, c.ACTION_CALL, 0)
            return _pack_action(invalid, c.ACTION_CHECK, 0)

        a = amount
        if a > max_d:
            a = max_d

        if a < min_d:
            # We've already returned for the short-stack case where max_d < min_d
            # (line above), so min_d is always a legal raise size here. Clamp UP
            # to min_d rather than substituting an all-in shove, which would
            # silently turn a tiny illegal bet into a stack-pushing action.
            if min_d <= max_d:
                a = min_d
            elif all_in_delta > 0 and all_in_delta <= max_d:
                a = all_in_delta
            else:
                a = max_d

        if a <= 0:
            if to_call > 0:
                return _pack_action(invalid, c.ACTION_CALL, 0)
            return _pack_action(invalid, c.ACTION_CHECK, 0)

        return _pack_action(invalid, c.ACTION_RAISE, a)

    if action_type == c.ACTION_FOLD:
        return _pack_action(invalid, c.ACTION_FOLD, 0)

    return _pack_action(invalid, c.ACTION_FOLD, 0)


@wp.func
def apply_action(
    state: GameState,
    env_idx: wp.int32,
    player_idx: wp.int32,
    action_type: wp.int32,
    amount: wp.int32,
) -> wp.int32:
    """
    Updates game state based on the action.
    Assumes action is validated or forced valid before calling.
    """
    if action_type == c.ACTION_FOLD:
        state.done[env_idx] = True
        return 0

    if action_type == c.ACTION_CHECK:
        return 0

    if action_type == c.ACTION_CALL:
        to_call = get_to_call(state, env_idx, player_idx)
        actual_bet = execute_bet(state, env_idx, player_idx, to_call)
        state.last_aggressor[env_idx] = -1  # Call ends aggression usually
        return actual_bet

    if action_type == c.ACTION_RAISE:
        to_call = get_to_call(state, env_idx, player_idx)
        total_bet = to_call + amount

        # Effective-stack cap (HU simplification): cap resulting total bet.
        opp = get_opponent(player_idx)
        current_bet = state.bets[env_idx, player_idx]
        desired_total = current_bet + total_bet

        opp_max_total = state.bets[env_idx, opp] + state.stacks[env_idx, opp]
        if desired_total > opp_max_total:
            desired_total = opp_max_total

        player_max_total = current_bet + state.stacks[env_idx, player_idx]
        if desired_total > player_max_total:
            desired_total = player_max_total

        total_bet = desired_total - current_bet
        if total_bet < 0:
            total_bet = 0

        actual_bet = execute_bet(state, env_idx, player_idx, total_bet)

        actual_raise = actual_bet - to_call
        if actual_raise > 0:
            # Standard NLHE: an all-in raise of less than a full raise size does
            # NOT update the min-raise reference (it does not reopen action for
            # players already past). Only "full" raises (>= prior min) update
            # last_raise. Always advance aggressor and counter.
            prior_min = state.last_raise[env_idx]
            bb = state.cfg_big_blind[env_idx]
            if bb <= 0:
                bb = c.BIG_BLIND
            if prior_min < bb:
                prior_min = bb
            if actual_raise >= prior_min:
                state.last_raise[env_idx] = actual_raise
            state.last_aggressor[env_idx] = player_idx
            state.num_raises[env_idx] = state.num_raises[env_idx] + 1
        return actual_bet

    return 0
