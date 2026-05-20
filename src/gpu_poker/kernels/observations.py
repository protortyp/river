import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import actions
from gpu_poker.struct_types import GameState


@wp.func
def get_visible_card(
    state: GameState,
    env_idx: wp.int32,
    card_slot: wp.int32,  # 0-1 (hole), 2-6 (board)
) -> wp.int32:
    """
    Returns the card index if it should be visible to the agent, else -1.
    """
    player = state.active_player[env_idx]

    # Hole Cards (0, 1)
    if card_slot < 2:
        return state.hole_cards[env_idx, player, card_slot]

    # Community Cards (2, 3, 4, 5, 6)
    board_idx = card_slot - 2

    # Check if this board card is dealt based on stage
    current_stage = state.stage[env_idx]

    # Flop (indices 0,1,2) requires stage >= FLOP
    if board_idx < 3:
        if current_stage >= c.STAGE_FLOP:
            return state.community_cards[env_idx, board_idx]

    # Turn (index 3) requires stage >= TURN
    elif board_idx == 3:
        if current_stage >= c.STAGE_TURN:
            return state.community_cards[env_idx, board_idx]

    # River (index 4) requires stage >= RIVER
    elif board_idx == 4 and current_stage >= c.STAGE_RIVER:
        return state.community_cards[env_idx, board_idx]

    return c.INVALID_CARD


@wp.func
def normalize(val: wp.int32, max_val: wp.int32) -> wp.float32:
    return wp.float32(val) / wp.float32(max_val)


@wp.func
def write_observation(
    state: GameState,
    env_idx: wp.int32,
    obs_cards: wp.array(dtype=wp.int32, ndim=2),
    obs_scalars: wp.array(dtype=wp.float32, ndim=2),
    primes: wp.array(dtype=wp.int32),
    unsuited_lut: wp.array(dtype=wp.int16),
    flush_lut: wp.array(dtype=wp.int16),
):
    """
    Extracts observation for the active player and writes directly to output arrays.
    """
    player = state.active_player[env_idx]
    opp = 1 - player

    starting_stack = state.cfg_starting_stack[env_idx]
    if starting_stack <= 0:
        starting_stack = c.STARTING_STACK
    big_blind = state.cfg_big_blind[env_idx]
    if big_blind <= 0:
        big_blind = c.BIG_BLIND

    # Cards (7): 2 hole + 5 community (or -1 if not visible)
    for i in range(7):
        obs_cards[env_idx, i] = get_visible_card(state, env_idx, i)

    # Scalars (existing normalized features + a few additions at the end).

    # 0: Position (0=Button/SB, 1=BB)
    if player == state.button[env_idx]:
        obs_scalars[env_idx, 0] = 0.0
    else:
        obs_scalars[env_idx, 0] = 1.0

    # 1-2: Stacks (normalized by starting stack)
    obs_scalars[env_idx, 1] = normalize(state.stacks[env_idx, player], starting_stack)
    obs_scalars[env_idx, 2] = normalize(state.stacks[env_idx, opp], starting_stack)

    # 3-4: Current Bets (normalized by pot)
    pot_total = state.pot[env_idx] + state.bets[env_idx, 0] + state.bets[env_idx, 1]
    if pot_total > 0:
        obs_scalars[env_idx, 3] = wp.float32(state.bets[env_idx, player]) / wp.float32(pot_total)
        obs_scalars[env_idx, 4] = wp.float32(state.bets[env_idx, opp]) / wp.float32(pot_total)
    else:
        obs_scalars[env_idx, 3] = 0.0
        obs_scalars[env_idx, 4] = 0.0

    # 5: Pot Size (normalized)
    obs_scalars[env_idx, 5] = normalize(pot_total, starting_stack * 2)

    # 6: To Call (normalized by stack, clipped to [0, 1]).
    # When to_call > stack the actor can only call all-in for less, so the
    # fraction of stack committed is at most 1.
    to_call = actions.get_to_call(state, env_idx, player)
    player_stack = state.stacks[env_idx, player]
    if player_stack > 0:
        effective_to_call = to_call
        if effective_to_call > player_stack:
            effective_to_call = player_stack
        obs_scalars[env_idx, 6] = normalize(effective_to_call, player_stack)
    else:
        obs_scalars[env_idx, 6] = 0.0

    # 7: Stage (normalized: 0..1)
    obs_scalars[env_idx, 7] = wp.float32(state.stage[env_idx]) / 5.0

    # 8: Last Raise Size (normalized)
    obs_scalars[env_idx, 8] = normalize(state.last_raise[env_idx], starting_stack)

    # 9: Number of Raises
    obs_scalars[env_idx, 9] = wp.float32(state.num_raises[env_idx]) / 4.0  # Cap at 4

    # 10: Last Aggressor Indicator (0=none, 0.5=opp, 1.0=me)
    if state.last_aggressor[env_idx] == player:
        obs_scalars[env_idx, 10] = 1.0
    elif state.last_aggressor[env_idx] == opp:
        obs_scalars[env_idx, 10] = 0.5
    else:
        obs_scalars[env_idx, 10] = 0.0

    # --- Public history scalars (lightweight, complements LSTM memory) ---

    # 11: Actions this street (normalized, hard-capped)
    actions_this_street = state.actions_this_street[env_idx]
    if actions_this_street < 0:
        actions_this_street = 0
    if actions_this_street > 15:
        actions_this_street = 15
    obs_scalars[env_idx, 11] = wp.float32(actions_this_street) / 15.0

    # 12: Last action type (0.0 means "none yet", otherwise (type+1)/4)
    lat = state.last_action_type[env_idx]
    if lat < 0:
        obs_scalars[env_idx, 12] = 0.0
    else:
        # type in {0..3} -> {0.25, 0.5, 0.75, 1.0}
        obs_scalars[env_idx, 12] = (wp.float32(lat) + 1.0) / 4.0

    # 13: Last action was raise (0/1)
    obs_scalars[env_idx, 13] = wp.float32(state.last_action_was_raise[env_idx])

    # 14: Last action amount (normalized, clamped)
    amt = state.last_action_amount[env_idx]
    if amt < 0:
        amt = 0
    amt_f = normalize(amt, starting_stack)
    if amt_f > 1.0:
        amt_f = 1.0
    obs_scalars[env_idx, 14] = amt_f

    # 15: big blind relative to starting stack (helps if we later vary currency)
    obs_scalars[env_idx, 15] = wp.float32(big_blind) / wp.float32(starting_stack)

    # 16: effective stack depth in bb (normalized/clipped)
    eff = state.stacks[env_idx, 0]
    if state.stacks[env_idx, 1] < eff:
        eff = state.stacks[env_idx, 1]
    eff_bb = wp.float32(eff) / wp.float32(big_blind) if big_blind > 0 else 0.0
    # Normalize to [0,1] by dividing by a soft max depth (200bb) and clipping.
    eff_bb_norm = eff_bb / 200.0
    if eff_bb_norm > 1.0:
        eff_bb_norm = 1.0
    obs_scalars[env_idx, 16] = eff_bb_norm
