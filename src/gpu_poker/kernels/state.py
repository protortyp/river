import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import actions, card_utils, evaluator
from gpu_poker.struct_types import GameState

# =============================================================================
# A. Card Handling & Initialization
# =============================================================================


@wp.func
def shuffle_and_deal(state: GameState, env_idx: wp.int32):
    """
    Resets the deck, shuffles it using the environment seed, and deals hole cards.
    Updates the RNG state for the next hand.
    """
    # 1. Reset deck order
    for i in range(c.NUM_CARDS):
        state.deck[env_idx, i] = i

    # 2. Shuffle
    rng = state.rng_state[env_idx]
    card_utils.shuffle_deck(state.deck[env_idx], rng)

    # 3. Update RNG for next time (simple LCG step)
    state.rng_state[env_idx] = rng * wp.uint32(1664525) + wp.uint32(1013904223)

    # 4. Deal Hole Cards
    # P0 gets indices 0,1. P1 gets 2,3.
    state.hole_cards[env_idx, c.PLAYER_0, 0] = state.deck[env_idx, 0]
    state.hole_cards[env_idx, c.PLAYER_0, 1] = state.deck[env_idx, 1]
    state.hole_cards[env_idx, c.PLAYER_1, 0] = state.deck[env_idx, 2]
    state.hole_cards[env_idx, c.PLAYER_1, 1] = state.deck[env_idx, 3]

    # Set pointer past dealt cards
    state.deck_top[env_idx] = 4


@wp.func
def deal_street(state: GameState, env_idx: wp.int32):
    """
    Deals community cards based on the *new* stage.
    """
    stage = state.stage[env_idx]
    count = 0
    current = 0

    if stage == c.STAGE_FLOP:
        current = 0
        count = 3
    elif stage == c.STAGE_TURN:
        current = 3
        count = 1
    elif stage == c.STAGE_RIVER:
        current = 4
        count = 1

    top = state.deck_top[env_idx]

    for i in range(count):
        card = state.deck[env_idx, top + i]
        state.community_cards[env_idx, current + i] = card

    state.deck_top[env_idx] = top + count
    state.num_community[env_idx] = current + count


# =============================================================================
# B. Game Rules & Transitions
# =============================================================================


@wp.func
def is_betting_settled(state: GameState, env_idx: wp.int32) -> wp.bool:
    """
    Determines if the betting round is over.
    Handles the standard rule: bets equal.
    Handles the edge case: Preflop Big Blind option.
    """
    bet0 = state.bets[env_idx, c.PLAYER_0]
    bet1 = state.bets[env_idx, c.PLAYER_1]

    # 1. Basic condition: Bets must be equal.
    # HU exception: if a player is all-in for less (i.e., they are the smaller
    # bettor), action is closed even if totals are unequal (uncalled portion is
    # returned in collect_pot()).
    if bet0 != bet1:
        if state.stacks[env_idx, 0] == 0 and bet0 < bet1:
            return True
        return bool(state.stacks[env_idx, 1] == 0 and bet1 < bet0)

    # 2. Preflop Edge Case (The BB Option)
    # If Preflop, bets equal (SB called BB), and active player is BB...
    # The round is NOT over, BB gets to Check or Raise.
    # Check if no raises have occurred (Limped pot)
    if state.stage[env_idx] == c.STAGE_PREFLOP and state.num_raises[env_idx] == 0:
        # Map position to player ID based on button
        # Heads Up: Button=0 -> SB=0, BB=1. Button=1 -> SB=1, BB=0.
        bb_player = 1 - state.button[env_idx]

        if state.active_player[env_idx] == bb_player:
            return False

    # 3. Equal bets only close the street once enough action has happened.
    # This prevents postflop "first player checks -> street ends" bugs.
    min_actions = 2
    if state.stage[env_idx] == c.STAGE_PREFLOP and state.num_raises[env_idx] > 0:
        # In a raised preflop pot, the caller can close action before the original
        # raiser acts again.
        min_actions = 1

    return state.actions_this_street[env_idx] >= min_actions


@wp.func
def rotate_button(state: GameState, env_idx: wp.int32):
    """Swaps button and determines starting active player."""
    # Toggle button 0 <-> 1
    state.button[env_idx] = 1 - state.button[env_idx]

    # Heads Up Preflop: Button is Small Blind and acts first
    state.active_player[env_idx] = state.button[env_idx]


@wp.func
def post_blinds(state: GameState, env_idx: wp.int32, sb_amount: wp.int32, bb_amount: wp.int32):
    """Posts SB and BB and initializes betting tracking."""
    sb_player = state.button[env_idx]
    bb_player = 1 - sb_player

    actions.execute_bet(state, env_idx, sb_player, sb_amount)
    actions.execute_bet(state, env_idx, bb_player, bb_amount)

    # Reset tracking
    state.last_raise[env_idx] = bb_amount  # The BB is the "raise" to match
    state.num_raises[env_idx] = 0
    state.last_aggressor[env_idx] = c.INVALID_PLAYER  # No aggressive action (raise) yet


# =============================================================================
# C. State Mutation & Payouts
# =============================================================================


@wp.func
def collect_pot(state: GameState, env_idx: wp.int32):
    """
    Sweeps current bets into the main pot and resets round trackers.

    Also handles HU "uncalled bet" situations (typically when a player is all-in
    for less): the unmatched portion of the larger bet is returned to that
    player's stack before collecting.
    """
    bet0 = state.bets[env_idx, 0]
    bet1 = state.bets[env_idx, 1]

    if bet0 != bet1:
        # Return the uncalled portion to the player with the larger bet.
        if bet0 > bet1:
            uncalled = bet0 - bet1
            state.stacks[env_idx, 0] = state.stacks[env_idx, 0] + uncalled
            bet0 = bet1
        else:
            uncalled = bet1 - bet0
            state.stacks[env_idx, 1] = state.stacks[env_idx, 1] + uncalled
            bet1 = bet0

    state.pot[env_idx] = state.pot[env_idx] + bet0 + bet1
    state.bets[env_idx, 0] = 0
    state.bets[env_idx, 1] = 0

    state.last_raise[env_idx] = 0
    state.num_raises[env_idx] = 0
    state.last_aggressor[env_idx] = c.INVALID_PLAYER


@wp.func
def calculate_rewards(state: GameState, env_idx: wp.int32, winner_idx: wp.int32) -> wp.float32:
    """
    Calculates exact zero-sum rewards based on stack changes.
    Reward = (Final Stack + PotShare) - Initial Stack.

    Args:
        winner_idx: 0 or 1 for winner, -1 for split pot.
    """
    total_pot = state.pot[env_idx]  # Includes bets from current round if collected

    # Calculate Pot Share
    p0_share = 0

    if winner_idx == c.PLAYER_0:
        p0_share = total_pot
    elif winner_idx == c.PLAYER_1:
        pass
    else:  # Split
        # HU odd chip rule: award the odd chip to the player closest to the button.
        # In heads-up, the button is also the small blind.
        p0_share = total_pot / 2
        if (total_pot & 1) == 1 and state.button[env_idx] == c.PLAYER_0:
            p0_share = p0_share + 1

    # Calculate Reward: What I ended up with MINUS what I started with
    # Note: state.stacks is currently (Initial - Invested)
    p0_final = state.stacks[env_idx, 0] + p0_share
    p0_initial = state.initial_stacks[env_idx, 0]

    reward_p0 = wp.float32(p0_final - p0_initial)
    return reward_p0


@wp.func
def resolve_showdown(
    state: GameState,
    env_idx: wp.int32,
    primes: wp.array(dtype=wp.int32),
    unsuited_lut: wp.array(dtype=wp.int16),
    flush_lut: wp.array(dtype=wp.int16),
) -> wp.float32:
    """Evaluates hands and calculates payout."""

    score0 = evaluator.eval_7(
        state.hole_cards[env_idx, 0],
        state.community_cards[env_idx],
        flush_lut,
        unsuited_lut,
        primes,
    )
    score1 = evaluator.eval_7(
        state.hole_cards[env_idx, 1],
        state.community_cards[env_idx],
        flush_lut,
        unsuited_lut,
        primes,
    )

    winner = -1
    if score0 > score1:
        winner = c.PLAYER_0
    elif score1 > score0:
        winner = c.PLAYER_1

    return calculate_rewards(state, env_idx, winner)


# =============================================================================
# D. Main Lifecycle Functions
# =============================================================================


@wp.func
def reset_env(
    state: GameState,
    env_idx: wp.int32,
    starting_stack: wp.int32,
    small_blind: wp.int32,
    big_blind: wp.int32,
):
    """
    Master reset function. Called at initialization and after terminal states.
    """
    # 1. Reset Scalar State
    state.stage[env_idx] = c.STAGE_PREFLOP
    state.done[env_idx] = False
    state.pot[env_idx] = 0
    state.num_community[env_idx] = 0
    state.num_actions[env_idx] = 0
    state.actions_this_street[env_idx] = 0
    state.last_action_type[env_idx] = c.INVALID_ACTION
    state.last_action_amount[env_idx] = 0
    state.last_action_was_raise[env_idx] = 0

    # 1b. Persist per-env configuration for kernels/observations.
    state.cfg_starting_stack[env_idx] = starting_stack
    state.cfg_small_blind[env_idx] = small_blind
    state.cfg_big_blind[env_idx] = big_blind

    # 2. Reset Stacks & Snapshot
    state.stacks[env_idx, 0] = starting_stack
    state.stacks[env_idx, 1] = starting_stack
    state.initial_stacks[env_idx, 0] = starting_stack
    state.initial_stacks[env_idx, 1] = starting_stack
    state.bets[env_idx, 0] = 0
    state.bets[env_idx, 1] = 0

    # 3. Clean Board
    for i in range(c.COMMUNITY_CARDS):
        state.community_cards[env_idx, i] = c.INVALID_CARD

    # 4. Rotate & Setup
    rotate_button(state, env_idx)
    shuffle_and_deal(state, env_idx)
    post_blinds(state, env_idx, small_blind, big_blind)

    # 5. Episode tracking
    state.episode_id[env_idx] = state.episode_id[env_idx] + 1


@wp.func
def _auto_reset(
    state: GameState,
    env_idx: wp.int32,
    starting_stack: wp.int32,
    small_blind: wp.int32,
    big_blind: wp.int32,
) -> wp.float32:
    state.done[env_idx] = True
    reset_env(state, env_idx, starting_stack, small_blind, big_blind)
    return 0.0


@wp.func
def _handle_fold(
    state: GameState,
    env_idx: wp.int32,
    folding_player: wp.int32,
    starting_stack: wp.int32,
    small_blind: wp.int32,
    big_blind: wp.int32,
) -> wp.float32:
    collect_pot(state, env_idx)
    winner = 1 - folding_player
    reward = calculate_rewards(state, env_idx, winner)
    state.done[env_idx] = True
    reset_env(state, env_idx, starting_stack, small_blind, big_blind)
    return reward


@wp.func
def _maybe_advance_round_or_resolve(
    state: GameState,
    env_idx: wp.int32,
    primes: wp.array(dtype=wp.int32),
    unsuited_lut: wp.array(dtype=wp.int16),
    flush_lut: wp.array(dtype=wp.int16),
    starting_stack: wp.int32,
    small_blind: wp.int32,
    big_blind: wp.int32,
) -> wp.float32:
    if not is_betting_settled(state, env_idx):
        return 0.0

    collect_pot(state, env_idx)
    state.actions_this_street[env_idx] = 0

    is_all_in = (state.stacks[env_idx, 0] == 0) or (state.stacks[env_idx, 1] == 0)

    state.stage[env_idx] = state.stage[env_idx] + 1

    # Loop until showdown if All-In, or just once if normal play.
    while state.stage[env_idx] <= c.STAGE_SHOWDOWN:
        if state.stage[env_idx] == c.STAGE_SHOWDOWN:
            reward = resolve_showdown(state, env_idx, primes, unsuited_lut, flush_lut)
            state.done[env_idx] = True
            reset_env(state, env_idx, starting_stack, small_blind, big_blind)
            return reward

        deal_street(state, env_idx)

        if not is_all_in:
            # Heads-up postflop: the non-button / big blind acts first.
            state.active_player[env_idx] = 1 - state.button[env_idx]
            break

        state.stage[env_idx] = state.stage[env_idx] + 1

    return 0.0


@wp.func
def step(
    state: GameState,
    env_idx: wp.int32,
    action_type: wp.int32,
    action_amount: wp.int32,
    primes: wp.array(dtype=wp.int32),
    unsuited_lut: wp.array(dtype=wp.int16),
    flush_lut: wp.array(dtype=wp.int16),
    starting_stack: wp.int32,
    small_blind: wp.int32,
    big_blind: wp.int32,
) -> wp.float32:
    """
    Main entry point for one environment step.
    Handles Action -> Transition -> Resolution -> Auto-Reset.
    Returns P0 reward.
    """

    # 0. Safety Check (Idempotency)
    if state.done[env_idx]:
        return _auto_reset(state, env_idx, starting_stack, small_blind, big_blind)

    player = state.active_player[env_idx]

    state.num_actions[env_idx] = state.num_actions[env_idx] + 1
    state.actions_this_street[env_idx] = state.actions_this_street[env_idx] + 1

    # 1. Handle FOLD (Immediate Termination)
    if action_type == c.ACTION_FOLD:
        state.last_action_type[env_idx] = action_type
        state.last_action_amount[env_idx] = 0
        state.last_action_was_raise[env_idx] = 0
        return _handle_fold(state, env_idx, player, starting_stack, small_blind, big_blind)

    # 2. Handle Game Actions (Check/Call/Raise)
    # Assumes action is valid.
    actual_bet = actions.apply_action(state, env_idx, player, action_type, action_amount)
    state.last_action_type[env_idx] = action_type
    state.last_action_amount[env_idx] = actual_bet
    if action_type == c.ACTION_RAISE:
        state.last_action_was_raise[env_idx] = 1
    else:
        state.last_action_was_raise[env_idx] = 0

    # 3. Check for Round Transition
    # IMPORTANT: We switch the active player *before* checking if the round is settled.
    # This is because `is_betting_settled` checks the state for the player *about* to act.
    state.active_player[env_idx] = 1 - player

    return _maybe_advance_round_or_resolve(
        state,
        env_idx,
        primes,
        unsuited_lut,
        flush_lut,
        starting_stack,
        small_blind,
        big_blind,
    )
