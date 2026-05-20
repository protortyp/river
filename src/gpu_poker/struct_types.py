"""
Warp struct definitions for GPU poker environment.

Note: While we define structs here for clarity, the actual memory layout in the
environment uses Structure of Arrays (SoA) pattern with separate arrays per field,
all indexed by environment ID.

Shape convention: [N_ENVS, ...] for all state arrays.
"""

import warp as wp

# =============================================================================
# Game State Struct
# =============================================================================


@wp.struct
class GameState:
    """
    Complete state for GPU poker environment using SoA (Structure of Arrays) layout.

    This struct holds references to GLOBAL arrays indexed by environment ID.
    All fields are arrays with the first dimension being N_ENVS.

    This enables efficient parallel processing of thousands of environments on GPU.

    Access pattern in kernels:
    - state.pot[env_idx] instead of state.pot
    - state.stacks[env_idx, player_idx] instead of state.stacks[player_idx]
    """

    # Game flow arrays [N_ENVS]
    stage: wp.array(dtype=wp.int32)  # Current stage per environment
    button: wp.array(dtype=wp.int32)  # Button position (0 or 1)
    active_player: wp.array(dtype=wp.int32)  # Player to act (-1 if none)
    done: wp.array(dtype=wp.bool)  # Episode terminated flag

    # Deck arrays
    deck: wp.array(dtype=wp.int32, ndim=2)  # [N_ENVS, NUM_CARDS] shuffled card indices
    deck_top: wp.array(dtype=wp.int32)  # [N_ENVS] next card to deal

    # Card arrays
    hole_cards: wp.array(dtype=wp.int32, ndim=3)  # [N_ENVS, NUM_PLAYERS, HOLE_CARDS]
    community_cards: wp.array(dtype=wp.int32, ndim=2)  # [N_ENVS, COMMUNITY_CARDS]
    num_community: wp.array(dtype=wp.int32)  # [N_ENVS] cards dealt so far

    # Chip arrays
    stacks: wp.array(dtype=wp.int32, ndim=2)  # [N_ENVS, NUM_PLAYERS]
    bets: wp.array(dtype=wp.int32, ndim=2)  # [N_ENVS, NUM_PLAYERS] current round bets
    initial_stacks: wp.array(dtype=wp.int32, ndim=2)  # [N_ENVS, NUM_PLAYERS] snapshot at hand start
    pot: wp.array(dtype=wp.int32)  # [N_ENVS] chips in pot from previous rounds

    # Betting round state arrays [N_ENVS]
    last_raise: wp.array(dtype=wp.int32)  # Size of last raise
    num_raises: wp.array(dtype=wp.int32)  # Number of raises in current round
    last_aggressor: wp.array(dtype=wp.int32)  # Player who last raised/bet (-1 if none)

    # Action tracking [N_ENVS]
    num_actions: wp.array(dtype=wp.int32)  # Total actions taken this hand
    actions_this_street: wp.array(dtype=wp.int32)  # Actions taken in current betting round
    last_action_type: wp.array(dtype=wp.int32)  # Last action type (0..3), or -1 if none
    last_action_amount: wp.array(dtype=wp.int32)  # Chips committed by last action (0 if none)
    last_action_was_raise: wp.array(dtype=wp.int32)  # 1 if last action was a raise

    # RNG state [N_ENVS]
    rng_state: wp.array(dtype=wp.uint32)  # Random seed per environment

    # Episode tracking [N_ENVS]
    episode_id: wp.array(dtype=wp.int32)  # Monotonic counter, increments on reset

    # Per-env configuration [N_ENVS]
    cfg_starting_stack: wp.array(dtype=wp.int32)
    cfg_small_blind: wp.array(dtype=wp.int32)
    cfg_big_blind: wp.array(dtype=wp.int32)


# =============================================================================
# Action Struct
# =============================================================================


@wp.struct
class Action:
    """
    Represents a poker action.

    action_type: ACTION_FOLD, ACTION_CHECK, ACTION_CALL, or ACTION_RAISE
    amount: Raise amount (only used if action_type == ACTION_RAISE)
    """

    action_type: wp.int32
    amount: wp.int32


# =============================================================================
# Observation Struct
# =============================================================================


@wp.struct
class Observation:
    """
    Observation for RL agent (single player perspective).

    This will be flattened into a 1D tensor for neural network input.
    Fields are normalized to [0, 1] or [-1, 1] ranges.
    """

    # Player hole cards (2 cards, each encoded as one-hot or normalized)
    hole_cards: wp.array(dtype=wp.int32, ndim=1)  # Shape: [HOLE_CARDS]

    # Community cards (5 cards, -1 if not dealt yet)
    community_cards: wp.array(dtype=wp.int32, ndim=1)  # Shape: [COMMUNITY_CARDS]

    # Normalized stack sizes [0, 1] (divided by starting stack)
    stack_p0: wp.float32  # Current player stack
    stack_p1: wp.float32  # Opponent stack

    # Normalized pot size [0, 1]
    pot_normalized: wp.float32

    # Normalized current bets [0, 1]
    bet_p0: wp.float32  # Current player bet
    bet_p1: wp.float32  # Opponent bet

    # Stage encoding (one-hot in practice, or just the integer)
    stage: wp.int32

    # Position encoding
    position: wp.int32  # 0 = button/SB, 1 = BB

    # Legal action mask (1 = legal, 0 = illegal)
    legal_fold: wp.bool
    legal_check: wp.bool
    legal_call: wp.bool
    legal_raise: wp.bool

    # Raise size bounds (normalized)
    min_raise: wp.float32
    max_raise: wp.float32


# =============================================================================
# Reward/Done Struct (for RL interface)
# =============================================================================


@wp.struct
class StepResult:
    """
    Result of environment.step() for a single environment.
    """

    # Reward for each player (zero-sum: reward_p0 = -reward_p1)
    reward: wp.array(dtype=wp.float32, ndim=1)  # Shape: [NUM_PLAYERS]

    # Done flag
    done: wp.bool

    # Info (optional, can add more fields as needed)
    winner: wp.int32  # -1 if tie, 0/1 if player won


# =============================================================================
# Memory Layout Documentation
# =============================================================================

# SoA Layout for N_ENVS parallel environments:
#
# Game flow arrays:
# - stage[N_ENVS]: int32
# - button[N_ENVS]: int32
# - active_player[N_ENVS]: int32
# - done[N_ENVS]: bool
#
# Deck arrays:
# - deck[N_ENVS, NUM_CARDS]: int32 (shuffled card indices)
# - deck_top[N_ENVS]: int32
#
# Card arrays:
# - hole_cards[N_ENVS, NUM_PLAYERS, HOLE_CARDS]: int32
# - community_cards[N_ENVS, COMMUNITY_CARDS]: int32
# - num_community[N_ENVS]: int32
#
# Chip arrays:
# - stacks[N_ENVS, NUM_PLAYERS]: int32
# - bets[N_ENVS, NUM_PLAYERS]: int32
# - pot[N_ENVS]: int32
#
# Betting state arrays:
# - last_raise[N_ENVS]: int32
# - num_raises[N_ENVS]: int32
# - last_aggressor[N_ENVS]: int32
#
# Action tracking:
# - num_actions[N_ENVS]: int32
#
# RNG state:
# - rng_state[N_ENVS]: uint32
