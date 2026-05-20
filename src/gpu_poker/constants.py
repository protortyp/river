"""
Core constants and enumerations for GPU poker environment.

All constants use Warp-compatible types (wp.int32, wp.float32).
"""

import warp as wp

# =============================================================================
# Card Representation
# =============================================================================

# Number of ranks (2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A)
NUM_RANKS: int = 13

# Number of suits (Clubs, Diamonds, Hearts, Spades)
NUM_SUITS: int = 4

# Total number of cards in deck
NUM_CARDS: int = 52

# Maximum cards per player (2 hole cards)
HOLE_CARDS: int = 2

# Maximum community cards (5: flop + turn + river)
COMMUNITY_CARDS: int = 5

# Cards needed for evaluation (hole + community)
MAX_EVAL_CARDS: int = 7

# =============================================================================
# Rank Constants (0-12)
# =============================================================================

RANK_2: int = 0
RANK_3: int = 1
RANK_4: int = 2
RANK_5: int = 3
RANK_6: int = 4
RANK_7: int = 5
RANK_8: int = 6
RANK_9: int = 7
RANK_T: int = 8
RANK_J: int = 9
RANK_Q: int = 10
RANK_K: int = 11
RANK_A: int = 12

# =============================================================================
# Suit Constants (0-3)
# =============================================================================

SUIT_CLUBS: int = 0
SUIT_DIAMONDS: int = 1
SUIT_HEARTS: int = 2
SUIT_SPADES: int = 3

# =============================================================================
# Game Stage Constants
# =============================================================================

STAGE_PREFLOP: int = 0
STAGE_FLOP: int = 1
STAGE_TURN: int = 2
STAGE_RIVER: int = 3
STAGE_SHOWDOWN: int = 4
STAGE_TERMINAL: int = 5

NUM_STAGES: int = 6

# =============================================================================
# Action Constants
# =============================================================================

ACTION_FOLD: int = 0
ACTION_CHECK: int = 1
ACTION_CALL: int = 2
ACTION_RAISE: int = 3

NUM_ACTIONS: int = 4

# =============================================================================
# Player Constants
# =============================================================================

# Number of players in heads-up
NUM_PLAYERS: int = 2

PLAYER_0: int = 0
PLAYER_1: int = 1

# Player positions
POSITION_SB: int = 0  # Small blind / button in heads-up
POSITION_BB: int = 1  # Big blind

# Invalid player marker
INVALID_PLAYER: int = -1

# =============================================================================
# Betting Constants
# =============================================================================

# Default stack size in big blinds
DEFAULT_STACK_BB: int = 100

# Blind amounts (in chips)
SMALL_BLIND: int = 1
BIG_BLIND: int = 2

# Starting stack (in chips)
STARTING_STACK: int = 1000

# Minimum raise amount (must raise at least the previous raise amount)
MIN_RAISE_MULTIPLIER: int = 2

# =============================================================================
# Array Shape Constants
# =============================================================================

# Default number of parallel environments
DEFAULT_NUM_ENVS: int = 131072  # 128K environments

# Maximum number of actions per hand (conservative estimate)
MAX_ACTIONS_PER_HAND: int = 100

# =============================================================================
# Warp Type Aliases
# =============================================================================

# Integer types
Int32 = wp.int32
UInt32 = wp.uint32
Int64 = wp.int64

# Float types
Float32 = wp.float32
Float64 = wp.float64

# Boolean type
Bool = wp.bool

# =============================================================================
# Hand Rank Constants (for evaluator)
# =============================================================================

HAND_HIGH_CARD: int = 0
HAND_PAIR: int = 1
HAND_TWO_PAIR: int = 2
HAND_THREE_OF_KIND: int = 3
HAND_STRAIGHT: int = 4
HAND_FLUSH: int = 5
HAND_FULL_HOUSE: int = 6
HAND_FOUR_OF_KIND: int = 7
HAND_STRAIGHT_FLUSH: int = 8

# =============================================================================
# Prime Numbers for Cactus Kev Evaluator
# =============================================================================

# Prime numbers for each rank (used in card encoding)
RANK_PRIMES: tuple[int, ...] = (
    2,  # 2
    3,  # 3
    5,  # 4
    7,  # 5
    11,  # 6
    13,  # 7
    17,  # 8
    19,  # 9
    23,  # T
    29,  # J
    31,  # Q
    37,  # K
    41,  # A
)

# =============================================================================
# Invalid/Sentinel Values
# =============================================================================

INVALID_CARD: int = -1
INVALID_ACTION: int = -1
INVALID_STAGE: int = -1
