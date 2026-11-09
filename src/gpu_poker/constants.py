"""Core constants and enumerations for the GPU poker environment."""

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

# =============================================================================
# Array Shape Constants
# =============================================================================

# Default number of parallel environments
DEFAULT_NUM_ENVS: int = 131072  # 128K environments

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
