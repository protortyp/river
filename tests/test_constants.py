"""Unit tests for constants module."""

from gpu_poker import constants


class TestCardConstants:
    """Test card-related constants."""

    def test_deck_size(self):
        """Verify standard 52-card deck."""
        assert constants.NUM_CARDS == 52
        assert constants.NUM_RANKS == 13
        assert constants.NUM_SUITS == 4
        assert constants.NUM_RANKS * constants.NUM_SUITS == constants.NUM_CARDS

    def test_hand_sizes(self):
        """Verify hand size constants."""
        assert constants.HOLE_CARDS == 2
        assert constants.COMMUNITY_CARDS == 5
        assert constants.MAX_EVAL_CARDS == 7
        assert constants.HOLE_CARDS + constants.COMMUNITY_CARDS == constants.MAX_EVAL_CARDS


class TestRankConstants:
    """Test rank enumeration."""

    def test_rank_range(self):
        """Verify ranks are 0-12."""
        assert constants.RANK_2 == 0
        assert constants.RANK_A == 12
        assert constants.NUM_RANKS == 13

    def test_rank_ordering(self):
        """Verify ranks are sequential."""
        ranks = [
            constants.RANK_2,
            constants.RANK_3,
            constants.RANK_4,
            constants.RANK_5,
            constants.RANK_6,
            constants.RANK_7,
            constants.RANK_8,
            constants.RANK_9,
            constants.RANK_T,
            constants.RANK_J,
            constants.RANK_Q,
            constants.RANK_K,
            constants.RANK_A,
        ]
        assert ranks == list(range(13))


class TestSuitConstants:
    """Test suit enumeration."""

    def test_suit_range(self):
        """Verify suits are 0-3."""
        assert constants.SUIT_CLUBS == 0
        assert constants.SUIT_SPADES == 3
        assert constants.NUM_SUITS == 4

    def test_suit_ordering(self):
        """Verify suits are sequential."""
        suits = [
            constants.SUIT_CLUBS,
            constants.SUIT_DIAMONDS,
            constants.SUIT_HEARTS,
            constants.SUIT_SPADES,
        ]
        assert suits == list(range(4))


class TestGameStageConstants:
    """Test game stage enumeration."""

    def test_stage_range(self):
        """Verify stage values."""
        assert constants.STAGE_PREFLOP == 0
        assert constants.STAGE_TERMINAL == 5
        assert constants.NUM_STAGES == 6

    def test_stage_ordering(self):
        """Verify stages are sequential."""
        stages = [
            constants.STAGE_PREFLOP,
            constants.STAGE_FLOP,
            constants.STAGE_TURN,
            constants.STAGE_RIVER,
            constants.STAGE_SHOWDOWN,
            constants.STAGE_TERMINAL,
        ]
        assert stages == list(range(6))


class TestActionConstants:
    """Test action enumeration."""

    def test_action_range(self):
        """Verify action values."""
        assert constants.ACTION_FOLD == 0
        assert constants.ACTION_RAISE == 3
        assert constants.NUM_ACTIONS == 4

    def test_action_ordering(self):
        """Verify actions are sequential."""
        actions = [
            constants.ACTION_FOLD,
            constants.ACTION_CHECK,
            constants.ACTION_CALL,
            constants.ACTION_RAISE,
        ]
        assert actions == list(range(4))


class TestPlayerConstants:
    """Test player-related constants."""

    def test_heads_up(self):
        """Verify heads-up setup."""
        assert constants.NUM_PLAYERS == 2
        assert constants.PLAYER_0 == 0
        assert constants.PLAYER_1 == 1

    def test_positions(self):
        """Verify position constants."""
        assert constants.POSITION_SB == 0
        assert constants.POSITION_BB == 1

    def test_invalid_player(self):
        """Verify invalid player marker."""
        assert constants.INVALID_PLAYER == -1
        assert constants.INVALID_PLAYER < 0


class TestBettingConstants:
    """Test betting-related constants."""

    def test_blind_structure(self):
        """Verify blind amounts."""
        assert constants.SMALL_BLIND == 1
        assert constants.BIG_BLIND == 2
        assert constants.BIG_BLIND == 2 * constants.SMALL_BLIND

    def test_stack_size(self):
        """Verify default stack size."""
        assert constants.DEFAULT_STACK_BB == 100
        assert constants.DEFAULT_STACK_BB > 0

    def test_raise_multiplier(self):
        """Verify minimum raise constraint."""
        assert constants.MIN_RAISE_MULTIPLIER == 2


class TestHandRankConstants:
    """Test hand ranking enumeration."""

    def test_hand_rank_range(self):
        """Verify hand ranks are 0-8."""
        assert constants.HAND_HIGH_CARD == 0
        assert constants.HAND_STRAIGHT_FLUSH == 8

    def test_hand_rank_ordering(self):
        """Verify hand ranks are ordered by strength."""
        hand_ranks = [
            constants.HAND_HIGH_CARD,
            constants.HAND_PAIR,
            constants.HAND_TWO_PAIR,
            constants.HAND_THREE_OF_KIND,
            constants.HAND_STRAIGHT,
            constants.HAND_FLUSH,
            constants.HAND_FULL_HOUSE,
            constants.HAND_FOUR_OF_KIND,
            constants.HAND_STRAIGHT_FLUSH,
        ]
        assert hand_ranks == list(range(9))


class TestPrimes:
    """Test prime number constants for evaluator."""

    def test_prime_count(self):
        """Verify we have primes for all ranks."""
        assert len(constants.RANK_PRIMES) == constants.NUM_RANKS

    def test_primes_are_prime(self):
        """Verify all rank primes are actually prime."""

        def is_prime(n):
            if n < 2:
                return False
            return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

        for prime in constants.RANK_PRIMES:
            assert is_prime(prime), f"{prime} is not prime"

    def test_primes_are_unique(self):
        """Verify all primes are distinct."""
        assert len(constants.RANK_PRIMES) == len(set(constants.RANK_PRIMES))

    def test_primes_ascending(self):
        """Verify primes are in ascending order."""
        assert list(constants.RANK_PRIMES) == sorted(constants.RANK_PRIMES)


class TestInvalidSentinels:
    """Test invalid/sentinel values."""

    def test_invalid_values_negative(self):
        """Verify invalid markers are negative."""
        assert constants.INVALID_CARD < 0
        assert constants.INVALID_ACTION < 0
        assert constants.INVALID_STAGE < 0
        assert constants.INVALID_PLAYER < 0

    def test_invalid_values_unique(self):
        """Verify invalid markers don't conflict with valid values."""
        assert constants.INVALID_CARD not in range(constants.NUM_CARDS)
        assert constants.INVALID_ACTION not in range(constants.NUM_ACTIONS)
        assert constants.INVALID_STAGE not in range(constants.NUM_STAGES)
        assert constants.INVALID_PLAYER not in range(constants.NUM_PLAYERS)


class TestArrayShapeConstants:
    """Test array shape and sizing constants."""

    def test_default_env_count(self):
        """Verify default environment count is power of 2."""
        assert constants.DEFAULT_NUM_ENVS == 131072  # 128K
        assert constants.DEFAULT_NUM_ENVS & (constants.DEFAULT_NUM_ENVS - 1) == 0  # Power of 2

    def test_max_actions_reasonable(self):
        """Verify max actions per hand is reasonable."""
        assert constants.MAX_ACTIONS_PER_HAND == 100
        assert constants.MAX_ACTIONS_PER_HAND > 0
