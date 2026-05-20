"""Tests for interactive poker game utilities."""

from __future__ import annotations

import pytest
import torch
from play.interactive import (
    format_card,
    format_cards,
    get_legal_actions,
    parse_human_input,
    parse_observation_cards,
)
from rich.text import Text

from gpu_poker import constants as c


class TestCardFormatting:
    """Test card formatting utilities."""

    def test_format_card_valid(self):
        """Test formatting valid card indices."""
        # Test some specific cards
        # Card index = rank + suit * NUM_RANKS
        # 2♣ = 0 + 0*13 = 0
        result = format_card(0)
        assert isinstance(result, Text)
        assert "2" in result.plain
        assert "♣" in result.plain

        # A♠ = 12 + 3*13 = 51
        result = format_card(51)
        assert isinstance(result, Text)
        assert "A" in result.plain
        assert "♠" in result.plain

        # K♥ = 11 + 2*13 = 37
        result = format_card(37)
        assert isinstance(result, Text)
        assert "K" in result.plain
        assert "♥" in result.plain

    def test_format_card_hidden(self):
        """Test formatting hidden/undealt cards."""
        result = format_card(-1)
        assert isinstance(result, Text)
        assert "?" in result.plain

    def test_format_cards_multiple(self):
        """Test formatting multiple cards."""
        # Test with a few cards
        cards = [0, 12, 51]  # 2♣, A♣, A♠
        result = format_cards(cards)
        assert isinstance(result, Text)
        plain = result.plain
        assert "2" in plain
        assert "A" in plain
        assert "♣" in plain
        assert "♠" in plain

    def test_format_cards_empty(self):
        """Test formatting empty card list."""
        result = format_cards([])
        assert isinstance(result, Text)
        assert result.plain == ""


class TestObservationParsing:
    """Test observation parsing utilities."""

    def test_parse_observation_cards_preflop(self):
        """Test parsing cards during preflop (only hole cards)."""
        # Create mock observation
        obs = {
            "cards": torch.tensor(
                [
                    [0, 12, -1, -1, -1, -1, -1]  # 2♣, A♣, no community cards
                ]
            )
        }

        result = parse_observation_cards(obs, env_idx=0)
        assert "hole" in result
        assert "community" in result
        assert result["hole"] == [0, 12]
        assert result["community"] == []

    def test_parse_observation_cards_flop(self):
        """Test parsing cards during flop."""
        obs = {
            "cards": torch.tensor(
                [
                    [0, 12, 1, 2, 3, -1, -1]  # Hole: 2♣ A♣, Flop: 3♣ 4♣ 5♣
                ]
            )
        }

        result = parse_observation_cards(obs, env_idx=0)
        assert result["hole"] == [0, 12]
        assert result["community"] == [1, 2, 3]

    def test_parse_observation_cards_river(self):
        """Test parsing cards on river (all cards dealt)."""
        obs = {
            "cards": torch.tensor(
                [
                    [0, 12, 1, 2, 3, 4, 5]  # All cards
                ]
            )
        }

        result = parse_observation_cards(obs, env_idx=0)
        assert result["hole"] == [0, 12]
        assert result["community"] == [1, 2, 3, 4, 5]


class TestLegalActions:
    """Test legal action utilities."""

    def test_get_legal_actions_all_legal(self):
        """Test when all actions are legal."""
        obs = {
            "action_mask": torch.tensor([[True, True, True, True]])  # All legal
        }

        legal = get_legal_actions(obs, env_idx=0)
        assert legal == [0, 1, 2, 3]

    def test_get_legal_actions_some_legal(self):
        """Test when only some actions are legal."""
        obs = {
            "action_mask": torch.tensor(
                [[True, False, True, True]]  # Fold, call, raise (no check)
            )
        }

        legal = get_legal_actions(obs, env_idx=0)
        assert legal == [0, 2, 3]

    def test_get_legal_actions_fold_only(self):
        """Test when only fold is legal."""
        obs = {
            "action_mask": torch.tensor([[True, False, False, False]])  # Only fold
        }

        legal = get_legal_actions(obs, env_idx=0)
        assert legal == [0]


class TestHumanInputParsing:
    """Test human input parsing."""

    def create_obs(
        self, action_mask: list[bool], min_raise: int = 10, max_raise: int = 100
    ) -> dict:
        """Helper to create observation dict."""
        return {
            "action_mask": torch.tensor([action_mask]),
            "min_raise": torch.tensor([min_raise]),
            "max_raise": torch.tensor([max_raise]),
            "scalars": torch.tensor([[0.0] * 17]),  # Dummy scalars
        }

    def test_parse_fold(self):
        """Test parsing fold commands."""
        obs = self.create_obs([True, True, True, True])

        # Test 'f'
        result = parse_human_input("f", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_FOLD

        # Test 'fold'
        result = parse_human_input("fold", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_FOLD

    def test_parse_check(self):
        """Test parsing check commands."""
        obs = self.create_obs([True, True, True, True])

        # Test 'k'
        result = parse_human_input("k", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_CHECK

        # Test 'check'
        result = parse_human_input("check", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_CHECK

    def test_parse_call(self):
        """Test parsing call commands."""
        obs = self.create_obs([True, True, True, True])

        # Test 'c'
        result = parse_human_input("c", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_CALL

        # Test 'call'
        result = parse_human_input("call", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_CALL

    def test_parse_raise_with_amount(self):
        """Test parsing raise commands with amount."""
        obs = self.create_obs([True, True, True, True], min_raise=10, max_raise=100)

        # Test 'r 50'
        result = parse_human_input("r 50", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_RAISE
        assert result[1] == 50

        # Test 'raise 75'
        result = parse_human_input("raise 75", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_RAISE
        assert result[1] == 75

    def test_parse_raise_out_of_bounds(self):
        """Test parsing raise with out-of-bounds amount."""
        obs = self.create_obs([True, True, True, True], min_raise=10, max_raise=100)

        # Too small
        result = parse_human_input("r 5", obs, env_idx=0)
        assert result is None

        # Too large
        result = parse_human_input("r 200", obs, env_idx=0)
        assert result is None

    def test_parse_raise_at_boundaries(self):
        """Test parsing raise at min/max boundaries."""
        obs = self.create_obs([True, True, True, True], min_raise=10, max_raise=100)

        # Minimum
        result = parse_human_input("r 10", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_RAISE
        assert result[1] == 10

        # Maximum
        result = parse_human_input("r 100", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_RAISE
        assert result[1] == 100

    def test_parse_illegal_action(self):
        """Test parsing illegal action (not in mask)."""
        obs = self.create_obs([True, False, True, True])  # Check not allowed

        result = parse_human_input("check", obs, env_idx=0)
        assert result is None

    def test_parse_invalid_command(self):
        """Test parsing invalid command."""
        obs = self.create_obs([True, True, True, True])

        result = parse_human_input("invalid", obs, env_idx=0)
        assert result is None

        result = parse_human_input("", obs, env_idx=0)
        assert result is None

    def test_parse_case_insensitive(self):
        """Test that parsing is case insensitive."""
        obs = self.create_obs([True, True, True, True])

        result = parse_human_input("FOLD", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_FOLD

        result = parse_human_input("CaLl", obs, env_idx=0)
        assert result is not None
        assert result[0] == c.ACTION_CALL


class TestIntegration:
    """Integration tests for interactive game components."""

    def test_card_formatting_round_trip(self):
        """Test that we can format all valid card indices."""
        # Test all 52 cards can be formatted
        for card_idx in range(52):
            result = format_card(card_idx)
            assert isinstance(result, Text)
            assert len(result.plain) >= 2  # At least rank + suit

    def test_action_parsing_coverage(self):
        """Test that all action types can be parsed."""
        obs = {
            "action_mask": torch.tensor([[True, True, True, True]]),
            "min_raise": torch.tensor([10]),
            "max_raise": torch.tensor([100]),
            "scalars": torch.tensor([[0.0] * 17]),
        }

        # Fold
        result = parse_human_input("f", obs)
        assert result[0] == c.ACTION_FOLD

        # Check
        result = parse_human_input("k", obs)
        assert result[0] == c.ACTION_CHECK

        # Call
        result = parse_human_input("c", obs)
        assert result[0] == c.ACTION_CALL

        # Raise
        result = parse_human_input("r 50", obs)
        assert result[0] == c.ACTION_RAISE
        assert result[1] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
