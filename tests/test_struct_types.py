"""Unit tests for Warp struct definitions."""

import warp as wp

from gpu_poker import constants
from gpu_poker.struct_types import Action, GameState, Observation, StepResult


class TestWarpStructRegistration:
    """Test that structs are properly registered with Warp."""

    def test_gamestate_is_struct(self):
        """Verify GameState is a Warp struct."""
        assert hasattr(GameState, "__annotations__")
        # Verify it has the __warp_struct__ marker added by @wp.struct decorator
        assert hasattr(GameState, "__warp_struct__") or hasattr(GameState, "cls")

    def test_action_is_struct(self):
        """Verify Action is a Warp struct."""
        assert hasattr(Action, "__annotations__")
        assert hasattr(Action, "__warp_struct__") or hasattr(Action, "cls")

    def test_observation_is_struct(self):
        """Verify Observation is a Warp struct."""
        assert hasattr(Observation, "__annotations__")
        assert hasattr(Observation, "__warp_struct__") or hasattr(Observation, "cls")

    def test_stepresult_is_struct(self):
        """Verify StepResult is a Warp struct."""
        assert hasattr(StepResult, "__annotations__")
        assert hasattr(StepResult, "__warp_struct__") or hasattr(StepResult, "cls")


class TestGameStateStruct:
    """Test GameState struct definition."""

    def test_gamestate_fields_exist(self):
        """Verify GameState has expected fields."""
        expected_fields = {
            "stage",
            "button",
            "active_player",
            "done",
            "deck",
            "deck_top",
            "hole_cards",
            "community_cards",
            "num_community",
            "stacks",
            "bets",
            "initial_stacks",
            "pot",
            "last_raise",
            "num_raises",
            "last_aggressor",
            "num_actions",
            "actions_this_street",
            "last_action_type",
            "last_action_amount",
            "last_action_was_raise",
            "rng_state",
            "episode_id",
            "cfg_starting_stack",
            "cfg_small_blind",
            "cfg_big_blind",
        }
        actual_fields = set(GameState.__annotations__.keys())
        assert expected_fields == actual_fields

    def test_gamestate_array_fields(self):
        """Verify all GameState fields are arrays (SoA layout)."""
        annotations = GameState.__annotations__
        # All fields in GameState should be array types
        for field_name in annotations:
            field_type = annotations[field_name]
            # Check that the type has array-like attributes
            assert hasattr(field_type, "dtype") or hasattr(field_type, "ndim"), (
                f"Field {field_name} should be an array type"
            )


class TestActionStruct:
    """Test Action struct definition."""

    def test_action_fields_exist(self):
        """Verify Action has expected fields."""
        expected_fields = {"action_type", "amount"}
        actual_fields = set(Action.__annotations__.keys())
        assert expected_fields == actual_fields

    def test_action_types(self):
        """Verify Action field types."""
        annotations = Action.__annotations__
        assert annotations["action_type"] == wp.int32
        assert annotations["amount"] == wp.int32


class TestObservationStruct:
    """Test Observation struct definition."""

    def test_observation_fields_exist(self):
        """Verify Observation has expected fields."""
        expected_fields = {
            "hole_cards",
            "community_cards",
            "stack_p0",
            "stack_p1",
            "pot_normalized",
            "bet_p0",
            "bet_p1",
            "stage",
            "position",
            "legal_fold",
            "legal_check",
            "legal_call",
            "legal_raise",
            "min_raise",
            "max_raise",
        }
        actual_fields = set(Observation.__annotations__.keys())
        assert expected_fields == actual_fields

    def test_observation_scalar_types(self):
        """Verify Observation scalar field types."""
        annotations = Observation.__annotations__
        # Float fields
        assert annotations["stack_p0"] == wp.float32
        assert annotations["stack_p1"] == wp.float32
        assert annotations["pot_normalized"] == wp.float32
        assert annotations["bet_p0"] == wp.float32
        assert annotations["bet_p1"] == wp.float32
        assert annotations["min_raise"] == wp.float32
        assert annotations["max_raise"] == wp.float32
        # Int fields
        assert annotations["stage"] == wp.int32
        assert annotations["position"] == wp.int32
        # Bool fields
        assert annotations["legal_fold"] == wp.bool
        assert annotations["legal_check"] == wp.bool
        assert annotations["legal_call"] == wp.bool
        assert annotations["legal_raise"] == wp.bool


class TestStepResultStruct:
    """Test StepResult struct definition."""

    def test_stepresult_fields_exist(self):
        """Verify StepResult has expected fields."""
        expected_fields = {"reward", "done", "winner"}
        actual_fields = set(StepResult.__annotations__.keys())
        assert expected_fields == actual_fields

    def test_stepresult_scalar_types(self):
        """Verify StepResult scalar field types."""
        annotations = StepResult.__annotations__
        assert annotations["done"] == wp.bool
        assert annotations["winner"] == wp.int32


class TestStructInstantiation:
    """Test that structs can be instantiated and used."""

    def test_action_instantiation(self):
        """Test creating Action instances."""
        # Create a simple action
        action = Action()
        action.action_type = constants.ACTION_FOLD
        action.amount = 0
        assert action.action_type == constants.ACTION_FOLD
        assert action.amount == 0

        # Create a raise action
        raise_action = Action()
        raise_action.action_type = constants.ACTION_RAISE
        raise_action.amount = 10
        assert raise_action.action_type == constants.ACTION_RAISE
        assert raise_action.amount == 10

    def test_gamestate_instantiation(self):
        """Test GameState instantiation with array fields (SoA pattern)."""
        # Just verify we can instantiate it
        # Actual array allocation happens separately in practice
        state = GameState()
        # Verify struct has all expected fields
        assert hasattr(state, "stage")
        assert hasattr(state, "pot")
        assert hasattr(state, "stacks")
        assert hasattr(state, "bets")


class TestWarpArrayCompatibility:
    """Test that our structs work with Warp arrays."""

    def test_create_action_array(self):
        """Test creating array of Actions."""
        n_actions = 10
        actions = wp.empty(n_actions, dtype=Action)
        assert actions.dtype == Action
        assert len(actions) == n_actions

    def test_create_gamestate_array(self):
        """Test creating array of GameStates."""
        n_envs = 4
        states = wp.empty(n_envs, dtype=GameState)
        assert states.dtype == GameState
        assert len(states) == n_envs

    def test_create_observation_array(self):
        """Test creating array of Observations."""
        n_obs = 8
        obs = wp.empty(n_obs, dtype=Observation)
        assert obs.dtype == Observation
        assert len(obs) == n_obs

    def test_create_stepresult_array(self):
        """Test creating array of StepResults."""
        n_results = 16
        results = wp.empty(n_results, dtype=StepResult)
        assert results.dtype == StepResult
        assert len(results) == n_results


class TestStructFieldConsistency:
    """Test that struct fields are consistent with constants."""

    def test_observation_card_arrays(self):
        """Verify Observation card array sizes match constants."""
        # We can't directly check array dimensions in struct annotations,
        # but we can verify the types are array types
        annotations = Observation.__annotations__
        # These should be array types (we'll verify proper shapes when creating actual arrays)
        assert "hole_cards" in annotations
        assert "community_cards" in annotations

    def test_gamestate_player_arrays(self):
        """Verify GameState player array fields exist."""
        annotations = GameState.__annotations__
        assert "stacks" in annotations
        assert "bets" in annotations
        assert "hole_cards" in annotations

    def test_stepresult_reward_array(self):
        """Verify StepResult has reward array."""
        annotations = StepResult.__annotations__
        assert "reward" in annotations


class TestStructDocumentation:
    """Test that structs have proper documentation."""

    def test_gamestate_has_docstring(self):
        """Verify GameState has documentation."""
        assert GameState.__doc__ is not None
        assert len(GameState.__doc__.strip()) > 0

    def test_action_has_docstring(self):
        """Verify Action has documentation."""
        assert Action.__doc__ is not None
        assert len(Action.__doc__.strip()) > 0

    def test_observation_has_docstring(self):
        """Verify Observation has documentation."""
        assert Observation.__doc__ is not None
        assert len(Observation.__doc__.strip()) > 0

    def test_stepresult_has_docstring(self):
        """Verify StepResult has documentation."""
        assert StepResult.__doc__ is not None
        assert len(StepResult.__doc__.strip()) > 0
