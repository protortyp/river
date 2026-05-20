"""
Integration tests for the WarpPokerEnv class.

Verifies:
1. Environment initialization and memory allocation.
2. Interaction with PyTorch tensors (Zero-Copy).
3. Reset and Step mechanics.
"""

import os

import pytest
import torch
import warp as wp

from gpu_poker import constants as c

# Skip if torch not installed or if lookup data missing
try:
    import torch

    from gpu_poker.env import WarpPokerEnv
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

# Ensure data exists
DATA_PATH = os.path.join(os.path.dirname(__file__), "../src/gpu_poker/lookup_data/hand_ranks.npz")
if not os.path.exists(DATA_PATH):
    pytest.skip("Lookup data not found. Run generator first.", allow_module_level=True)

# Device selection
try:
    DEVICE = "cuda" if wp.is_device_available("cuda") and torch.cuda.is_available() else "cpu"
except RuntimeError:
    DEVICE = "cpu"


class TestWarpPokerEnv:
    def test_initialization(self):
        """Tests that environment allocates memory correctly."""
        num_envs = 128
        env = WarpPokerEnv(num_envs=num_envs, device=DEVICE)

        assert env.num_envs == num_envs
        assert env.state.stacks.shape == (num_envs, 2)

        # Verify zero-copy tensors are created
        assert env.obs_cards.shape == (num_envs, 7)
        assert env.obs_scalars.shape == (num_envs, 17)
        assert env.rewards.shape == (num_envs,)

        # Check device alignment
        if DEVICE == "cuda":
            # This check is tricky because env.obs_cards is a wp.array, not a torch tensor.
            # We verify the device during creation and usage implicitly.
            # A simple check is to ensure the warp device matches.
            assert env.device == "cuda"

    def test_reset(self):
        """Tests reset functionality."""
        num_envs = 64
        env = WarpPokerEnv(num_envs=num_envs, device=DEVICE)

        # Reset returns dict of torch tensors
        obs = env.reset()

        assert "cards" in obs
        assert "scalars" in obs
        assert "action_mask" in obs
        assert "min_raise" in obs
        assert "max_raise" in obs
        assert "terminated" in obs
        assert "episode_id" in obs
        assert "player_id" in obs
        assert "big_blind" in obs

        cards = obs["cards"]
        scalars = obs["scalars"]
        action_mask = obs["action_mask"]
        min_raise = obs["min_raise"]
        max_raise = obs["max_raise"]
        terminated = obs["terminated"]
        episode_id = obs["episode_id"]
        player_id = obs["player_id"]
        big_blind = obs["big_blind"]

        assert action_mask.shape == (num_envs, c.NUM_ACTIONS)
        assert min_raise.shape == (num_envs,)
        assert max_raise.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert episode_id.shape == (num_envs,)
        assert player_id.shape == (num_envs,)
        assert big_blind.shape == (num_envs,)
        assert not torch.any(terminated)

        # Verify Preflop State
        # Board cards (indices 2-6) should be -1
        board_cards = cards[:, 2:]
        assert torch.all(board_cards == -1)

        # Hole cards should be valid (>=0)
        hole_cards = cards[:, :2]
        assert torch.all(hole_cards >= 0)

        # Verify Scalars
        # Stage (index 7) should be 0.0 (Preflop)
        assert torch.all(scalars[:, 7] == 0.0)

        # Pot (index 5) should be SB+BB / (2*STARTING_STACK)
        # (1+2) / 2000 = 0.0015
        expected_pot = (c.SMALL_BLIND + c.BIG_BLIND) / (c.STARTING_STACK * 2)
        assert torch.allclose(scalars[:, 5], torch.tensor(expected_pot, device=DEVICE))

        # Check player stacks (indices 1 and 2)
        # Active player is SB (player 0), so obs are from their perspective
        # Player stack (index 1) is SB's stack: 1000 - 1 = 999
        # Opponent stack (index 2) is BB's stack: 1000 - 2 = 998
        norm_sb_stack = (c.STARTING_STACK - c.SMALL_BLIND) / c.STARTING_STACK
        norm_bb_stack = (c.STARTING_STACK - c.BIG_BLIND) / c.STARTING_STACK
        assert torch.allclose(scalars[:, 1], torch.tensor(norm_sb_stack, device=DEVICE))
        assert torch.allclose(scalars[:, 2], torch.tensor(norm_bb_stack, device=DEVICE))

        # Verify basic preflop legality from SB perspective:
        # Facing 1 chip to call (BB - SB), so Check should be illegal, Call legal.
        assert torch.all(action_mask[:, c.ACTION_FOLD])
        assert torch.all(~action_mask[:, c.ACTION_CHECK])

        # Public history scalars should be empty on reset.
        # actions_this_street, last_action_type, last_action_was_raise, last_action_amount_norm
        assert torch.all(scalars[:, 11] == 0.0)
        assert torch.all(scalars[:, 12] == 0.0)
        assert torch.all(scalars[:, 13] == 0.0)
        assert torch.all(scalars[:, 14] == 0.0)

        # BB features should be non-negative.
        assert torch.all(scalars[:, 15] >= 0.0)
        assert torch.all(scalars[:, 16] >= 0.0)
        assert torch.all(action_mask[:, c.ACTION_CALL])
        assert torch.all(action_mask[:, c.ACTION_RAISE])

        # Raise bounds are raise deltas. Default min raise is BB.
        assert torch.all(min_raise == c.BIG_BLIND)
        assert torch.all(max_raise > 0)

    def test_step(self):
        """Tests taking a step in the environment."""
        num_envs = 10
        env = WarpPokerEnv(num_envs=num_envs, device=DEVICE)
        env.reset()

        # Create dummy actions (Fold = 0)
        actions = torch.zeros(num_envs, dtype=torch.int32, device=DEVICE)

        # Step
        result = env.step(actions)

        assert "cards" in result
        assert "rewards" in result
        assert "dones" in result
        assert "action_mask" in result
        assert "min_raise" in result
        assert "max_raise" in result
        assert "invalid_action" in result
        assert "terminated" in result
        assert "episode_id" in result

        rewards = result["rewards"]
        dones = result["dones"]
        invalid_action = result["invalid_action"]
        terminated = result["terminated"]

        # Because of auto-reset, dones should be False
        assert not torch.any(dones)
        assert not torch.any(invalid_action)
        # Folding ends the hand; we auto-reset but signal termination.
        assert torch.all(terminated)

        # Rewards should be non-zero (SB lost 1 chip)
        # The reward is from the perspective of the player who just acted (SB)
        assert torch.all(rewards == -c.SMALL_BLIND)

        # Verify auto-reset happened by looking at next observation
        # Stage (index 7) should be back to Preflop (0.0)
        next_scalars = result["scalars"]
        assert torch.all(next_scalars[:, 7] == 0.0)

    def test_step_from_buffers(self):
        """Smoke test for fast-path stepping with pre-filled device buffers."""
        num_envs = 64
        env = WarpPokerEnv(num_envs=num_envs, device=DEVICE)
        env.reset()

        @wp.kernel
        def fill_actions(
            actions_arr: wp.array(dtype=wp.int32),
            amounts_arr: wp.array(dtype=wp.int32),
        ):
            i = wp.tid()
            actions_arr[i] = c.ACTION_FOLD
            amounts_arr[i] = 0

        wp.launch(fill_actions, dim=num_envs, inputs=[env.actions, env.amounts], device=DEVICE)
        result = env.step_from_buffers()

        assert torch.all(result["terminated"])
        assert not torch.any(result["invalid_action"])

    def test_step_check_flow(self):
        """Tests basic Check/Call flow."""
        num_envs = 2
        env = WarpPokerEnv(num_envs=num_envs, device=DEVICE)
        env.reset()

        # Action 2 = Call (SB calls the BB)
        actions = torch.full((num_envs,), c.ACTION_CALL, dtype=torch.int32, device=DEVICE)
        result = env.step(actions)  # SB acts

        # Now it's BB's turn, they can check
        actions = torch.full((num_envs,), c.ACTION_CHECK, dtype=torch.int32, device=DEVICE)
        result = env.step(actions)  # BB acts

        # Game should NOT be done (Preflop -> Flop)
        dones = result["dones"]
        assert not torch.any(dones)
        assert not torch.any(result["terminated"])

        # Stage should advance to Flop (Stage 1)
        # Stage is at index 7, normalized by 5.0
        # Flop is stage 1 -> 1/5 = 0.2
        scalars = result["scalars"]
        assert torch.allclose(scalars[:, 7], torch.tensor(0.2, device=DEVICE))

        # Board cards (indices 2,3,4) should now be visible (>=0)
        cards = result["cards"]
        assert torch.all(cards[:, 2:5] >= 0)

        # Pot (index 5) should be 4 chips (2 from SB, 2 from BB)
        # Normalized pot: 4 / 2000 = 0.002
        expected_pot = (c.BIG_BLIND * 2) / (c.STARTING_STACK * 2)
        assert torch.allclose(scalars[:, 5], torch.tensor(expected_pot, device=DEVICE))

    def test_heads_up_postflop_action_order_and_check_check_settlement(self):
        """Postflop HU action starts with the non-button and requires both checks."""
        env = WarpPokerEnv(num_envs=1, device=DEVICE)
        obs = env.reset()

        button = int(env.state.button.numpy()[0])
        non_button = 1 - button
        assert int(obs["player_id"][0].item()) == button
        assert float(obs["scalars"][0, 0].item()) == 0.0

        # Preflop: SB/button calls, BB checks.
        out = env.step(
            torch.tensor([c.ACTION_CALL], device=DEVICE, dtype=torch.int32),
            torch.tensor([0], device=DEVICE, dtype=torch.int32),
        )
        assert int(out["player_id"][0].item()) == non_button

        out = env.step(
            torch.tensor([c.ACTION_CHECK], device=DEVICE, dtype=torch.int32),
            torch.tensor([0], device=DEVICE, dtype=torch.int32),
        )
        assert int(env.state.stage.numpy()[0]) == c.STAGE_FLOP
        assert int(out["player_id"][0].item()) == non_button
        assert torch.allclose(out["scalars"][0, 7], torch.tensor(0.2, device=DEVICE))
        assert float(out["scalars"][0, 0].item()) == 1.0

        # First postflop check must pass action to the button, not advance the street.
        out = env.step(
            torch.tensor([c.ACTION_CHECK], device=DEVICE, dtype=torch.int32),
            torch.tensor([0], device=DEVICE, dtype=torch.int32),
        )
        assert int(env.state.stage.numpy()[0]) == c.STAGE_FLOP
        assert int(out["player_id"][0].item()) == button
        assert torch.allclose(out["scalars"][0, 7], torch.tensor(0.2, device=DEVICE))
        assert float(out["scalars"][0, 0].item()) == 0.0

        # Second check closes the flop and deals the turn.
        out = env.step(
            torch.tensor([c.ACTION_CHECK], device=DEVICE, dtype=torch.int32),
            torch.tensor([0], device=DEVICE, dtype=torch.int32),
        )
        assert int(env.state.stage.numpy()[0]) == c.STAGE_TURN
        assert int(out["player_id"][0].item()) == non_button
        assert torch.allclose(out["scalars"][0, 7], torch.tensor(0.4, device=DEVICE))
