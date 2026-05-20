import os

import pytest
import torch
import warp as wp

from gpu_poker import constants as c

try:
    from gpu_poker.env import WarpPokerEnv
except ImportError:  # pragma: no cover
    pytest.skip("PyTorch not installed", allow_module_level=True)


DATA_PATH = os.path.join(os.path.dirname(__file__), "../src/gpu_poker/lookup_data/hand_ranks.npz")
if not os.path.exists(DATA_PATH):
    pytest.skip("Lookup data not found. Run generator first.", allow_module_level=True)

try:
    DEVICE = "cuda" if wp.is_device_available("cuda") and torch.cuda.is_available() else "cpu"
except RuntimeError:
    DEVICE = "cpu"


def test_reset_with_config_applies_per_env_stacks_and_blinds():
    num_envs = 8
    env = WarpPokerEnv(num_envs=num_envs, device=DEVICE)

    starting_stack = torch.arange(200, 200 + num_envs, device=DEVICE, dtype=torch.int32)
    big_blind = torch.arange(4, 4 + num_envs, device=DEVICE, dtype=torch.int32)
    small_blind = (big_blind // 2).to(dtype=torch.int32)

    env.reset_with_config(
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
    )

    button = env.state.button.numpy()
    bets = env.state.bets.numpy()
    stacks = env.state.stacks.numpy()
    initial_stacks = env.state.initial_stacks.numpy()

    for i in range(num_envs):
        sb_p = button[i]
        bb_p = 1 - sb_p
        ss = int(starting_stack[i].item())
        sb = int(small_blind[i].item())
        bb = int(big_blind[i].item())

        assert initial_stacks[i, 0] == ss
        assert initial_stacks[i, 1] == ss

        assert bets[i, sb_p] == sb
        assert bets[i, bb_p] == bb

        assert stacks[i, sb_p] == ss - sb
        assert stacks[i, bb_p] == ss - bb


def test_step_auto_reset_uses_config():
    env = WarpPokerEnv(num_envs=1, device=DEVICE)
    env.reset_with_config(starting_stack=200, small_blind=2, big_blind=4)

    # Force fold for the active player; should terminate and auto-reset.
    out = env.step(
        torch.tensor([c.ACTION_FOLD], device=DEVICE, dtype=torch.int32),
        torch.tensor([0], device=DEVICE, dtype=torch.int32),
    )
    assert bool(out["terminated"][0].item()) is True

    stacks = env.state.stacks.numpy()[0]
    bets = env.state.bets.numpy()[0]
    pot = int(env.state.pot.numpy()[0])
    initial_stacks = env.state.initial_stacks.numpy()[0]

    assert initial_stacks[0] == 200
    assert initial_stacks[1] == 200
    assert int(stacks[0]) + int(stacks[1]) + int(bets[0]) + int(bets[1]) + pot == 400
