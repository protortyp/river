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


def test_env_outputs_player_id_and_it_flips_on_step():
    env = WarpPokerEnv(num_envs=1, device=DEVICE)
    obs = env.reset()
    pid0 = int(obs["player_id"][0].item())
    assert pid0 in (0, 1)

    out = env.step(
        torch.tensor([c.ACTION_CALL], device=DEVICE, dtype=torch.int32),
        torch.tensor([0], device=DEVICE, dtype=torch.int32),
    )
    pid1 = int(out["player_id"][0].item())
    assert pid1 in (0, 1)

    # In normal play, active player alternates unless a terminal auto-reset occurred.
    if not bool(out["terminated"][0].item()):
        assert pid1 == 1 - pid0
