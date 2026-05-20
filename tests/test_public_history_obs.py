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


def test_public_history_scalars_update():
    env = WarpPokerEnv(num_envs=1, device=DEVICE)
    obs0 = env.reset()

    scalars0 = obs0["scalars"][0]
    assert scalars0.shape == (17,)
    assert float(scalars0[11].item()) == 0.0  # actions_this_street
    assert float(scalars0[12].item()) == 0.0  # last_action_type (none)
    assert float(scalars0[13].item()) == 0.0  # last_action_was_raise
    assert float(scalars0[14].item()) == 0.0  # last_action_amount_norm

    # Preflop: SB is active and faces 1 chip to call -> CALL is legal.
    out1 = env.step(
        torch.tensor([c.ACTION_CALL], device=DEVICE, dtype=torch.int32),
        torch.tensor([0], device=DEVICE, dtype=torch.int32),
    )
    scalars1 = out1["scalars"][0]
    assert torch.allclose(scalars1[11], torch.tensor(1.0 / 15.0, device=DEVICE))
    assert torch.allclose(scalars1[12], torch.tensor((c.ACTION_CALL + 1) / 4.0, device=DEVICE))
    assert float(scalars1[13].item()) == 0.0
    assert torch.allclose(
        scalars1[14],
        torch.tensor(1.0 / c.STARTING_STACK, device=DEVICE),
        atol=1e-6,
        rtol=0.0,
    )

    # BB option: to_call=0 -> CHECK is legal.
    # This should also advance to the flop and reset actions_this_street.
    out2 = env.step(
        torch.tensor([c.ACTION_CHECK], device=DEVICE, dtype=torch.int32),
        torch.tensor([0], device=DEVICE, dtype=torch.int32),
    )
    scalars2 = out2["scalars"][0]
    assert float(scalars2[11].item()) == 0.0
    assert torch.allclose(scalars2[12], torch.tensor((c.ACTION_CHECK + 1) / 4.0, device=DEVICE))
    assert float(scalars2[13].item()) == 0.0
    assert float(scalars2[14].item()) == 0.0
