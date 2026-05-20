import os

import pytest


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_torchrl_wrapper_cuda_smoke():
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("torch not installed")

    if not torch.cuda.is_available():
        pytest.skip("cuda not available")

    try:
        import warp as wp
    except ModuleNotFoundError:
        pytest.skip("warp not installed")

    try:
        from tensordict import TensorDict
    except ModuleNotFoundError:
        pytest.skip("tensordict not installed")

    try:
        import torchrl  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("torchrl not installed")

    if not wp.is_device_available("cuda"):
        pytest.skip("warp cuda device not available")

    from gpu_poker.torchrl_env import WarpPokerTorchRLEnv

    n = 4096
    env = WarpPokerTorchRLEnv(num_envs=n, device="cuda:0")
    td0 = env.reset()
    assert td0.device.type == "cuda"
    assert td0["cards"].is_cuda
    assert td0["action_mask"].is_cuda

    action_type = torch.full((n,), 0, dtype=torch.int64, device="cuda")  # fold
    raise_frac = torch.zeros((n, 1), dtype=torch.float32, device="cuda")
    td = TensorDict(
        {
            "action": TensorDict(
                {"action_type": action_type, "raise_frac": raise_frac}, batch_size=[n]
            )
        },
        batch_size=[n],
        device="cuda",
    )
    out = env.step(td)

    assert "next" in out
    next_td = out["next"]
    assert next_td.device.type == "cuda"
    assert next_td["reward"].is_cuda
    assert next_td["done"].is_cuda
    assert next_td["terminated"].is_cuda
    assert next_td["episode_id"].is_cuda
    assert next_td["invalid_action"].is_cuda

    # Folding always ends the hand; env auto-resets but signals termination.
    assert bool(next_td["terminated"].all().item()) is True
