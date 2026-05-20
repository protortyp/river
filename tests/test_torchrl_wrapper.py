import pytest


def test_torchrl_wrapper_smoke():
    try:
        from tensordict import TensorDict
    except ModuleNotFoundError:
        pytest.skip("tensordict not installed")

    try:
        import torch
        import torchrl  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("torchrl not installed")

    from gpu_poker.torchrl_env import WarpPokerTorchRLEnv

    env = WarpPokerTorchRLEnv(num_envs=8, device="cpu")
    td0 = env.reset()
    assert td0.batch_size == torch.Size([8])
    assert td0["cards"].shape == (8, 7)
    assert td0["action_mask"].shape == (8, 4)

    action_type = torch.zeros((8,), dtype=torch.int64)
    raise_frac = torch.zeros((8, 1), dtype=torch.float32)
    td = TensorDict(
        {
            "action": TensorDict(
                {"action_type": action_type, "raise_frac": raise_frac}, batch_size=[8]
            )
        },
        batch_size=[8],
    )
    out = env.step(td)
    assert "next" in out
    assert out["next"]["reward"].shape == (8, 1)
    assert out["next"]["done"].shape == (8, 1)
