import os

import pytest


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_torchrl_smoke_script_cpu_small():
    try:
        import torchrl  # noqa: F401
        from tensordict import TensorDict  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("torchrl/tensordict not installed")

    from train.torchrl_smoke import SmokeConfig, run

    cfg = SmokeConfig(device="cpu", num_envs=32, warmup_steps=1, steps=2, seed=0)
    result = run(cfg)
    assert result.env_steps == 64
    assert result.sps > 0
