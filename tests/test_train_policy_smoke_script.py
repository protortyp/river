import os

import pytest


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_policy_train_smoke_script_cpu_small():
    from train.policy_train_smoke import TrainSmokeConfig, run

    cfg = TrainSmokeConfig(
        device="cpu",
        num_envs=32,
        warmup_steps=1,
        rollout_steps=2,
        updates=1,
        seed=0,
        lr=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=1.0,
    )
    result = run(cfg)
    assert result.env_steps == 64
    assert result.overall_sps > 0
