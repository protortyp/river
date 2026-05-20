import os

import pytest
import torch


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_eval_vs_snapshot_mix_smoke_cpu():
    from train.eval import eval_vs_snapshot_mix_seat_swap

    from gpu_poker.env import WarpPokerEnv
    from gpu_poker.policy import PokerPolicyNet

    env = WarpPokerEnv(num_envs=64, device="cpu")
    obs = env.reset()
    scalar_dim = int(obs["scalars"].shape[1])

    pol = PokerPolicyNet(
        scalar_dim=scalar_dim, lstm_hidden=32, mlp_dim=32, torso_layers=1, card_embed_dim=16
    )
    snaps = [
        PokerPolicyNet(
            scalar_dim=scalar_dim, lstm_hidden=32, mlp_dim=32, torso_layers=1, card_embed_dim=16
        )
        for _ in range(3)
    ]
    w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    m = eval_vs_snapshot_mix_seat_swap(
        env=env,
        policy=pol,
        snapshots=snaps,
        snap_weights=w,
        target_hands=200,
        max_steps=200000,
        deterministic=False,
    )
    assert m["hands"] > 0
    assert abs(float(m["bb_per_hand"])) < 1e6
