import os

import pytest
import torch
import warp as wp


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_eval_vs_bot_cpu_smoke():
    from train.eval import eval_vs_bot, eval_vs_bot_seat_swap

    from gpu_poker.env import WarpPokerEnv
    from gpu_poker.policy import PokerPolicyNet

    try:
        has_cuda = wp.is_device_available("cuda") and torch.cuda.is_available()
    except RuntimeError:
        has_cuda = False
    device = "cuda:0" if has_cuda else "cpu"
    env = WarpPokerEnv(num_envs=32, device=device)
    obs = env.reset()
    policy = PokerPolicyNet(scalar_dim=int(obs["scalars"].shape[1]), lstm_hidden=128).to(
        device=device
    )

    m = eval_vs_bot(env=env, policy=policy, bot_name="calling_station", target_hands=16)
    assert "bb_per_hand" in m
    assert "bb_per_100" in m
    assert "hands" in m
    assert m["hands"] >= 0

    m2 = eval_vs_bot_seat_swap(env=env, policy=policy, bot_name="calling_station", target_hands=16)
    assert "bb_per_hand" in m2
    assert "bb_per_100" in m2
    assert "hands" in m2
    assert "bb_per_hand_p0" in m2
    assert "bb_per_hand_p1" in m2
