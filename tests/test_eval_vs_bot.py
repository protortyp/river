import math

import pytest
import torch
import warp as wp


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
    assert m["hands"] > 0
    assert math.isfinite(m["bb_per_hand"])
    assert m["bb_per_100"] == pytest.approx(100 * m["bb_per_hand"])

    m2 = eval_vs_bot_seat_swap(env=env, policy=policy, bot_name="calling_station", target_hands=16)
    assert m2["hands"] > 0
    assert math.isfinite(m2["bb_per_hand"])
    assert math.isfinite(m2["bb_per_hand_p0"])
    assert math.isfinite(m2["bb_per_hand_p1"])
