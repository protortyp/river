import os

import pytest


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_lookup_tables_are_cached_per_device_cpu():
    from gpu_poker.env import WarpPokerEnv

    a = WarpPokerEnv(num_envs=8, device="cpu")
    b = WarpPokerEnv(num_envs=8, device="cpu")

    assert a.primes is b.primes
    assert a.unsuited_lut is b.unsuited_lut
    assert a.flush_lut is b.flush_lut
