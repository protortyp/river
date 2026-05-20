from __future__ import annotations

from pathlib import Path

import torch
from train.league import (
    LeaguePool,
    pfsp_weights_for_exploiter,
    pfsp_weights_from_bb_per_hand,
)


def test_league_pool_add_and_update(tmp_path: Path) -> None:
    pool = LeaguePool(pool_dir=tmp_path / "league_pool")
    m = torch.nn.Linear(4, 3)
    meta = pool.add_snapshot(model=m, env_steps=123, update=7)
    assert Path(meta.path).exists()

    snaps = pool.load_index()
    assert len(snaps) == 1
    assert snaps[0].env_steps == 123
    assert snaps[0].update == 7
    assert snaps[0].bb_per_hand is None

    pool.update_eval(path=meta.path, bb_per_hand=0.25, hands=1024)
    snaps2 = pool.load_index()
    assert snaps2[0].bb_per_hand == 0.25
    assert snaps2[0].hands == 1024
    assert snaps2[0].bb_per_hand_p0 is None
    assert snaps2[0].bb_per_hand_p1 is None

    pool.update_eval(
        path=meta.path, bb_per_hand=0.5, hands=2048, bb_per_hand_p0=1.0, bb_per_hand_p1=0.0
    )
    snaps3 = pool.load_index()
    assert snaps3[0].bb_per_hand == 0.5
    assert snaps3[0].hands == 2048
    assert snaps3[0].bb_per_hand_p0 == 1.0
    assert snaps3[0].bb_per_hand_p1 == 0.0


def test_pfsp_weights_default_hard_favors_losing_matchups() -> None:
    """Default mode is f_hard = (1 - p)^q: weight decreases monotonically as the
    learner's bb/hand vs the opponent rises, concentrating sampling mass on the
    opponents it currently loses to."""
    bb = torch.tensor([-10.0, 0.0, 10.0])
    w = pfsp_weights_from_bb_per_hand(bb, temperature=1.0, epsilon=0.0)
    assert w[0] > w[1] > w[2]


def test_pfsp_weights_var_mode_peaks_near_zero() -> None:
    """mode="var" uses f_var = p*(1 - p): weight peaks at ~50% win probability
    and falls off toward both extremes."""
    bb = torch.tensor([-10.0, 0.0, 10.0])
    w = pfsp_weights_from_bb_per_hand(bb, temperature=1.0, epsilon=0.0, mode="var")
    assert w[1] > w[0]
    assert w[1] > w[2]


def test_pfsp_weights_for_exploiter_is_monotone_decreasing() -> None:
    """Exploiter PFSP must give the highest weight to opponents it currently
    loses to (negative bb/hand from the exploiter's view). Regression for the
    league-reviewer bug where the symmetric `p*(1-p)` was used instead, which
    biased the sampler toward 50/50 matchups."""
    bb = torch.tensor([-5.0, 0.0, 5.0])
    w = pfsp_weights_for_exploiter(bb, temperature=1.0, epsilon=0.0)
    # Losing badly (-5) -> highest weight; winning (+5) -> lowest weight.
    assert w[0] > w[1] > w[2]
    # Temperature only changes sharpness, not order.
    w_cold = pfsp_weights_for_exploiter(bb, temperature=0.1, epsilon=0.0)
    assert w_cold[0] > w_cold[1] > w_cold[2]


def test_league_pool_prune_recent_and_exponential(tmp_path: Path) -> None:
    pool = LeaguePool(pool_dir=tmp_path / "league_pool")
    m = torch.nn.Linear(1, 1)
    # Create snapshots with increasing env_steps (dense in each log2 bucket).
    steps_list = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
    ]
    for k, steps in enumerate(steps_list):
        pool.add_snapshot(model=m, env_steps=steps, update=k)
    pool.prune(keep_recent=2, keep_exponential=True)
    snaps = pool.load_index()
    kept_steps = [s.env_steps for s in snaps]
    # Should keep the last 2.
    assert 256 in kept_steps and 255 in kept_steps
    # Should keep at least some older buckets (e.g. 1,2,4,...), but not all.
    assert 1 in kept_steps
    assert len(kept_steps) < len(steps_list)
