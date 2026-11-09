from __future__ import annotations

import torch
from train.league import (
    pfsp_weights_for_exploiter,
    pfsp_weights_from_bb_per_hand,
)


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
