"""Tests for RiverDCFRSolver.solve_cfvs and ReBeL RiverCFVNet."""

from __future__ import annotations

import numpy as np
import torch
from train.train_river_cfv import generate_dataset, train_cfv

from gpu_poker.cfv_net import (
    RiverCFVNet,
    load_cfv_checkpoint,
    river_hand_strength_pct,
    save_cfv_checkpoint,
)
from gpu_poker.search import NUM_COMBOS, RiverDCFRSolver, RiverHandEvaluator, combo_id


def _card(rank_ch: str, suit_ch: str) -> int:
    return "23456789TJQKA".index(rank_ch) + "cdhs".index(suit_ch) * 13


def test_solve_cfvs_matches_clairvoyance_nash_ev_and_zero_sum():
    """Clairvoyance pot=100, stack=100: analytical OOP Nash EV is +0.25*pot (vs 0 checkdown)."""
    board = [_card("2", "c"), _card("3", "d"), _card("7", "h"), _card("8", "s"), _card("J", "c")]
    nuts = (_card("A", "h"), _card("A", "s"))
    air = (_card("4", "c"), _card("5", "d"))
    catcher = (_card("Q", "h"), _card("Q", "s"))

    r0 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r0[combo_id(*nuts)] = 0.5
    r0[combo_id(*air)] = 0.5
    r1 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r1[combo_id(*catcher)] = 1.0

    solver = RiverDCFRSolver(bet_fractions=(1.0,), max_raises=1)
    cfv0, cfv1, ck0, ck1 = solver.solve_cfvs(
        board=board, pot=100, eff_stack=100, range_p0=r0, range_p1=r1, big_blind=10, iterations=400
    )
    ev_oop = float(np.dot(r0, cfv0))
    ev_ip = float(np.dot(r1, cfv1))
    # Exact zero-sum.
    assert abs(ev_oop + ev_ip) < 1e-4
    # Analytical Nash OOP EV = +0.25 pot (checkdown EV is 0.0).
    assert abs(ev_oop - 0.25) < 0.015
    assert abs(float(np.dot(r0, ck0))) < 1e-5
    assert abs(float(np.dot(r1, ck1))) < 1e-5


def test_river_cfv_net_exact_zero_sum_spr_zero_and_smoke_train(tmp_path):
    net = RiverCFVNet(hidden_dim=64, num_blocks=2)
    board = [_card("A", "s"), _card("K", "s"), _card("7", "d"), _card("3", "c"), _card("2", "s")]
    pct, valid = river_hand_strength_pct(board, RiverHandEvaluator())
    assert abs(valid.sum() - 1081.0) < 1e-5

    rng = np.random.default_rng(0)
    r0 = rng.random((2, NUM_COMBOS), dtype=np.float32) * valid[None, :]
    r0 /= r0.sum(axis=1, keepdims=True)
    r1 = rng.random((2, NUM_COMBOS), dtype=np.float32) * valid[None, :]
    r1 /= r1.sum(axis=1, keepdims=True)
    ck0 = np.zeros_like(r0)
    ck1 = np.zeros_like(r1)

    p0, p1 = net(
        r0=torch.from_numpy(r0),
        r1=torch.from_numpy(r1),
        check_cfv0=torch.from_numpy(ck0),
        check_cfv1=torch.from_numpy(ck1),
        rank_pct=torch.from_numpy(np.stack([pct, pct])),
        valid_mask=torch.from_numpy(np.stack([valid, valid])),
        spr=torch.tensor([0.0, 2.5]),
    )
    # Exact SPR == 0 matches check-down identically.
    assert torch.allclose(p0[0], torch.zeros(NUM_COMBOS), atol=1e-6)
    assert torch.allclose(p1[0], torch.zeros(NUM_COMBOS), atol=1e-6)
    # Exact zero-sum at arbitrary SPR.
    imbal = (torch.from_numpy(r0) * p0).sum(dim=-1) + (torch.from_numpy(r1) * p1).sum(dim=-1)
    assert torch.allclose(imbal, torch.zeros(2), atol=1e-5)

    # End-to-end dataset generation + training + checkpoint round-trip smoke test.
    data = generate_dataset(num_samples=4, iters=10, device="cpu", seed=1, bet_fractions=(1.0,))
    trained, metrics = train_cfv(
        data, device="cpu", hidden_dim=32, num_blocks=1, epochs=2, batch_size=2, seed=1
    )
    assert np.isfinite(metrics["val_rmse_pot"])
    out_p = save_cfv_checkpoint(tmp_path / "cfv.pt", trained)
    reloaded = load_cfv_checkpoint(out_p, device="cpu")
    assert reloaded.hidden_dim == 32
