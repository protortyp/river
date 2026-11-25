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


def test_river_cfv_net_exact_card_permutation_equivariance():
    """Relabeling the 52 cards permutes RiverCFVNet outputs identically across all 1,326 combos."""
    torch.manual_seed(7)
    net = RiverCFVNet(hidden_dim=32, num_blocks=2)
    # Break zero-init on out_head so equivariance is tested on non-trivial weights.
    with torch.no_grad():
        net.out_head.weight.normal_(0.0, 0.1)
        net.out_head.bias.normal_(0.0, 0.05)
    net.eval()

    board = [_card("A", "s"), _card("K", "s"), _card("7", "d"), _card("3", "c"), _card("2", "s")]
    pct, valid = river_hand_strength_pct(board, RiverHandEvaluator())
    rng = np.random.default_rng(123)
    r0 = rng.random(NUM_COMBOS, dtype=np.float32) * valid
    r0 /= r0.sum()
    r1 = rng.random(NUM_COMBOS, dtype=np.float32) * valid
    r1 /= r1.sum()
    ck0 = (rng.random(NUM_COMBOS, dtype=np.float32) - 0.5) * valid
    ck1 = (rng.random(NUM_COMBOS, dtype=np.float32) - 0.5) * valid

    # Build a random permutation of the 52 cards and induced permutation on 1,326 combos.
    from gpu_poker.search import COMBO_C0, COMBO_C1

    card_perm = rng.permutation(52)
    combo_perm = np.array(
        [
            combo_id(int(card_perm[c0]), int(card_perm[c1]))
            for c0, c1 in zip(COMBO_C0, COMBO_C1, strict=True)
        ],
        dtype=np.int64,
    )
    inv_perm = np.argsort(combo_perm)

    with torch.no_grad():
        p0, p1 = net(
            r0=torch.from_numpy(r0)[None],
            r1=torch.from_numpy(r1)[None],
            check_cfv0=torch.from_numpy(ck0)[None],
            check_cfv1=torch.from_numpy(ck1)[None],
            rank_pct=torch.from_numpy(pct)[None],
            valid_mask=torch.from_numpy(valid)[None],
            spr=torch.tensor([2.0]),
        )
        q0, q1 = net(
            r0=torch.from_numpy(r0[inv_perm])[None],
            r1=torch.from_numpy(r1[inv_perm])[None],
            check_cfv0=torch.from_numpy(ck0[inv_perm])[None],
            check_cfv1=torch.from_numpy(ck1[inv_perm])[None],
            rank_pct=torch.from_numpy(pct[inv_perm])[None],
            valid_mask=torch.from_numpy(valid[inv_perm])[None],
            spr=torch.tensor([2.0]),
        )
    assert torch.allclose(p0[0], q0[0, combo_perm], atol=1e-5)
    assert torch.allclose(p1[0], q1[0, combo_perm], atol=1e-5)

    # Active-combo slicing must match full 1,326 evaluation bit-for-bit on active combos.
    active = torch.from_numpy(np.where(valid > 0.5)[0].astype(np.int64))
    with torch.no_grad():
        s0, s1 = net(
            r0=torch.from_numpy(r0)[None],
            r1=torch.from_numpy(r1)[None],
            check_cfv0=torch.from_numpy(ck0)[None],
            check_cfv1=torch.from_numpy(ck1)[None],
            rank_pct=torch.from_numpy(pct)[None],
            valid_mask=torch.from_numpy(valid)[None],
            spr=torch.tensor([2.0]),
            active_idx=active,
        )
    assert torch.allclose(s0[0], p0[0, active], atol=1e-5)
    assert torch.allclose(s1[0], p1[0, active], atol=1e-5)


def test_turn_dcfr_with_zero_init_cfv_net_matches_exact_checkdown_and_caps_deep_jams():
    """Zero-init RiverCFVNet equals exact checkdown equity -> Turn DCFR matches identically."""
    board4 = [_card("2", "c"), _card("3", "d"), _card("7", "h"), _card("8", "s")]
    aa = (_card("A", "h"), _card("A", "s"))
    air = (_card("4", "c"), _card("5", "d"))
    kk = (_card("K", "h"), _card("K", "s"))
    r0 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r0[combo_id(*aa)] = 0.5
    r0[combo_id(*air)] = 0.5
    r1 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r1[combo_id(*kk)] = 1.0

    # Deep SPR = 10 -> turn_all_in_max_spr=3.0 suppresses absurd 10x-pot all-in turn shove.
    s_base = RiverDCFRSolver(bet_fractions=(1.0,), max_raises=1, turn_all_in_max_spr=3.0)
    res_base = s_base.solve(
        board=board4,
        pot=100,
        stacks=(1000, 1000),
        bets=(0, 0),
        active_player=0,
        range_p0=r0,
        range_p1=r1,
        big_blind=10,
        iterations=30,
    )
    assert all(amt <= 100 for _, amt in res_base.actions)

    # With zero-init RiverCFVNet (all 48 river cards), Turn DCFR matches exact checkdown.
    zero_net = RiverCFVNet(hidden_dim=32, num_blocks=1)
    s_cfv = RiverDCFRSolver(
        bet_fractions=(1.0,),
        max_raises=1,
        cfv_net=zero_net,
        turn_all_in_max_spr=3.0,
        cfv_sample_rivers=48,
    )
    res_cfv = s_cfv.solve(
        board=board4,
        pot=100,
        stacks=(1000, 1000),
        bets=(0, 0),
        active_player=0,
        range_p0=r0,
        range_p1=r1,
        big_blind=10,
        iterations=30,
    )
    assert res_cfv.actions == res_base.actions
    assert np.allclose(res_cfv.strategy, res_base.strategy, atol=1e-4)
