"""Tests for RiverHandEvaluator, BeliefTracker, and RiverDCFRSolver."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_poker import constants as c
from gpu_poker.search import (
    COMBO_C0,
    COMBO_C1,
    NUM_COMBOS,
    BeliefTracker,
    RiverDCFRSolver,
    RiverHandEvaluator,
    combo_id,
    unblocked_combo_indices,
)


def _card(rank_ch: str, suit_ch: str) -> int:
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return ranks.index(rank_ch) + suits.index(suit_ch) * 13


def test_river_hand_evaluator_matches_treys_on_random_boards():
    treys = pytest.importorskip("treys")
    treys_eval = treys.Evaluator()
    ev = RiverHandEvaluator()
    rng = np.random.default_rng(42)

    ranks_s = "23456789TJQKA"
    suits_s = "cdhs"

    def to_treys(idx: int) -> int:
        return treys.Card.new(ranks_s[idx % 13] + suits_s[idx // 13])

    for _ in range(5):
        deck = rng.permutation(52)
        board = deck[:5].tolist()
        t_board = [to_treys(x) for x in board]
        combos = unblocked_combo_indices(board)
        sample = combos[:40]
        pairs = np.stack([COMBO_C0[sample], COMBO_C1[sample]], axis=1)
        our_scores = ev.eval_river_combos(pairs, board)

        # Lower score is better in treys; higher score is better in our LUT.
        t_scores = np.array(
            [treys_eval.evaluate(t_board, [to_treys(int(a)), to_treys(int(b))]) for a, b in pairs]
        )
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                our_cmp = np.sign(int(our_scores[i]) - int(our_scores[j]))
                treys_cmp = np.sign(int(t_scores[j]) - int(t_scores[i]))
                assert our_cmp == treys_cmp


def test_river_dcfr_clairvoyance_game_hits_exact_gto_frequencies():
    """Analytical GTO benchmark: classic 1-bet river clairvoyance game.

    Board: 2c 3d 7h 8s Jc (no flush, no straight possible for our test hands).
    Pot = 100, effective stack = 100 (exact pot-sized bet B = P = 100).
    - P0 (IP/OOP acting first) has 50% nuts (AA) and 50% pure air (4c 5d).
    - P1 holds a pure bluff-catcher (QQ) that beats air and loses to AA.

    Analytical Nash equilibrium for pot-sized bet B = P:
      - P0 bets 100% of nuts (AA) and bluffs at alpha = B / (P + B) = 50% of air
        (1:2 bluff-to-value ratio, since P0 has equal combos of AA and air, wait:
        with 1 nut combo and 1 air combo, P0 bets 100% AA and 50% air!).
      - P1 calls P0's pot bet with frequency P / (P + B) = 50% (making P0's air
        indifferent between checking and bluffing!).
      - Subgame exploitability -> 0.
    """
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
    res = solver.solve(
        board=board,
        pot=100,
        stacks=(100, 100),
        bets=(0, 0),
        active_player=0,
        range_p0=r0,
        range_p1=r1,
        big_blind=10,
        iterations=500,
        compute_exploitability=True,
    )

    # Actions should be [CHECK, RAISE(100)].
    assert len(res.actions) == 2
    assert res.actions[0] == (c.ACTION_CHECK, 0)
    assert res.actions[1] == (c.ACTION_RAISE, 100)

    p_nuts = res.action_probs_for_hand(nuts)
    p_air = res.action_probs_for_hand(air)

    # GTO: nuts bet ~100%, air bluffs ~50%.
    assert p_nuts[1] > 0.98, f"Expected nuts to bet ~100%, got {p_nuts}"
    assert abs(p_air[1] - 0.50) < 0.03, f"Expected air bluff freq ~0.50, got {p_air}"
    # Exploitability < 0.5% of pot.
    assert res.exploitability_chips < 0.5


def test_river_dcfr_full_1081_combo_subgame_converges_low_exploitability():
    """Full 1,081-combo river solve reaches low exploitability and IP always value-bets the nuts."""
    board = [_card("A", "s"), _card("K", "s"), _card("7", "d"), _card("3", "c"), _card("2", "s")]
    r0 = np.ones(NUM_COMBOS, dtype=np.float32)
    r1 = np.ones(NUM_COMBOS, dtype=np.float32)

    solver = RiverDCFRSolver(bet_fractions=(0.5, 1.0), max_raises=2)
    # IP acting after OOP checked (actions_taken=1): closing action so no trap incentive.
    res = solver.solve(
        board=board,
        pot=1000,
        stacks=(2000, 2000),
        bets=(0, 0),
        active_player=1,
        actions_taken=1,
        range_p0=r0,
        range_p1=r1,
        big_blind=100,
        iterations=150,
        compute_exploitability=True,
    )

    assert len(res.combo_indices) == 1081
    assert np.allclose(res.strategy.sum(axis=0), 1.0, atol=1e-5)
    # Exploitability under 2% of pot on a full 1,081-combo multi-sizing tree in 150 iters.
    assert res.exploitability_chips < 20.0, f"Too high: {res.exploitability_chips}"

    # IP closing action with the nut flush (Qs Js) must value-bet ~100%.
    p_nut_flush = res.action_probs_for_hand((_card("Q", "s"), _card("J", "s")))
    assert p_nut_flush[0] < 0.05, f"IP nut flush checked back too often: {p_nut_flush}"


def test_belief_tracker_masks_board_and_bayes_updates():
    bt = BeliefTracker(smoothing=0.0)
    board = [_card("A", "h"), _card("K", "h"), _card("Q", "h")]
    bt.mask_board(board)

    # Any combo containing Ah must have zero mass.
    assert bt.ranges[0][combo_id(_card("A", "h"), _card("2", "c"))] == 0.0
    assert abs(bt.ranges[0].sum() - 1.0) < 1e-6

    # Bayes update concentrates mass on high-likelihood hands.
    active = unblocked_combo_indices(board)
    lik = np.full(len(active), 0.01, dtype=np.float32)
    target_cid = combo_id(_card("J", "h"), _card("T", "h"))
    target_pos = int(np.searchsorted(active, target_cid))
    lik[target_pos] = 1.0
    bt.update_action(0, lik, active_indices=active)
    assert bt.ranges[0][target_cid] > 0.05
    assert abs(bt.ranges[0].sum() - 1.0) < 1e-6


def test_disjoint_mass_matches_dense_matrix_and_slumbot_river_action_lookup():
    from train.slumbot_eval import HUNLState, _board_per_street, format_action_for_slumbot

    from gpu_poker.search import _disjoint_mass, build_river_matrices

    board_strs = ["As", "Ks", "7d", "3c", "2s"]
    bps = _board_per_street(board_strs)
    combos = unblocked_combo_indices(bps[3])
    c0 = COMBO_C0[combos]
    c1 = COMBO_C1[combos]
    ranks = RiverHandEvaluator().eval_river_combos(np.stack([c0, c1], axis=1), bps[3])
    _, d_mat = build_river_matrices(combos, ranks)

    rng = np.random.default_rng(7)
    r = rng.random(len(combos), dtype=np.float32)
    assert np.allclose(_disjoint_mass(r, c0, c1), d_mat @ r, atol=1e-3)

    # End-to-end river solve lookup on HUNLState (matching slumbot_eval driver).
    hole = [_card("Q", "s"), _card("J", "s")]
    st = HUNLState(button=1, hero_seat=0, hole_cards_hero=hole)
    st.stage = c.STAGE_RIVER
    st.pot = 1000
    st.bets = [0, 0]
    st.stacks = [19500, 19500]
    st.active_player = 0
    st.actions_this_street = 0
    st.reveal_board(bps[3])

    solver = RiverDCFRSolver(bet_fractions=(0.5, 1.0), max_raises=2)
    res = solver.solve(
        board=bps[3],
        pot=st.pot,
        stacks=(st.stacks[0], st.stacks[1]),
        bets=(st.bets[0], st.bets[1]),
        active_player=st.active_player,
        range_p0=np.ones(NUM_COMBOS, dtype=np.float32),
        range_p1=np.ones(NUM_COMBOS, dtype=np.float32),
        actions_taken=st.actions_this_street,
        big_blind=st.bb_chips,
        iterations=30,
        keep_combo=combo_id(hole[0], hole[1]),
    )
    probs = res.action_probs_for_hand(hole)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    act_type, amt = res.actions[int(np.argmax(probs))]
    incr = format_action_for_slumbot(st, act_type, amt)
    assert incr == "k" or incr.startswith("b")
