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
    RiverSubgameSession,
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


def test_pseudo_harmonic_translation_and_nested_river_session():
    from gpu_poker.search import RiverSubgameSession, pseudo_harmonic_bucket_weights

    # Exact bucket match (0.5 * 1000 = 500 -> bucket 1) returns weight 1.0.
    w_exact = pseudo_harmonic_bucket_weights(500, pot_total=1000, min_raise=100, max_raise=5000)
    assert w_exact == [(1, 1.0)]

    # Off-tree 0.6p (600 between 0.5p=500 and 0.75p=750) returns exact pseudo-harmonic split.
    w_mid = pseudo_harmonic_bucket_weights(600, pot_total=1000, min_raise=100, max_raise=5000)
    assert len(w_mid) == 2
    assert w_mid[0][0] == 1 and w_mid[1][0] == 2
    assert abs(w_mid[0][1] + w_mid[1][1] - 1.0) < 1e-6
    # Expected P(0.5) = ((0.75 - 0.6) * 1.5) / ((0.75 - 0.5) * 1.6) = 0.225 / 0.40 = 0.5625
    assert abs(w_mid[0][1] - 0.5625) < 1e-6

    # Nested RiverSubgameSession: reuses cached solved tree on in-tree villain bet,
    # and re-solves cleanly on off-tree villain bet.
    board = [_card("2", "c"), _card("3", "d"), _card("7", "h"), _card("8", "s"), _card("J", "c")]
    nuts = [_card("A", "h"), _card("A", "s")]
    air = [_card("4", "c"), _card("5", "d")]
    catcher = [_card("Q", "h"), _card("Q", "s")]
    r0 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r0[combo_id(*nuts)] = 0.5
    r0[combo_id(*air)] = 0.5
    r1 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r1[combo_id(*catcher)] = 1.0

    solver = RiverDCFRSolver(bet_fractions=(1.0,), max_raises=2)
    sess = RiverSubgameSession(
        solver=solver,
        board=board,
        range_p0=r0,
        range_p1=r1,
        big_blind=10,
        iterations=100,
        prune_eps=0.0,
        keep_combo=combo_id(*nuts),
    )
    # Turn 1: P0 acts -> triggers 1st solve.
    act1, amt1 = sess.act(
        pot=100,
        stacks=(300, 300),
        bets=(0, 0),
        active_player=0,
        actions_taken=0,
        raises_taken=0,
        last_raise=0,
        hole_cards=nuts,
        deterministic=True,
    )
    assert sess.solves_run == 1
    assert act1 == c.ACTION_RAISE and amt1 in (100, 300)

    # Villain makes an off-tree min-raise on a non-all-in bet -> triggers nested re-solve.
    sess.current_node = sess.current_node  # keep tree state; test off-tree on fresh 100 bet
    sess2 = RiverSubgameSession(
        solver=RiverDCFRSolver(bet_fractions=(1.0,), max_raises=2),
        board=board,
        range_p0=r0,
        range_p1=r1,
        big_blind=10,
        iterations=100,
        prune_eps=0.0,
        keep_combo=combo_id(*nuts),
    )
    a1, d1 = sess2.act(
        pot=100,
        stacks=(400, 400),
        bets=(0, 0),
        active_player=0,
        actions_taken=0,
        raises_taken=0,
        last_raise=0,
        hole_cards=nuts,
        deterministic=True,
    )
    assert sess2.solves_run == 1 and a1 == c.ACTION_RAISE
    # Simulate villain off-tree raise of +25 (neither call nor all-in) when chips remain.
    bet_hero = min(d1, 100)
    sess2.observe_action(c.ACTION_RAISE, delta=25)
    act2, _amt2 = sess2.act(
        pot=100,
        stacks=(400 - bet_hero, 400 - bet_hero - 25),
        bets=(bet_hero, bet_hero + 25),
        active_player=0,
        actions_taken=2,
        raises_taken=2,
        last_raise=25,
        hole_cards=nuts,
        deterministic=True,
    )
    assert sess2.solves_run == 2
    assert act2 in (c.ACTION_CALL, c.ACTION_RAISE)


def test_turn_equity_matrix_exact_outs_and_continual_turn_to_river_handoff():
    from gpu_poker.search import RiverSubgameSession, build_turn_equity_matrix

    # Board: 2c 3d 7h 8s (no flush/straight draws for AA vs KK).
    # KK has 2 outs (Kc, Kd) out of 44 unblocked river cards -> AA equity sign = (42 - 2)/44.
    board4 = [_card("2", "c"), _card("3", "d"), _card("7", "h"), _card("8", "s")]
    aa = [_card("A", "h"), _card("A", "s")]
    kk = [_card("K", "h"), _card("K", "s")]
    combos = np.array([combo_id(*aa), combo_id(*kk)], dtype=np.int32)
    c0 = COMBO_C0[combos]
    c1 = COMBO_C1[combos]
    s_bar = build_turn_equity_matrix(c0, c1, board4, RiverHandEvaluator())
    assert abs(float(s_bar[0, 1]) - (40.0 / 44.0)) < 1e-5
    assert abs(float(s_bar[1, 0]) + (40.0 / 44.0)) < 1e-5

    # Continual Turn -> River handoff masks the river card and solves the river subgame.
    air = [_card("4", "c"), _card("5", "d")]
    r0 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r0[combo_id(*aa)] = 0.5
    r0[combo_id(*air)] = 0.5
    r1 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r1[combo_id(*kk)] = 1.0

    turn_solver = RiverDCFRSolver(bet_fractions=(1.0,), max_raises=1)
    turn_sess = RiverSubgameSession(
        solver=turn_solver,
        board=board4,
        range_p0=r0,
        range_p1=r1,
        big_blind=10,
        iterations=80,
        prune_eps=0.0,
        keep_combo=combo_id(*aa),
    )
    t_act, t_amt = turn_sess.act(
        pot=100,
        stacks=(200, 200),
        bets=(0, 0),
        active_player=0,
        actions_taken=0,
        raises_taken=0,
        last_raise=0,
        hole_cards=aa,
        deterministic=True,
    )
    assert t_act == c.ACTION_RAISE and t_amt > 0
    turn_sess.observe_action(c.ACTION_CALL, delta=0)

    # River card Kd hits KK's set! Handoff masks Kd and solves river cleanly.
    river_card = _card("K", "d")
    river_sess = turn_sess.handoff_to_river(river_card=river_card, river_solver=turn_solver)
    assert len(river_sess.board) == 5
    assert river_sess.ranges[0][combo_id(river_card, _card("Q", "c"))] == 0.0
    r_act, _r_amt = river_sess.act(
        pot=300,
        stacks=(100, 100),
        bets=(0, 0),
        active_player=0,
        actions_taken=0,
        raises_taken=0,
        last_raise=0,
        hole_cards=aa,
        deterministic=True,
    )
    assert r_act in (c.ACTION_CHECK, c.ACTION_RAISE)


def test_midstreet_facing_overbet_exact_pot_odds_and_zero_sum():
    """Mid-street solve facing a 5x-pot bet must keep c0 == c1 and enforce exact 45.45% odds."""
    board = [_card("2", "c"), _card("3", "d"), _card("7", "h"), _card("8", "s"), _card("J", "c")]
    nuts = [_card("A", "h"), _card("A", "s")]
    air = [_card("4", "c"), _card("5", "d")]
    catcher = [_card("Q", "h"), _card("Q", "s")]

    # pot=1000 before street, Villain (P1) bet 5000 (5x pot) all-in -> Hero (P0) faces 5000 to win
    # 6000 (total final pot 11000), so break-even equity to call is 5000 / 11000 = 45.45%.
    r0 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r0[combo_id(*catcher)] = 1.0

    # Case 1: Villain has 60% nuts / 40% air -> Hero has 40% equity (< 45.45%) -> must 100% FOLD.
    r1_value_heavy = np.zeros(NUM_COMBOS, dtype=np.float32)
    r1_value_heavy[combo_id(*nuts)] = 0.60
    r1_value_heavy[combo_id(*air)] = 0.40

    solver = RiverDCFRSolver(bet_fractions=(1.0,), max_raises=1)
    res_fold = solver.solve(
        board=board,
        pot=1000,
        stacks=(5000, 0),
        bets=(0, 5000),
        active_player=0,
        range_p0=r0,
        range_p1=r1_value_heavy,
        actions_taken=1,
        raises_taken=1,
        last_raise=5000,
        big_blind=100,
        iterations=100,
    )
    assert res_fold.root_node is not None
    # Verify exact zero-sum c0 == c1 at every showdown leaf in the mid-street tree.
    stack = [res_fold.root_node]
    while stack:
        nd = stack.pop()
        if nd.kind == "showdown":
            assert abs(nd.c0 - nd.c1) < 1e-5
            assert abs(nd.c0 - 5500.0) < 1e-5
        stack.extend(nd.children)

    pos = int(np.searchsorted(res_fold.combo_indices, combo_id(*catcher)))
    fold_idx = next(i for i, (a, _) in enumerate(res_fold.actions) if a == c.ACTION_FOLD)
    call_idx = next(i for i, (a, _) in enumerate(res_fold.actions) if a == c.ACTION_CALL)
    assert float(res_fold.strategy[fold_idx, pos]) > 0.99

    # Case 2: Villain has 50% nuts / 50% air -> Hero has 50% equity (> 45.45%) -> must 100% CALL.
    r1_bluff_heavy = np.zeros(NUM_COMBOS, dtype=np.float32)
    r1_bluff_heavy[combo_id(*nuts)] = 0.50
    r1_bluff_heavy[combo_id(*air)] = 0.50
    res_call = solver.solve(
        board=board,
        pot=1000,
        stacks=(5000, 0),
        bets=(0, 5000),
        active_player=0,
        range_p0=r0,
        range_p1=r1_bluff_heavy,
        actions_taken=1,
        raises_taken=1,
        last_raise=5000,
        big_blind=100,
        iterations=100,
    )
    pos2 = int(np.searchsorted(res_call.combo_indices, combo_id(*catcher)))
    assert float(res_call.strategy[call_idx, pos2]) > 0.99


def test_river_subgame_session_clamps_offtree_cached_raise_to_legal_bounds() -> None:
    """Off-tree villain bets matched to a cached template branch must yield legal raise deltas."""
    # Board: 2c 7d 9h Js Kd; Hero holds As Ah (strong value hand that raises vs small bet).
    board = [0, 18, 30, 39, 46]
    hero = [51, 50]
    nuts = [47, 45]  # Kc Kh
    air = [4, 9]  # 3c 4d

    r0 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r0[combo_id(*hero)] = 1.0
    r1 = np.zeros(NUM_COMBOS, dtype=np.float32)
    r1[combo_id(*air)] = 0.95
    r1[combo_id(*nuts)] = 0.05

    solver = RiverDCFRSolver(bet_fractions=(0.33, 0.75, 1.5), max_raises=3)
    session = RiverSubgameSession(
        solver=solver,
        board=board,
        range_p0=r0,
        range_p1=r1,
        big_blind=100,
        iterations=30,
        keep_combo=combo_id(*hero),
        hero_seat=0,
    )
    # Solve river entry (pot=424, p1 acts first; 0.33p template bet = 140).
    entry_res = solver.solve(
        board=board,
        pot=424,
        stacks=(19788, 19788),
        bets=(0, 0),
        active_player=1,
        range_p0=r0,
        range_p1=r1,
        actions_taken=0,
        raises_taken=0,
        last_raise=0,
        big_blind=100,
        iterations=30,
        keep_combo=combo_id(*hero),
    )
    session.current_node = entry_res.root_node
    session.combo_indices = entry_res.combo_indices

    # Villain bets 188 (within 14% off-tree tolerance of 140 -> reuses template child where
    # 0.33p raise delta is 186 < live min-raise 188).
    session.observe_action(c.ACTION_RAISE, 188)
    assert session.current_node is not None
    # Force the template's smallest raise action (delta=186) to be picked deterministically.
    node = session.current_node
    assert node.strategy_sum is not None and session.combo_indices is not None
    small_raise_idx = min(
        (i for i, (a, _) in enumerate(node.actions) if a == c.ACTION_RAISE),
        key=lambda i: node.actions[i][1],
    )
    assert node.actions[small_raise_idx][1] < 188
    node.strategy_sum.fill(0.0)
    node.strategy_sum[small_raise_idx, :] = 1.0

    act_type, delta = session.act(
        pot=424,
        stacks=(19788, 19600),
        bets=(0, 188),
        active_player=0,
        actions_taken=1,
        raises_taken=1,
        last_raise=188,
        hole_cards=hero,
        deterministic=True,
    )
    assert act_type == c.ACTION_RAISE
    assert delta >= 188
