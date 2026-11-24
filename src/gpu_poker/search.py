"""Bayesian range tracking and exact Discounted CFR (DCFR) river subgame solver.

Combines a neural blueprint policy (PyTorch or ONNX) with exact game-theoretic
endgame solving on the river:

1. `BeliefTracker` maintains each player's probability distribution over all
   1,326 2-card combos, masking dead board cards and applying exact Bayes updates
   `range[h] *= pi_blueprint(action | h, history)` at each public action.
2. `RiverDCFRSolver` constructs the exact river betting tree from the current
   public state and solves it via Discounted CFR (Brown & Sandholm, 2019;
   alpha=1.5, beta=0, gamma=2) over all unblocked 2-card combos with exact
   card-removal blockers and exact Cactus Kev 7-card hand evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gpu_poker.cfv_net import RiverCFVNet

from gpu_poker import constants as c
from gpu_poker.policy import (
    ALL_IN_BUCKET,
    MIN_RAISE_BUCKET,
    RAISE_BUCKET_FRACTIONS,
)


def pseudo_harmonic_bucket_weights(
    delta: int,
    *,
    pot_total: int,
    min_raise: int,
    max_raise: int,
    fractions: tuple[float, ...] = RAISE_BUCKET_FRACTIONS,
) -> list[tuple[int, float]]:
    """Ganzfried & Sandholm (2013) pseudo-harmonic action translation.

    Maps an arbitrary off-tree raise `delta` to a convex combination of at most
    two adjacent discrete raise buckets `(idx_lo, w_lo), (idx_hi, w_hi)` using
    `P(A) = ((B - x) * (1 + A)) / ((B - A) * (1 + x))` in pot-fraction space.
    """
    pot_f = max(1.0, float(pot_total))
    targets: list[tuple[int, int]] = []
    for idx, frac in enumerate(fractions):
        if frac == MIN_RAISE_BUCKET:
            t = min_raise
        elif frac == ALL_IN_BUCKET:
            t = max_raise
        else:
            t = round(frac * pot_f)
        t = max_raise if max_raise < min_raise else max(min_raise, min(max_raise, int(t)))
        targets.append((t, idx))

    # Deduplicate identical chip targets (keeping lowest bucket index) and sort ascending.
    seen: dict[int, int] = {}
    for t, idx in targets:
        if t not in seen:
            seen[t] = idx
    ordered = sorted(seen.items(), key=lambda kv: kv[0])
    if not ordered:
        return [(0, 1.0)]

    d = int(delta)
    if d <= ordered[0][0]:
        return [(ordered[0][1], 1.0)]
    if d >= ordered[-1][0]:
        return [(ordered[-1][1], 1.0)]

    for i in range(len(ordered) - 1):
        t_lo, idx_lo = ordered[i]
        t_hi, idx_hi = ordered[i + 1]
        if d == t_lo:
            return [(idx_lo, 1.0)]
        if d == t_hi:
            return [(idx_hi, 1.0)]
        if t_lo < d < t_hi:
            a = float(t_lo) / pot_f
            b = float(t_hi) / pot_f
            x = float(d) / pot_f
            denom = (b - a) * (1.0 + x)
            p_lo = ((b - x) * (1.0 + a)) / denom if denom > 1e-12 else 0.5
            p_lo = max(0.0, min(1.0, p_lo))
            return [(idx_lo, p_lo), (idx_hi, 1.0 - p_lo)]

    return [(ordered[-1][1], 1.0)]


# Canonical 1,326 unordered 2-card combinations (c0 < c1, c in 0..51).
_COMBO_C0_LIST: list[int] = []
_COMBO_C1_LIST: list[int] = []
_COMBO_INDEX: dict[tuple[int, int], int] = {}
for _a in range(52):
    for _b in range(_a + 1, 52):
        _COMBO_INDEX[(_a, _b)] = len(_COMBO_C0_LIST)
        _COMBO_C0_LIST.append(_a)
        _COMBO_C1_LIST.append(_b)

COMBO_C0 = np.array(_COMBO_C0_LIST, dtype=np.int32)  # [1326]
COMBO_C1 = np.array(_COMBO_C1_LIST, dtype=np.int32)  # [1326]
NUM_COMBOS = len(_COMBO_C0_LIST)  # 1326

# 21 combinations of 5 indices out of 7 for 7-card hand evaluation.
_COMBOS_7C5: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 1, 2, 3, 4),
    (0, 1, 2, 3, 5),
    (0, 1, 2, 3, 6),
    (0, 1, 2, 4, 5),
    (0, 1, 2, 4, 6),
    (0, 1, 2, 5, 6),
    (0, 1, 3, 4, 5),
    (0, 1, 3, 4, 6),
    (0, 1, 3, 5, 6),
    (0, 1, 4, 5, 6),
    (0, 2, 3, 4, 5),
    (0, 2, 3, 4, 6),
    (0, 2, 3, 5, 6),
    (0, 2, 4, 5, 6),
    (0, 3, 4, 5, 6),
    (1, 2, 3, 4, 5),
    (1, 2, 3, 4, 6),
    (1, 2, 3, 5, 6),
    (1, 2, 4, 5, 6),
    (1, 3, 4, 5, 6),
    (2, 3, 4, 5, 6),
)


def combo_id(c0: int, c1: int) -> int:
    """Return canonical index in `[0, 1326)` for an unordered pair of cards."""
    a, b = (int(c0), int(c1)) if int(c0) < int(c1) else (int(c1), int(c0))
    return _COMBO_INDEX[(a, b)]


class RiverHandEvaluator:
    """Vectorized NumPy 7-card hand evaluator backed by `hand_ranks.npz`."""

    _CACHED_TABLES: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def __init__(self) -> None:
        if RiverHandEvaluator._CACHED_TABLES is None:
            lookup_path = Path(__file__).parent / "lookup_data" / "hand_ranks.npz"
            data = np.load(lookup_path)
            RiverHandEvaluator._CACHED_TABLES = (
                np.array(c.RANK_PRIMES, dtype=np.int32),
                data["unsuited_table"].astype(np.int32),
                data["flush_table"].astype(np.int32),
            )
        self.primes, self.unsuited_lut, self.flush_lut = RiverHandEvaluator._CACHED_TABLES

    def eval_river_combos(
        self, hole_pairs: np.ndarray, board: list[int] | np.ndarray
    ) -> np.ndarray:
        """Evaluate `[H, 2]` hole-card combos on a 5-card river board -> `[H]` int32 ranks.

        Higher rank wins; identical rank indicates an exact tie (split pot).
        Matches `src/gpu_poker/kernels/evaluator.py::eval_7`.
        """
        board_arr = np.asarray(board, dtype=np.int32)
        if board_arr.shape != (5,):
            raise ValueError(f"Expected 5 board cards, got shape {board_arr.shape}")
        h_arr = np.asarray(hole_pairs, dtype=np.int32)
        n = h_arr.shape[0]
        cards7 = np.empty((n, 7), dtype=np.int32)
        cards7[:, :2] = h_arr
        cards7[:, 2:] = board_arr[None, :]

        ranks7 = cards7 % 13
        suits7 = cards7 // 13
        primes7 = self.primes[ranks7]
        bits7 = np.left_shift(1, ranks7)

        best = np.zeros(n, dtype=np.int32)
        for i0, i1, i2, i3, i4 in _COMBOS_7C5:
            s0 = suits7[:, i0]
            is_flush = (
                (s0 == suits7[:, i1])
                & (s0 == suits7[:, i2])
                & (s0 == suits7[:, i3])
                & (s0 == suits7[:, i4])
            )
            score = np.empty(n, dtype=np.int32)
            if np.any(is_flush):
                mask = (
                    bits7[is_flush, i0]
                    | bits7[is_flush, i1]
                    | bits7[is_flush, i2]
                    | bits7[is_flush, i3]
                    | bits7[is_flush, i4]
                )
                score[is_flush] = self.flush_lut[mask]
            non_flush = ~is_flush
            if np.any(non_flush):
                prod = (
                    primes7[non_flush, i0]
                    * primes7[non_flush, i1]
                    * primes7[non_flush, i2]
                    * primes7[non_flush, i3]
                    * primes7[non_flush, i4]
                )
                score[non_flush] = self.unsuited_lut[prod]
            best = np.maximum(best, score)
        return best


def unblocked_combo_indices(dead_cards: list[int] | np.ndarray) -> np.ndarray:
    """Return sorted combo indices in `[0, 1326)` sharing no card with `dead_cards`."""
    dead = {int(x) for x in dead_cards if int(x) >= 0}
    if not dead:
        return np.arange(NUM_COMBOS, dtype=np.int32)
    mask = np.ones(NUM_COMBOS, dtype=bool)
    for d in dead:
        mask &= (d != COMBO_C0) & (d != COMBO_C1)
    return np.nonzero(mask)[0].astype(np.int32)


def _disjoint_mass(
    opp_r: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> np.ndarray:
    """Exact `O(H)` inclusion-exclusion equivalent of `d_mat @ opp_r`.

    For combo `h = (c0[h], c1[h])`, the total weight of opponent combos sharing
    no card with `h` is `sum(r) - card_sum[c0] - card_sum[c1] + r[h]`.
    """
    total = float(opp_r.sum())
    card_sum = np.bincount(c0, weights=opp_r, minlength=52) + np.bincount(
        c1, weights=opp_r, minlength=52
    )
    return (total - card_sum[c0] - card_sum[c1] + opp_r).astype(np.float32)


def build_river_matrices(
    combo_indices: np.ndarray,
    hand_ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute exact `[H, H]` showdown sign matrix `S` and disjoint mask `D`.

    For combo pair `(i, j)`:
      - `D[i, j] = 1.0` iff combos `i` and `j` share no hole cards, else `0.0`.
      - `S[i, j] = +1.0` if disjoint and `rank[i] > rank[j]`
      - `S[i, j] = -1.0` if disjoint and `rank[i] < rank[j]`
      - `S[i, j] =  0.0` if overlapping cards or exact tie (`rank[i] == rank[j]`).
    """
    c0 = COMBO_C0[combo_indices]
    c1 = COMBO_C1[combo_indices]
    disjoint = (
        (c0[:, None] != c0[None, :])
        & (c0[:, None] != c1[None, :])
        & (c1[:, None] != c0[None, :])
        & (c1[:, None] != c1[None, :])
    )
    r_cmp = np.sign(hand_ranks[:, None] - hand_ranks[None, :]).astype(np.float32)
    d_mat = disjoint.astype(np.float32)
    s_mat = r_cmp * d_mat
    return s_mat, d_mat


def _build_showdown_matrix(
    c0: np.ndarray,
    c1: np.ndarray,
    hand_ranks: np.ndarray,
) -> np.ndarray:
    """Build only the `[H, H]` showdown sign matrix `S` (disjoint folds use O(H) bincount)."""
    disjoint = (
        (c0[:, None] != c0[None, :])
        & (c0[:, None] != c1[None, :])
        & (c1[:, None] != c0[None, :])
        & (c1[:, None] != c1[None, :])
    )
    r_cmp = np.sign(hand_ranks[:, None] - hand_ranks[None, :]).astype(np.float32)
    return np.where(disjoint, r_cmp, np.float32(0.0))


def build_turn_equity_matrix(
    c0: np.ndarray,
    c1: np.ndarray,
    board4: list[int] | np.ndarray,
    evaluator: RiverHandEvaluator,
    device: str = "cpu",
) -> np.ndarray:
    """Compute exact `[H, H]` turn expected showdown sign matrix averaged over all 44 river cards.

    For every disjoint combo pair `(i, j)` on a 4-card turn board, there are `52 - 4 - 4 = 44`
    unblocked river cards. Entry `[i, j]` equals `P(i beats j) - P(j beats i)` with exact
    card-removal blockers and split pots, and is `0.0` whenever combos share a card.
    """
    b4 = [int(x) for x in board4]
    if len(b4) != 4:
        raise ValueError(f"Expected 4 turn board cards, got {len(b4)}")
    b4_set = set(b4)
    h = len(c0)
    hole_pairs = np.stack([c0, c1], axis=1)
    disjoint = (
        (c0[:, None] != c0[None, :])
        & (c0[:, None] != c1[None, :])
        & (c1[:, None] != c0[None, :])
        & (c1[:, None] != c1[None, :])
    )

    if device.startswith("cuda"):
        import torch

        dev = torch.device(device)
        s_sum_t = torch.zeros((h, h), device=dev, dtype=torch.int16)
        for r_card in range(52):
            if r_card in b4_set:
                continue
            alive = np.nonzero((c0 != r_card) & (c1 != r_card))[0]
            if len(alive) < 2:
                continue
            ranks_r = evaluator.eval_river_combos(hole_pairs[alive], [*b4, r_card])
            rk_t = torch.from_numpy(ranks_r).to(dev)
            cmp_t = torch.sign(rk_t[:, None] - rk_t[None, :]).to(dtype=torch.int16)
            idx_t = torch.from_numpy(alive.astype(np.int64)).to(dev)
            s_sum_t[idx_t[:, None], idx_t[None, :]] += cmp_t
        disj_t = torch.from_numpy(disjoint).to(dev)
        s_bar_t = torch.where(
            disj_t, s_sum_t.to(dtype=torch.float32) * (1.0 / 44.0), torch.zeros((), device=dev)
        )
        return s_bar_t.cpu().numpy()

    s_sum = np.zeros((h, h), dtype=np.int8)
    for r_card in range(52):
        if r_card in b4_set:
            continue
        alive = np.nonzero((c0 != r_card) & (c1 != r_card))[0]
        if len(alive) < 2:
            continue
        ranks_r = evaluator.eval_river_combos(hole_pairs[alive], [*b4, r_card])
        cmp_r = np.sign(ranks_r[:, None] - ranks_r[None, :]).astype(np.int8)
        s_sum[np.ix_(alive, alive)] += cmp_r
    return np.where(disjoint, s_sum.astype(np.float32) * np.float32(1.0 / 44.0), np.float32(0.0))


# ---------------------------------------------------------------------------
# River betting subgame tree & Discounted CFR solver
# ---------------------------------------------------------------------------

# Action tags inside a river subgame node: (action_type, chip_delta_or_0)
RiverAction = tuple[int, int]


@dataclass
class _RiverNode:
    kind: str  # "decision", "fold", "showdown"
    player: int  # acting player (0 or 1) for decision, or folding player for fold
    pot_half_win: float = 0.0  # chips won from opponent at terminal (relative to starting root pot)
    # Symmetric zero-sum terminal accounting (root_pot/2 committed by each player at root):
    # - Showdown (matched bets, c0 == c1 = T): P0 payoff is T * S[i, j].
    # - Fold: folding player loses their committed chips c_f; opponent wins c_f.
    c0: float = 0.0
    c1: float = 0.0
    eff_stack_left: float = 0.0  # effective stack remaining at street-close leaf (for Turn CFVNet)
    actions: list[RiverAction] = field(default_factory=list)
    children: list[_RiverNode] = field(default_factory=list)
    regret_sum: np.ndarray | None = None  # [A, H] float32
    strategy_sum: np.ndarray | None = None  # [A, H] float32


def _build_river_tree(
    *,
    street_pot: float,
    stacks: tuple[int, int],
    bets: tuple[int, int],
    active_player: int,
    actions_taken: int,
    raises_taken: int,
    last_raise: int,
    big_blind: int,
    bet_fractions: tuple[float, ...],
    max_raises: int,
    inv: tuple[int, int],
    all_in_max_spr: float | None = None,
) -> _RiverNode:
    """Recursively construct the exact street betting tree from the current state.

    `street_pot` is the symmetric pot BEFORE any bets on the current street (`0.5 * street_pot`
    committed by each seat), and `inv` is each player's total chips committed on the current
    street (including `bets` already on the felt at the root). This keeps every fold and
    showdown leaf strictly zero-sum (`c0 == c1` at every showdown) even when solving mid-street
    facing an opponent bet.
    """
    p = active_player
    opp = 1 - p
    to_call = max(0, bets[opp] - bets[p])
    pot_total = int(round(street_pot)) + inv[0] + inv[1]
    base_c0 = 0.5 * street_pot + inv[0]
    base_c1 = 0.5 * street_pot + inv[1]

    node = _RiverNode(kind="decision", player=p, c0=base_c0, c1=base_c1)

    if to_call > 0:
        # 1. FOLD
        node.actions.append((c.ACTION_FOLD, 0))
        node.children.append(_RiverNode(kind="fold", player=p, c0=base_c0, c1=base_c1))

        # 2. CALL -> closes street betting -> showdown / next-street leaf
        call_amt = min(to_call, stacks[p])
        new_inv = (inv[0] + call_amt, inv[1]) if p == 0 else (inv[0], inv[1] + call_amt)
        new_b0 = bets[0] + (call_amt if p == 0 else 0)
        new_b1 = bets[1] + (call_amt if p == 1 else 0)
        street_matched = min(new_b0, new_b1)
        uncalled0 = new_b0 - street_matched
        uncalled1 = new_b1 - street_matched
        sd_c0 = 0.5 * street_pot + (new_inv[0] - uncalled0)
        sd_c1 = 0.5 * street_pot + (new_inv[1] - uncalled1)
        rem_s0 = stacks[0] - (call_amt if p == 0 else 0) + uncalled0
        rem_s1 = stacks[1] - (call_amt if p == 1 else 0) + uncalled1
        node.actions.append((c.ACTION_CALL, 0))
        node.children.append(
            _RiverNode(
                kind="showdown",
                player=-1,
                c0=sd_c0,
                c1=sd_c1,
                eff_stack_left=float(min(rem_s0, rem_s1)),
            )
        )
    else:
        # 1. CHECK
        node.actions.append((c.ACTION_CHECK, 0))
        if actions_taken >= 1:
            node.children.append(
                _RiverNode(
                    kind="showdown",
                    player=-1,
                    c0=base_c0,
                    c1=base_c1,
                    eff_stack_left=float(min(stacks[0], stacks[1])),
                )
            )
        else:
            node.children.append(
                _build_river_tree(
                    street_pot=street_pot,
                    stacks=stacks,
                    bets=bets,
                    active_player=opp,
                    actions_taken=actions_taken + 1,
                    raises_taken=raises_taken,
                    last_raise=last_raise,
                    big_blind=big_blind,
                    bet_fractions=bet_fractions,
                    max_raises=max_raises,
                    inv=inv,
                    all_in_max_spr=all_in_max_spr,
                )
            )

    # 3. BET / RAISE options (if effective stack allows and under raise cap)
    eff_max_total = min(bets[0] + stacks[0], bets[1] + stacks[1])
    max_delta = max(0, eff_max_total - (bets[p] + to_call))
    min_delta = max(big_blind, last_raise)

    if max_delta > 0 and stacks[p] > to_call and raises_taken < max_raises:
        candidate_deltas: list[int] = []
        for frac in bet_fractions:
            d = max_delta if frac <= 0.0 else int(round(frac * pot_total))
            d = max_delta if max_delta < min_delta else max(min_delta, min(max_delta, d))
            if d > 0 and d not in candidate_deltas:
                candidate_deltas.append(d)
        # Include all-in unless suppressed on deep-SPR turn spots (avoids 50x-pot turn shoves).
        allow_all_in = (all_in_max_spr is None) or (max_delta <= all_in_max_spr * max(1, pot_total))
        if allow_all_in and max_delta not in candidate_deltas:
            candidate_deltas.append(max_delta)
        candidate_deltas.sort()

        for delta in candidate_deltas:
            commit = to_call + delta
            new_stacks = (
                (stacks[0] - commit, stacks[1]) if p == 0 else (stacks[0], stacks[1] - commit)
            )
            new_bets = (bets[0] + commit, bets[1]) if p == 0 else (bets[0], bets[1] + commit)
            new_inv = (inv[0] + commit, inv[1]) if p == 0 else (inv[0], inv[1] + commit)
            node.actions.append((c.ACTION_RAISE, delta))
            node.children.append(
                _build_river_tree(
                    street_pot=street_pot,
                    stacks=new_stacks,
                    bets=new_bets,
                    active_player=opp,
                    actions_taken=actions_taken + 1,
                    raises_taken=raises_taken + 1,
                    last_raise=max(last_raise, delta),
                    big_blind=big_blind,
                    bet_fractions=bet_fractions,
                    max_raises=max_raises,
                    inv=new_inv,
                    all_in_max_spr=all_in_max_spr,
                )
            )

    return node


def _init_node_arrays(node: _RiverNode, num_hands: int) -> None:
    if node.kind != "decision":
        return
    num_a = len(node.actions)
    node.regret_sum = np.zeros((num_a, num_hands), dtype=np.float32)
    node.strategy_sum = np.zeros((num_a, num_hands), dtype=np.float32)
    for ch in node.children:
        _init_node_arrays(ch, num_hands)


def _regret_matching(regret_sum: np.ndarray) -> np.ndarray:
    """Compute current strategy `[A, H]` via positive regret matching (uniform fallback)."""
    pos = np.maximum(regret_sum, 0.0)
    denom = pos.sum(axis=0, keepdims=True)
    num_a = regret_sum.shape[0]
    uniform = np.full_like(pos, 1.0 / float(num_a))
    return np.where(denom > 1e-12, pos / np.maximum(denom, 1e-12), uniform)


def _average_strategy(strategy_sum: np.ndarray) -> np.ndarray:
    denom = strategy_sum.sum(axis=0, keepdims=True)
    num_a = strategy_sum.shape[0]
    uniform = np.full_like(strategy_sum, 1.0 / float(num_a))
    return np.where(denom > 1e-12, strategy_sum / np.maximum(denom, 1e-12), uniform)


@dataclass
class RiverSolveResult:
    actions: list[RiverAction]
    # Root average strategy `[A, H]` over the active combo subset.
    strategy: np.ndarray
    combo_indices: np.ndarray
    exploitability_chips: float
    root_pot: float
    root_node: _RiverNode | None = None

    def action_probs_for_hand(self, hole_cards: tuple[int, int] | list[int]) -> np.ndarray:
        """Return mixed strategy `[A]` over `self.actions` for Hero's specific 2 cards."""
        cid = combo_id(hole_cards[0], hole_cards[1])
        pos = int(np.searchsorted(self.combo_indices, cid))
        if pos >= len(self.combo_indices) or int(self.combo_indices[pos]) != cid:
            raise ValueError(f"Hand {hole_cards} (combo {cid}) is blocked by the board")
        return self.strategy[:, pos].copy()


class _TurnCFVLeafEvaluator:
    """ReBeL Turn-leaf evaluator averaging `RiverCFVNet` over unblocked river runouts."""

    def __init__(
        self,
        *,
        board4: list[int],
        combo_idx: np.ndarray,
        evaluator: RiverHandEvaluator,
        cfv_net: RiverCFVNet,
        device: str = "cpu",
        sample_rivers: int = 12,
    ) -> None:
        import torch

        from gpu_poker.cfv_net import river_hand_strength_pct

        self.torch = torch
        self.device = torch.device(device)
        self.cfv_net = cfv_net.to(self.device).eval()
        self.combo_idx = combo_idx
        b4_set = set(board4)
        all_rem = [r for r in range(52) if r not in b4_set]
        # Deterministic evenly-spaced subset of river runouts for fast, low-variance evaluation.
        step = max(1, len(all_rem) // max(1, sample_rivers))
        self.river_cards = all_rem[::step][:sample_rivers]

        c0 = COMBO_C0[combo_idx]
        c1 = COMBO_C1[combo_idx]
        hole_pairs = np.stack([c0, c1], axis=1)

        pct_list: list[np.ndarray] = []
        val_list: list[np.ndarray] = []
        s_list: list[np.ndarray] = []
        alive_list: list[np.ndarray] = []
        for rc in self.river_cards:
            b5 = [*board4, rc]
            pct, valid = river_hand_strength_pct(b5, evaluator)
            pct_list.append(pct)
            val_list.append(valid)
            alive = (c0 != rc) & (c1 != rc)
            alive_list.append(alive)
            # Zero out blocked combos in per-river showdown matrix.
            ranks = evaluator.eval_river_combos(hole_pairs, b5)
            s_r = _build_showdown_matrix(c0, c1, ranks)
            s_r[~alive, :] = 0.0
            s_r[:, ~alive] = 0.0
            s_list.append(s_r)

        self.pct_t = torch.from_numpy(np.stack(pct_list, axis=0)).to(self.device)  # [K, 1326]
        self.valid_t = torch.from_numpy(np.stack(val_list, axis=0)).to(self.device)  # [K, 1326]
        self.s_mats = s_list  # K matrices of [H, H]
        self.alive_masks = alive_list  # K bool masks of [H]
        self.alive_count = np.maximum(
            1.0, np.sum(np.stack(alive_list, axis=0).astype(np.float32), axis=0)
        )
        self.idx_t = torch.from_numpy(combo_idx.astype(np.int64)).to(self.device)

    def eval_leaf(
        self,
        *,
        r0: np.ndarray,
        r1: np.ndarray,
        pot_leaf: float,
        eff_stack_left: float,
        update_player: int,
    ) -> np.ndarray:
        """Return `[H]` counterfactual value vector for `update_player` at a deep Turn leaf."""
        torch = self.torch
        k = len(self.river_cards)
        r0_full = np.zeros((k, NUM_COMBOS), dtype=np.float32)
        r1_full = np.zeros((k, NUM_COMBOS), dtype=np.float32)
        ck0_full = np.zeros((k, NUM_COMBOS), dtype=np.float32)
        ck1_full = np.zeros((k, NUM_COMBOS), dtype=np.float32)
        opp_mass = np.zeros(k, dtype=np.float32)

        for i in range(k):
            alive = self.alive_masks[i]
            rr0 = np.where(alive, r0, 0.0)
            rr1 = np.where(alive, r1, 0.0)
            m0 = float(rr0.sum())
            m1 = float(rr1.sum())
            opp_m = m1 if update_player == 0 else m0
            if opp_m <= 1e-12:
                continue
            # Even when the acting player's own reach on this river card is 0, counterfactual
            # values for unblocked combos remain non-zero whenever opponent reach is positive.
            if m0 > 1e-12:
                r0_hat = rr0 / m0
            else:
                r0_hat = alive.astype(np.float32) / max(1.0, float(alive.sum()))
            if m1 > 1e-12:
                r1_hat = rr1 / m1
            else:
                r1_hat = alive.astype(np.float32) / max(1.0, float(alive.sum()))
            r0_full[i, self.combo_idx] = r0_hat
            r1_full[i, self.combo_idx] = r1_hat
            ck0_full[i, self.combo_idx] = 0.5 * (self.s_mats[i] @ r1_hat)
            ck1_full[i, self.combo_idx] = 0.5 * (self.s_mats[i] @ r0_hat)
            opp_mass[i] = opp_m

        spr_val = float(eff_stack_left) / max(1.0, float(pot_leaf))
        with torch.no_grad():
            p0, p1 = self.cfv_net(
                r0=torch.from_numpy(r0_full).to(self.device),
                r1=torch.from_numpy(r1_full).to(self.device),
                check_cfv0=torch.from_numpy(ck0_full).to(self.device),
                check_cfv1=torch.from_numpy(ck1_full).to(self.device),
                rank_pct=self.pct_t,
                valid_mask=self.valid_t,
                spr=torch.full((k,), spr_val, device=self.device, dtype=torch.float32),
            )
            chosen = p0 if update_player == 0 else p1
            vals_k = chosen[:, self.idx_t].cpu().numpy()  # [K, H]

        # Scale each river card's normalized CFV by pot_leaf * opp_mass[i] and divide by the
        # expected number of mutually-unblocked river cards `k * (44 / 48)` (44 for all 48 cards).
        denom = float(k) * (44.0 / 48.0)
        weighted = vals_k * (opp_mass[:, None] * float(pot_leaf))
        return (weighted.sum(axis=0) / denom).astype(np.float32)


class RiverDCFRSolver:
    """Exact range-vs-range Discounted CFR solver for HUNL river & turn subgames."""

    def __init__(
        self,
        *,
        bet_fractions: tuple[float, ...] = (0.33, 0.75, 1.5),
        max_raises: int = 3,
        evaluator: RiverHandEvaluator | None = None,
        device: str = "cpu",
        cfv_net: RiverCFVNet | None = None,
        turn_all_in_max_spr: float = 3.0,
        cfv_sample_rivers: int = 12,
    ) -> None:
        self.bet_fractions = tuple(float(f) for f in bet_fractions)
        self.max_raises = int(max_raises)
        self.evaluator = evaluator or RiverHandEvaluator()
        self.device = str(device)
        self.cfv_net = cfv_net
        self.turn_all_in_max_spr = float(turn_all_in_max_spr)
        self.cfv_sample_rivers = int(cfv_sample_rivers)

    def solve(
        self,
        *,
        board: list[int] | np.ndarray,
        pot: int,
        stacks: tuple[int, int],
        bets: tuple[int, int],
        active_player: int,
        range_p0: np.ndarray,
        range_p1: np.ndarray,
        actions_taken: int = 0,
        raises_taken: int = 0,
        last_raise: int = 0,
        big_blind: int = 100,
        iterations: int = 200,
        compute_exploitability: bool = False,
        prune_eps: float = 0.0,
        keep_combo: int | None = None,
    ) -> RiverSolveResult:
        """Solve the river subgame from the current state and return root strategy."""
        combo_idx = unblocked_combo_indices(board)
        r0 = np.asarray(range_p0, dtype=np.float32)
        r1 = np.asarray(range_p1, dtype=np.float32)
        r0 = r0[combo_idx].copy() if r0.shape == (NUM_COMBOS,) else r0.copy()
        r1 = r1[combo_idx].copy() if r1.shape == (NUM_COMBOS,) else r1.copy()
        if r0.sum() <= 0 or r1.sum() <= 0:
            raise ValueError("Ranges must have positive total weight on unblocked combos")
        r0 = r0 / r0.sum()
        r1 = r1 / r1.sum()

        # Drop combos with negligible/zero weight in both ranges to shrink the showdown matrix
        # without changing any active combo's counterfactual values.
        keep = (r0 > prune_eps) | (r1 > prune_eps)
        if keep_combo is not None:
            keep = keep | (combo_idx == int(keep_combo))
        if not np.all(keep) and np.any(keep):
            combo_idx = combo_idx[keep]
            r0 = r0[keep]
            r1 = r1[keep]
            if r0.sum() > 0:
                r0 = r0 / r0.sum()
            if r1.sum() > 0:
                r1 = r1 / r1.sum()

        h_count = len(combo_idx)
        c0 = COMBO_C0[combo_idx]
        c1 = COMBO_C1[combo_idx]
        board_list = [int(x) for x in board]
        if len(board_list) == 5:
            hole_pairs = np.stack([c0, c1], axis=1)
            ranks = self.evaluator.eval_river_combos(hole_pairs, board_list)
            s_mat = _build_showdown_matrix(c0, c1, ranks)
        elif len(board_list) == 4:
            s_mat = build_turn_equity_matrix(c0, c1, board_list, self.evaluator, device=self.device)
        else:
            raise ValueError(f"Expected 4 (turn) or 5 (river) board cards, got {len(board_list)}")

        street_pot = float(pot)
        root_pot = float(pot + bets[0] + bets[1])
        is_turn = len(board_list) == 4
        root = _build_river_tree(
            street_pot=street_pot,
            stacks=(int(stacks[0]), int(stacks[1])),
            bets=(int(bets[0]), int(bets[1])),
            active_player=int(active_player),
            actions_taken=int(actions_taken),
            raises_taken=int(raises_taken),
            last_raise=int(last_raise),
            big_blind=int(big_blind),
            bet_fractions=self.bet_fractions,
            max_raises=self.max_raises,
            inv=(int(bets[0]), int(bets[1])),
            all_in_max_spr=self.turn_all_in_max_spr if is_turn else None,
        )

        leaf_eval: _TurnCFVLeafEvaluator | None = None
        if is_turn and self.cfv_net is not None:
            leaf_eval = _TurnCFVLeafEvaluator(
                board4=board_list,
                combo_idx=combo_idx,
                evaluator=self.evaluator,
                cfv_net=self.cfv_net,
                device=self.device,
                sample_rivers=self.cfv_sample_rivers,
            )

        if self.device.startswith("cuda") and leaf_eval is None:
            return self._solve_torch(
                root=root,
                combo_idx=combo_idx,
                r0=r0,
                r1=r1,
                s_mat=s_mat,
                c0=c0,
                c1=c1,
                root_pot=root_pot,
                iterations=iterations,
                compute_exploitability=compute_exploitability,
            )

        _init_node_arrays(root, h_count)

        # Alternating Discounted CFR (alpha=1.5, beta=0, gamma=2.0).
        for t in range(1, int(iterations) + 1):
            t_f = float(t)
            pos_disc = (t_f**1.5) / (t_f**1.5 + 1.0)
            strat_w = (t_f / (t_f + 1.0)) ** 2.0

            self._cfr_traverse(
                root,
                r0,
                r1,
                update_player=0,
                s_mat=s_mat,
                c0=c0,
                c1=c1,
                pos_disc=pos_disc,
                strat_w=strat_w,
                leaf_eval=leaf_eval,
            )
            self._cfr_traverse(
                root,
                r0,
                r1,
                update_player=1,
                s_mat=s_mat,
                c0=c0,
                c1=c1,
                pos_disc=pos_disc,
                strat_w=strat_w,
                leaf_eval=leaf_eval,
            )

        assert root.strategy_sum is not None
        root_strat = _average_strategy(root.strategy_sum)

        expl = 0.0
        if compute_exploitability:
            expl = self._compute_exploitability(root, r0, r1, s_mat, c0, c1)

        return RiverSolveResult(
            actions=root.actions,
            strategy=root_strat,
            combo_indices=combo_idx,
            exploitability_chips=expl,
            root_pot=root_pot,
            root_node=root,
        )

    def _solve_torch(
        self,
        *,
        root: _RiverNode,
        combo_idx: np.ndarray,
        r0: np.ndarray,
        r1: np.ndarray,
        s_mat: np.ndarray,
        c0: np.ndarray,
        c1: np.ndarray,
        root_pot: float,
        iterations: int,
        compute_exploitability: bool,
    ) -> RiverSolveResult:
        """GPU-accelerated PyTorch DCFR traversal (exact arithmetic match to CPU NumPy)."""
        import torch

        dev = torch.device(self.device)
        h_count = len(combo_idx)
        r0_t = torch.from_numpy(r0).to(dev)
        r1_t = torch.from_numpy(r1).to(dev)
        s_t = torch.from_numpy(s_mat).to(dev)
        c0_t = torch.from_numpy(c0.astype(np.int64)).to(dev)
        c1_t = torch.from_numpy(c1.astype(np.int64)).to(dev)
        zeros52 = torch.zeros(52, device=dev, dtype=torch.float32)

        reg_map: dict[int, torch.Tensor] = {}
        str_map: dict[int, torch.Tensor] = {}

        def init_t(n: _RiverNode) -> None:
            if n.kind != "decision":
                return
            num_a = len(n.actions)
            nid = id(n)
            reg_map[nid] = torch.zeros((num_a, h_count), device=dev, dtype=torch.float32)
            str_map[nid] = torch.zeros((num_a, h_count), device=dev, dtype=torch.float32)
            for ch in n.children:
                init_t(ch)

        init_t(root)

        def disjoint_t(opp_r: torch.Tensor) -> torch.Tensor:
            card_sum = zeros52.clone()
            card_sum.index_add_(0, c0_t, opp_r)
            card_sum.index_add_(0, c1_t, opp_r)
            return opp_r.sum() - card_sum[c0_t] - card_sum[c1_t] + opp_r

        def rm_t(reg: torch.Tensor) -> torch.Tensor:
            pos = reg.clamp_min(0.0)
            denom = pos.sum(dim=0, keepdim=True)
            unif = torch.full_like(pos, 1.0 / float(reg.shape[0]))
            return torch.where(denom > 1e-12, pos / denom.clamp_min(1e-12), unif)

        def traverse_t(
            n: _RiverNode,
            rr0: torch.Tensor,
            rr1: torch.Tensor,
            update_player: int,
            pos_disc: float,
            strat_w: float,
        ) -> torch.Tensor:
            if n.kind == "showdown":
                opp_r = rr1 if update_player == 0 else rr0
                t_chips = n.c0 if update_player == 0 else n.c1
                return t_chips * torch.mv(s_t, opp_r)
            if n.kind == "fold":
                opp_r = rr1 if update_player == 0 else rr0
                dm = disjoint_t(opp_r)
                if n.player == update_player:
                    loss = n.c0 if update_player == 0 else n.c1
                    return -loss * dm
                gain = n.c1 if update_player == 0 else n.c0
                return gain * dm

            nid = id(n)
            strat = rm_t(reg_map[nid])
            if n.player == update_player:
                vals = []
                for a_idx, ch in enumerate(n.children):
                    if update_player == 0:
                        vals.append(traverse_t(ch, rr0 * strat[a_idx], rr1, 0, pos_disc, strat_w))
                    else:
                        vals.append(traverse_t(ch, rr0, rr1 * strat[a_idx], 1, pos_disc, strat_w))
                action_vals = torch.stack(vals, dim=0)
                node_val = (strat * action_vals).sum(dim=0)
                reg_map[nid] = reg_map[nid].clamp_min(0.0) * pos_disc + (
                    action_vals - node_val.unsqueeze(0)
                )
                own_r = rr0 if update_player == 0 else rr1
                str_map[nid] = str_map[nid] * strat_w + strat * own_r.unsqueeze(0)
                return node_val

            total_val = torch.zeros(h_count, device=dev, dtype=torch.float32)
            for a_idx, ch in enumerate(n.children):
                if n.player == 0:
                    total_val = total_val + traverse_t(
                        ch, rr0 * strat[a_idx], rr1, update_player, pos_disc, strat_w
                    )
                else:
                    total_val = total_val + traverse_t(
                        ch, rr0, rr1 * strat[a_idx], update_player, pos_disc, strat_w
                    )
            return total_val

        with torch.no_grad():
            for t in range(1, int(iterations) + 1):
                t_f = float(t)
                pos_disc = (t_f**1.5) / (t_f**1.5 + 1.0)
                strat_w = (t_f / (t_f + 1.0)) ** 2.0
                traverse_t(root, r0_t, r1_t, 0, pos_disc, strat_w)
                traverse_t(root, r0_t, r1_t, 1, pos_disc, strat_w)

            # Copy strategy_sum back to CPU tree nodes so nested subgame stepping works uniformly.
            def sync_back(n: _RiverNode) -> None:
                if n.kind != "decision":
                    return
                n.strategy_sum = str_map[id(n)].cpu().numpy()
                for ch in n.children:
                    sync_back(ch)

            sync_back(root)

        assert root.strategy_sum is not None
        root_strat = _average_strategy(root.strategy_sum)
        expl = (
            self._compute_exploitability(root, r0, r1, s_mat, c0, c1)
            if compute_exploitability
            else 0.0
        )
        return RiverSolveResult(
            actions=root.actions,
            strategy=root_strat,
            combo_indices=combo_idx,
            exploitability_chips=expl,
            root_pot=root_pot,
            root_node=root,
        )

    def _cfr_traverse(
        self,
        node: _RiverNode,
        r0: np.ndarray,
        r1: np.ndarray,
        *,
        update_player: int,
        s_mat: np.ndarray,
        c0: np.ndarray,
        c1: np.ndarray,
        pos_disc: float,
        strat_w: float,
        leaf_eval: _TurnCFVLeafEvaluator | None = None,
    ) -> np.ndarray:
        """Return counterfactual value vector `[H]` for `update_player`."""
        if node.kind == "showdown":
            if leaf_eval is not None and node.eff_stack_left > 0.0:
                return leaf_eval.eval_leaf(
                    r0=r0,
                    r1=r1,
                    pot_leaf=node.c0 + node.c1,
                    eff_stack_left=node.eff_stack_left,
                    update_player=update_player,
                )
            opp_r = r1 if update_player == 0 else r0
            t_chips = node.c0 if update_player == 0 else node.c1
            return (t_chips * (s_mat @ opp_r)).astype(np.float32)

        if node.kind == "fold":
            opp_r = r1 if update_player == 0 else r0
            disjoint_mass = _disjoint_mass(opp_r, c0, c1)
            if node.player == update_player:
                loss = node.c0 if update_player == 0 else node.c1
                return (-loss * disjoint_mass).astype(np.float32)
            gain = node.c1 if update_player == 0 else node.c0
            return (gain * disjoint_mass).astype(np.float32)

        assert node.regret_sum is not None and node.strategy_sum is not None
        strat = _regret_matching(node.regret_sum)
        num_a = len(node.actions)

        if node.player == update_player:
            action_vals = np.empty((num_a, r0.shape[0]), dtype=np.float32)
            for a_idx, ch in enumerate(node.children):
                if update_player == 0:
                    action_vals[a_idx] = self._cfr_traverse(
                        ch,
                        r0 * strat[a_idx],
                        r1,
                        update_player=0,
                        s_mat=s_mat,
                        c0=c0,
                        c1=c1,
                        pos_disc=pos_disc,
                        strat_w=strat_w,
                        leaf_eval=leaf_eval,
                    )
                else:
                    action_vals[a_idx] = self._cfr_traverse(
                        ch,
                        r0,
                        r1 * strat[a_idx],
                        update_player=1,
                        s_mat=s_mat,
                        c0=c0,
                        c1=c1,
                        pos_disc=pos_disc,
                        strat_w=strat_w,
                        leaf_eval=leaf_eval,
                    )
            node_val = (strat * action_vals).sum(axis=0)
            inst_regret = action_vals - node_val[None, :]
            # DCFR discounting: scale positive regrets by pos_disc, zero negative regrets (beta=0).
            pos_reg = np.maximum(node.regret_sum, 0.0) * pos_disc
            node.regret_sum = pos_reg + inst_regret

            own_r = r0 if update_player == 0 else r1
            node.strategy_sum = node.strategy_sum * strat_w + strat * own_r[None, :]
            return node_val

        # Opponent node: sum counterfactual values across opponent's actions.
        total_val = np.zeros(r0.shape[0], dtype=np.float32)
        for a_idx, ch in enumerate(node.children):
            if node.player == 0:
                total_val += self._cfr_traverse(
                    ch,
                    r0 * strat[a_idx],
                    r1,
                    update_player=update_player,
                    s_mat=s_mat,
                    c0=c0,
                    c1=c1,
                    pos_disc=pos_disc,
                    strat_w=strat_w,
                    leaf_eval=leaf_eval,
                )
            else:
                total_val += self._cfr_traverse(
                    ch,
                    r0,
                    r1 * strat[a_idx],
                    update_player=update_player,
                    s_mat=s_mat,
                    c0=c0,
                    c1=c1,
                    pos_disc=pos_disc,
                    strat_w=strat_w,
                    leaf_eval=leaf_eval,
                )
        return total_val

    def _best_response_value(
        self,
        node: _RiverNode,
        opp_r: np.ndarray,
        *,
        br_player: int,
        s_mat: np.ndarray,
        c0: np.ndarray,
        c1: np.ndarray,
    ) -> np.ndarray:
        """Exact best-response counterfactual value `[H]` for `br_player` vs average strategy."""
        if node.kind == "showdown":
            t_chips = node.c0 if br_player == 0 else node.c1
            return (t_chips * (s_mat @ opp_r)).astype(np.float32)

        if node.kind == "fold":
            disjoint_mass = _disjoint_mass(opp_r, c0, c1)
            if node.player == br_player:
                loss = node.c0 if br_player == 0 else node.c1
                return (-loss * disjoint_mass).astype(np.float32)
            gain = node.c1 if br_player == 0 else node.c0
            return (gain * disjoint_mass).astype(np.float32)

        assert node.strategy_sum is not None
        if node.player == br_player:
            vals = [
                self._best_response_value(ch, opp_r, br_player=br_player, s_mat=s_mat, c0=c0, c1=c1)
                for ch in node.children
            ]
            return np.max(np.stack(vals, axis=0), axis=0)

        avg_strat = _average_strategy(node.strategy_sum)
        total = np.zeros(opp_r.shape[0], dtype=np.float32)
        for a_idx, ch in enumerate(node.children):
            total += self._best_response_value(
                ch, opp_r * avg_strat[a_idx], br_player=br_player, s_mat=s_mat, c0=c0, c1=c1
            )
        return total

    def _expected_value(
        self,
        node: _RiverNode,
        opp_r: np.ndarray,
        *,
        player: int,
        s_mat: np.ndarray,
        c0: np.ndarray,
        c1: np.ndarray,
    ) -> np.ndarray:
        """Exact expected CFV `[H]` for `player` under the solved average strategy."""
        if node.kind == "showdown":
            t_chips = node.c0 if player == 0 else node.c1
            return (t_chips * (s_mat @ opp_r)).astype(np.float32)

        if node.kind == "fold":
            disjoint_mass = _disjoint_mass(opp_r, c0, c1)
            if node.player == player:
                loss = node.c0 if player == 0 else node.c1
                return (-loss * disjoint_mass).astype(np.float32)
            gain = node.c1 if player == 0 else node.c0
            return (gain * disjoint_mass).astype(np.float32)

        assert node.strategy_sum is not None
        avg_strat = _average_strategy(node.strategy_sum)
        if node.player == player:
            total = np.zeros(opp_r.shape[0], dtype=np.float32)
            for a_idx, ch in enumerate(node.children):
                ch_val = self._expected_value(ch, opp_r, player=player, s_mat=s_mat, c0=c0, c1=c1)
                total += avg_strat[a_idx] * ch_val
            return total

        total = np.zeros(opp_r.shape[0], dtype=np.float32)
        for a_idx, ch in enumerate(node.children):
            total += self._expected_value(
                ch, opp_r * avg_strat[a_idx], player=player, s_mat=s_mat, c0=c0, c1=c1
            )
        return total

    def solve_cfvs(
        self,
        *,
        board: list[int] | np.ndarray,
        pot: int,
        eff_stack: int,
        range_p0: np.ndarray,
        range_p1: np.ndarray,
        big_blind: int = 100,
        iterations: int = 150,
        prune_eps: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve a river-entry subgame and return pot-normalized `(cfv0, cfv1, check0, check1)`.

        All four arrays have shape `[1326]` (zero on blocked combos), normalized by `pot` so
        targets are scale-invariant in `SPR = eff_stack / pot` and satisfy exact zero-sum
        `dot(r0, cfv0) + dot(r1, cfv1) == 0`.
        """
        res = self.solve(
            board=board,
            pot=pot,
            stacks=(int(eff_stack), int(eff_stack)),
            bets=(0, 0),
            active_player=0,
            range_p0=range_p0,
            range_p1=range_p1,
            actions_taken=0,
            raises_taken=0,
            last_raise=0,
            big_blind=big_blind,
            iterations=iterations,
            prune_eps=prune_eps,
        )
        assert res.root_node is not None
        idx = res.combo_indices
        c0 = COMBO_C0[idx]
        c1 = COMBO_C1[idx]
        r0 = np.asarray(range_p0, dtype=np.float32)[idx].copy()
        r1 = np.asarray(range_p1, dtype=np.float32)[idx].copy()
        r0 /= max(1e-12, float(r0.sum()))
        r1 /= max(1e-12, float(r1.sum()))

        ranks = self.evaluator.eval_river_combos(np.stack([c0, c1], axis=1), board)
        s_mat = _build_showdown_matrix(c0, c1, ranks)
        inv_pot = 1.0 / max(1.0, float(pot))

        v0 = self._expected_value(res.root_node, r1, player=0, s_mat=s_mat, c0=c0, c1=c1) * inv_pot
        v1 = self._expected_value(res.root_node, r0, player=1, s_mat=s_mat, c0=c0, c1=c1) * inv_pot
        ck0 = 0.5 * (s_mat @ r1).astype(np.float32)
        ck1 = 0.5 * (s_mat @ r0).astype(np.float32)

        cfv0_full = np.zeros(NUM_COMBOS, dtype=np.float32)
        cfv1_full = np.zeros(NUM_COMBOS, dtype=np.float32)
        ck0_full = np.zeros(NUM_COMBOS, dtype=np.float32)
        ck1_full = np.zeros(NUM_COMBOS, dtype=np.float32)
        cfv0_full[idx] = v0
        cfv1_full[idx] = v1
        ck0_full[idx] = ck0
        ck1_full[idx] = ck1
        return cfv0_full, cfv1_full, ck0_full, ck1_full

    def _compute_exploitability(
        self,
        root: _RiverNode,
        r0: np.ndarray,
        r1: np.ndarray,
        s_mat: np.ndarray,
        c0: np.ndarray,
        c1: np.ndarray,
    ) -> float:
        """Exact subgame exploitability `(BR_0 + BR_1) / (2 * Z)` in chips."""
        z = float(np.dot(r0, _disjoint_mass(r1, c0, c1)))
        if z <= 1e-12:
            return 0.0
        br0 = float(
            np.dot(r0, self._best_response_value(root, r1, br_player=0, s_mat=s_mat, c0=c0, c1=c1))
        )
        br1 = float(
            np.dot(r1, self._best_response_value(root, r0, br_player=1, s_mat=s_mat, c0=c0, c1=c1))
        )
        return max(0.0, 0.5 * (br0 + br1) / z)


class BeliefTracker:
    """Tracks per-seat Bayesian beliefs over all 1,326 2-card combinations across a hand."""

    def __init__(self, smoothing: float = 1e-4) -> None:
        self.smoothing = float(smoothing)
        self.ranges = [
            np.ones(NUM_COMBOS, dtype=np.float32) / float(NUM_COMBOS),
            np.ones(NUM_COMBOS, dtype=np.float32) / float(NUM_COMBOS),
        ]

    def reset(self) -> None:
        self.ranges[0].fill(1.0 / float(NUM_COMBOS))
        self.ranges[1].fill(1.0 / float(NUM_COMBOS))

    def mask_board(self, board_cards: list[int] | np.ndarray) -> None:
        """Zero probability mass on any combo sharing a revealed board card."""
        for d in board_cards:
            di = int(d)
            if di < 0:
                continue
            blocked = (di == COMBO_C0) | (di == COMBO_C1)
            self.ranges[0][blocked] = 0.0
            self.ranges[1][blocked] = 0.0
        for p in (0, 1):
            s = float(self.ranges[p].sum())
            if s > 0:
                self.ranges[p] /= s

    def update_action(
        self,
        player: int,
        action_likelihoods: np.ndarray,
        active_indices: np.ndarray | None = None,
    ) -> None:
        """Apply exact Bayes update `r[h] *= P(a | h)` with small smoothing floor."""
        p = int(player)
        lik = np.asarray(action_likelihoods, dtype=np.float32)
        lik = np.clip(lik, 0.0, 1.0) + self.smoothing
        if active_indices is None:
            self.ranges[p] *= lik
        else:
            self.ranges[p][active_indices] *= lik
        s = float(self.ranges[p].sum())
        if s > 1e-12:
            self.ranges[p] /= s


class RiverSubgameSession:
    """Stateful nested river subgame solver across all actions within a single river street.

    Solves the full river tree once on river entry, reuses the exact solved child node when
    the opponent takes an in-tree action (check or matching bet size), and performs Libratus-style
    continual re-solving with updated solved ranges whenever the opponent makes an off-tree bet.
    """

    def __init__(
        self,
        *,
        solver: RiverDCFRSolver,
        board: list[int],
        range_p0: np.ndarray,
        range_p1: np.ndarray,
        big_blind: int = 100,
        iterations: int = 150,
        prune_eps: float = 1e-4,
        keep_combo: int | None = None,
    ) -> None:
        self.solver = solver
        self.board = [int(x) for x in board]
        self.ranges = [
            np.asarray(range_p0, dtype=np.float32).copy(),
            np.asarray(range_p1, dtype=np.float32).copy(),
        ]
        self.big_blind = int(big_blind)
        self.iterations = int(iterations)
        self.prune_eps = float(prune_eps)
        self.keep_combo = keep_combo
        self.current_node: _RiverNode | None = None
        self.combo_indices: np.ndarray | None = None
        self.solves_run: int = 0

    def _update_range_from_node(self, node: _RiverNode, action_idx: int) -> None:
        if node.strategy_sum is None or self.combo_indices is None:
            return
        strat = _average_strategy(node.strategy_sum)
        lik = strat[action_idx] + 1e-4
        p = int(node.player)
        updated = np.zeros_like(self.ranges[p])
        updated[self.combo_indices] = self.ranges[p][self.combo_indices] * lik
        s = float(updated.sum())
        if s > 1e-12:
            self.ranges[p] = updated / s

    def observe_action(self, action_type: int, delta: int = 0) -> None:
        """Advance the cached river tree by one observed action (hero or opponent).

        If an opponent's raise does not closely match any solved branch, we still update
        their range using the closest solved raise branch and invalidate `current_node`
        so the next decision re-solves cleanly at the exact chip state.
        """
        node = self.current_node
        if node is None or node.kind != "decision":
            self.current_node = None
            return

        if action_type != c.ACTION_RAISE:
            for idx, (at, _amt) in enumerate(node.actions):
                if at == action_type:
                    self._update_range_from_node(node, idx)
                    ch = node.children[idx]
                    self.current_node = ch if ch.kind == "decision" else None
                    return
            self.current_node = None
            return

        # Raise: find closest solved raise branch.
        raise_indices = [
            (idx, amt) for idx, (at, amt) in enumerate(node.actions) if at == c.ACTION_RAISE
        ]
        if not raise_indices:
            self.current_node = None
            return
        best_idx, best_amt = min(raise_indices, key=lambda pair: abs(pair[1] - int(delta)))
        self._update_range_from_node(node, best_idx)
        tol = max(self.big_blind // 2, int(0.08 * max(1, best_amt)))
        if abs(best_amt - int(delta)) <= tol:
            ch = node.children[best_idx]
            self.current_node = ch if ch.kind == "decision" else None
        else:
            # Off-tree villain raise: force a nested re-solve at the exact chip state next turn.
            self.current_node = None

    def act(
        self,
        *,
        pot: int,
        stacks: tuple[int, int],
        bets: tuple[int, int],
        active_player: int,
        actions_taken: int,
        raises_taken: int,
        last_raise: int,
        hole_cards: list[int],
        deterministic: bool = False,
    ) -> tuple[int, int]:
        """Return `(action_type, chip_delta)` from the cached solved node or a fresh solve."""
        node = self.current_node
        if (
            node is None
            or node.kind != "decision"
            or node.player != int(active_player)
            or node.strategy_sum is None
            or self.combo_indices is None
        ):
            res = self.solver.solve(
                board=self.board,
                pot=pot,
                stacks=stacks,
                bets=bets,
                active_player=active_player,
                range_p0=self.ranges[0],
                range_p1=self.ranges[1],
                actions_taken=actions_taken,
                raises_taken=raises_taken,
                last_raise=last_raise,
                big_blind=self.big_blind,
                iterations=self.iterations,
                prune_eps=self.prune_eps,
                keep_combo=self.keep_combo,
            )
            self.solves_run += 1
            self.current_node = res.root_node
            self.combo_indices = res.combo_indices
            node = self.current_node

        assert node is not None and node.strategy_sum is not None and self.combo_indices is not None
        strat = _average_strategy(node.strategy_sum)
        cid = combo_id(hole_cards[0], hole_cards[1])
        pos = int(np.searchsorted(self.combo_indices, cid))
        if pos >= len(self.combo_indices) or int(self.combo_indices[pos]) != cid:
            raise ValueError(f"Hero hand {hole_cards} missing from solved river combo set")
        probs = strat[:, pos].copy()
        probs = probs / max(1e-12, float(probs.sum()))
        chosen_idx = (
            int(np.argmax(probs)) if deterministic else int(np.random.choice(len(probs), p=probs))
        )
        act_type, amount = node.actions[chosen_idx]
        # Advance tree past hero's own solved action (updating hero range under solved strat).
        self._update_range_from_node(node, chosen_idx)
        ch = node.children[chosen_idx]
        self.current_node = ch if ch.kind == "decision" else None
        return act_type, amount

    def handoff_to_river(
        self,
        *,
        river_card: int,
        river_solver: RiverDCFRSolver,
        iterations: int | None = None,
        prune_eps: float | None = None,
        keep_combo: int | None = None,
    ) -> RiverSubgameSession:
        """Continual re-solving handoff from a solved 4-card turn session into the 5-card river.

        Masks `river_card` from both Turn-solved ranges and returns a fresh `RiverSubgameSession`
        without re-running blueprint belief tracking on earlier streets.
        """
        if len(self.board) != 4:
            raise ValueError(
                f"handoff_to_river requires a 4-card turn board, got {len(self.board)}"
            )
        rc = int(river_card)
        blocked = (rc == COMBO_C0) | (rc == COMBO_C1)
        next_ranges: list[np.ndarray] = []
        for p in (0, 1):
            r = self.ranges[p].copy()
            r[blocked] = 0.0
            s = float(r.sum())
            if s > 1e-12:
                r /= s
            next_ranges.append(r)
        return RiverSubgameSession(
            solver=river_solver,
            board=[*self.board, rc],
            range_p0=next_ranges[0],
            range_p1=next_ranges[1],
            big_blind=self.big_blind,
            iterations=self.iterations if iterations is None else int(iterations),
            prune_eps=self.prune_eps if prune_eps is None else float(prune_eps),
            keep_combo=self.keep_combo if keep_combo is None else keep_combo,
        )
