"""Benchmark our trained policy against Slumbot via its public HTTP API.

Slumbot (Eric Jackson) is the longest-running publicly-accessible HUNL bot and
the de-facto field benchmark — virtually every published HUNL agent reports
bb/100 against it. Stake is 50/100 with 20000-chip starting stacks (200 BB).

Usage:
    uv run python train/slumbot_eval.py --model /tmp/pokergpu_upd419.onnx --hands 1000

The script:
  1. Tracks a HUNL state machine that mirrors the Warp env (so the obs we hand
     to the policy is byte-compatible with what it saw during training).
  2. Polls Slumbot's /api/new_hand and /api/act endpoints.
  3. Translates Slumbot's action strings (ACPC-ish: 'f'/'c'/'r<amount>') to
     our (action_type, raise_bucket) and back.
  4. Computes bb/100 and standard error.

It does NOT use AIVAT yet (variance reduction comes later); expect SE around
2-5 bb/100 at 1000 hands, 0.5-1.5 bb/100 at 10k hands.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the constants and the bucket fraction table the training pipeline uses,
# so this script is in lockstep with whatever the policy was trained for.
from gpu_poker import constants as const  # noqa: E402
from gpu_poker.policy import (  # noqa: E402
    ALL_IN_BUCKET,
    MIN_RAISE_BUCKET,
    NUM_RAISE_BUCKETS,
    RAISE_BUCKET_FRACTIONS,
)
from gpu_poker.search import (  # noqa: E402
    COMBO_C0,
    COMBO_C1,
    BeliefTracker,
    RiverDCFRSolver,
    combo_id,
    unblocked_combo_indices,
)

SLUMBOT_API = "https://slumbot.com/api"
SLUMBOT_SB = 50
SLUMBOT_BB = 100
SLUMBOT_STACK = 20000  # 200 BB

# Card encoding (matches src/gpu_poker/kernels/card_utils.py):
#   rank = idx % 13   (0='2', ..., 12='A')
#   suit = idx // 13  (0=c, 1=d, 2=h, 3=s)
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def str_to_card(s: str) -> int:
    if len(s) != 2:
        raise ValueError(f"bad card string: {s!r}")
    r = _RANKS.index(s[0].upper())
    su = _SUITS.index(s[1].lower())
    return r + su * 13


# ---------------------------------------------------------------------------
# HUNL state machine. Pure Python, mirrors WarpPokerEnv so the obs we produce
# matches what the policy saw during training. Mirrors web/src/engine.mjs.
# ---------------------------------------------------------------------------


class HUNLState:
    def __init__(
        self,
        *,
        button: int,
        hero_seat: int,
        hole_cards_hero: list[int],
        sb: int = SLUMBOT_SB,
        bb: int = SLUMBOT_BB,
        starting_stack: int = SLUMBOT_STACK,
    ) -> None:
        if button not in (0, 1):
            raise ValueError("button must be 0 or 1")
        self.hero_seat = int(hero_seat)
        self.button = int(button)
        self.sb_chips = int(sb)
        self.bb_chips = int(bb)
        self.starting_stack = int(starting_stack)

        sb_seat = button  # In HU, button posts SB
        bb_seat = 1 - sb_seat

        self.stacks = [starting_stack, starting_stack]
        self.bets = [0, 0]
        self.stacks[sb_seat] -= min(sb, self.stacks[sb_seat])
        self.bets[sb_seat] = min(sb, sb)
        self.stacks[bb_seat] -= min(bb, self.stacks[bb_seat])
        self.bets[bb_seat] = min(bb, bb)

        self.pot = 0  # accumulates between streets
        self.stage = const.STAGE_PREFLOP  # 0
        self.active_player = sb_seat  # SB acts first preflop in HU
        self.last_raise = bb  # initial min-raise is BB
        self.num_raises = 0
        self.last_aggressor = -1
        self.actions_this_street = 0
        self.last_action_type = -1
        self.last_action_amount = 0
        self.last_action_was_raise = 0

        # Cards: hero's hole cards are known immediately; villain's are hidden.
        # Community cards are revealed by stage.
        self.hole_cards = [[-1, -1], [-1, -1]]
        self.hole_cards[hero_seat] = list(hole_cards_hero)
        self.community_cards = [-1, -1, -1, -1, -1]

        # Hand-over bookkeeping.
        self.done = False
        self.winner = None
        self.reward_hero = 0

    # ----- queries -----

    def to_call(self, player: int) -> int:
        opp = 1 - player
        return max(0, self.bets[opp] - self.bets[player])

    def pot_total(self) -> int:
        return self.pot + self.bets[0] + self.bets[1]

    def legal_actions(self) -> tuple[list[bool], int, int]:
        p = self.active_player
        tc = self.to_call(p)
        ps = self.stacks[p]

        mask = [True, False, False, False]
        mask[const.ACTION_CHECK] = tc == 0
        mask[const.ACTION_CALL] = tc > 0 and ps > 0

        min_d = max(self.last_raise, self.bb_chips)
        max_total = min(self.bets[1 - p] + self.stacks[1 - p], self.bets[p] + ps)
        max_d = max(0, max_total - (self.bets[p] + tc))
        all_in_d = ps - tc
        mask[const.ACTION_RAISE] = (max_d >= min_d) or (0 < all_in_d <= max_d)
        return mask, min_d, max_d

    # ----- transitions -----

    def _end_hand(self, winner: int | None) -> None:
        """winner: 0/1 or -1 for split. Updates stacks and freezes state."""
        self.done = True
        pot = self.pot_total()
        if winner == -1 or winner is None:
            half = pot // 2
            self.stacks[0] += half
            self.stacks[1] += pot - half
            self.winner = -1
        else:
            self.stacks[winner] += pot
            self.winner = winner
        self.pot = 0
        self.bets = [0, 0]
        # Hero's chip delta:
        self.reward_hero = self.stacks[self.hero_seat] - self.starting_stack

    def _advance_stage(self) -> None:
        # Sweep current bets into pot.
        self.pot += self.bets[0] + self.bets[1]
        self.bets = [0, 0]
        self.last_raise = 0
        self.num_raises = 0
        self.last_aggressor = -1
        self.actions_this_street = 0
        self.last_action_type = -1
        self.last_action_amount = 0
        self.last_action_was_raise = 0
        self.stage += 1
        # Postflop: BB acts first.
        if self.stage in (const.STAGE_FLOP, const.STAGE_TURN, const.STAGE_RIVER):
            self.active_player = 1 - self.button

    def reveal_board(self, board: list[int]) -> None:
        """Slumbot tells us the board cards; we set them when the stage advances."""
        for i, c in enumerate(board):
            self.community_cards[i] = c

    def _betting_round_closed(self) -> bool:
        # Both players acted at least once this street and bets are equal.
        return self.actions_this_street >= 2 and self.bets[0] == self.bets[1]

    def apply(self, action_type: int, amount: int = 0) -> None:
        """Apply an action chosen by self.active_player."""
        if self.done:
            raise RuntimeError("hand is over")
        p = self.active_player
        opp = 1 - p
        mask, min_d, max_d = self.legal_actions()
        if not mask[action_type]:
            raise ValueError(f"illegal action {action_type} (mask={mask})")

        if action_type == const.ACTION_FOLD:
            self._end_hand(opp)
            return

        if action_type == const.ACTION_CHECK:
            self.last_action_type = const.ACTION_CHECK
            self.last_action_amount = 0
            self.last_action_was_raise = 0
            self.actions_this_street += 1
        elif action_type == const.ACTION_CALL:
            tc = self.to_call(p)
            paid = min(tc, self.stacks[p])
            self.stacks[p] -= paid
            self.bets[p] += paid
            self.last_action_type = const.ACTION_CALL
            self.last_action_amount = paid
            self.last_action_was_raise = 0
            self.actions_this_street += 1
        elif action_type == const.ACTION_RAISE:
            tc = self.to_call(p)
            delta = int(amount)
            # Clamp delta into the legal raise range (with all-in fallback for short stacks).
            all_in_d = self.stacks[p] - tc
            delta = max(min_d, min(delta, max_d)) if max_d >= min_d else max_d
            if 0 < all_in_d < min_d:
                # All-in for less than min-raise — allowed but doesn't reopen action.
                delta = all_in_d
            paid = min(tc + delta, self.stacks[p])
            self.stacks[p] -= paid
            self.bets[p] += paid
            if delta >= min_d:
                self.last_raise = delta
                self.last_aggressor = p
                self.num_raises += 1
            self.last_action_type = const.ACTION_RAISE
            self.last_action_amount = paid
            self.last_action_was_raise = 1
            self.actions_this_street += 1

        self.active_player = opp
        if self._betting_round_closed():
            self._advance_stage()
            # Both all-in? Run out the board (caller will fill community_cards).
            if self.stacks[0] == 0 and self.stacks[1] == 0:
                while self.stage < const.STAGE_RIVER:
                    self.stage += 1
                # showdown happens externally (Slumbot tells us the winner)


# ---------------------------------------------------------------------------
# Obs encoder. Mirrors src/gpu_poker/kernels/observations.py and the JS port
# in web/src/encoder.mjs. Already validated against 32k Warp fixtures via
# web/tests/encoder.test.mjs.
# ---------------------------------------------------------------------------


def _normalize(val: int, max_val: int) -> float:
    return float(val) / float(max_val)


def encode_obs(state: HUNLState) -> dict[str, np.ndarray]:
    p = state.active_player
    opp = 1 - p
    starting_stack = state.starting_stack
    bb = state.bb_chips

    # 7 visible cards for the active player.
    cards = np.full(7, -1, dtype=np.int64)
    cards[0] = state.hole_cards[p][0]
    cards[1] = state.hole_cards[p][1]
    visible = {
        const.STAGE_PREFLOP: 0,
        const.STAGE_FLOP: 3,
        const.STAGE_TURN: 4,
        const.STAGE_RIVER: 5,
    }.get(state.stage, 5)
    for i in range(min(visible, 5)):
        cards[2 + i] = state.community_cards[i]

    scalars = np.zeros(17, dtype=np.float32)
    scalars[0] = 0.0 if p == state.button else 1.0
    scalars[1] = _normalize(state.stacks[p], starting_stack)
    scalars[2] = _normalize(state.stacks[opp], starting_stack)
    pot_total = state.pot_total()
    if pot_total > 0:
        scalars[3] = state.bets[p] / pot_total
        scalars[4] = state.bets[opp] / pot_total
    scalars[5] = _normalize(pot_total, starting_stack * 2)
    tc = state.to_call(p)
    if state.stacks[p] > 0:
        scalars[6] = _normalize(min(tc, state.stacks[p]), state.stacks[p])
    scalars[7] = state.stage / 5.0
    scalars[8] = _normalize(state.last_raise, starting_stack)
    scalars[9] = state.num_raises / 4.0
    if state.last_aggressor == p:
        scalars[10] = 1.0
    elif state.last_aggressor == opp:
        scalars[10] = 0.5
    actsstreet = max(0, min(state.actions_this_street, 15))
    scalars[11] = actsstreet / 15.0
    if state.last_action_type < 0:
        scalars[12] = 0.0
    else:
        scalars[12] = (state.last_action_type + 1) / 4.0
    scalars[13] = float(state.last_action_was_raise)
    amt = max(0, state.last_action_amount)
    scalars[14] = min(1.0, amt / starting_stack)
    scalars[15] = bb / starting_stack
    eff = min(state.stacks[0], state.stacks[1])
    eff_bb_norm = (eff / bb) / 200.0 if bb > 0 else 0.0
    scalars[16] = min(1.0, eff_bb_norm)

    mask, min_d, max_d = state.legal_actions()
    action_mask = np.array(mask, dtype=bool)
    return {
        "cards": cards,
        "scalars": scalars,
        "action_mask": action_mask,
        "pot_total": np.int32(pot_total),
        "min_raise": np.int32(min_d),
        "max_raise": np.int32(max_d),
    }


# ---------------------------------------------------------------------------
# Bucket → chip amount (mirrors policy.raise_bucket_to_amount, scalar version).
# ---------------------------------------------------------------------------


def bucket_to_amount(bucket: int, pot_total: int, min_raise: int, max_raise: int) -> int:
    bucket = max(0, min(NUM_RAISE_BUCKETS - 1, int(bucket)))
    frac = RAISE_BUCKET_FRACTIONS[bucket]
    if frac == MIN_RAISE_BUCKET:
        amt = min_raise
    elif frac == ALL_IN_BUCKET:
        amt = max_raise
    else:
        amt = int(round(frac * pot_total))
    if max_raise < min_raise:
        amt = max_raise
    lo, hi = min(min_raise, max_raise), max(min_raise, max_raise)
    return max(lo, min(hi, amt))


# ---------------------------------------------------------------------------
# Slumbot action-string parser. Action history looks like:
#   "r300c/cc/r500c/c"   (slashes separate streets; tokens: 'f', 'c', 'k', 'rN')
# 'k' = check (post-flop), 'c' = call/check, 'r<n>' = raise TO <n> total chips
# in the street, 'f' = fold.
# ---------------------------------------------------------------------------


def parse_action_tokens(action_str: str) -> list[tuple[int, str]]:
    """Split into list of (street_idx, token). Each token is 'f', 'c', 'k', or 'bN'.

    Slumbot's token alphabet (Eric Jackson's bot uses 'b' for bet, not ACPC's 'r'):
      'f' = fold, 'c' = call/check, 'k' = check (postflop), 'b<N>' = bet/raise to N.
    Streets are separated by '/'.
    """
    out: list[tuple[int, str]] = []
    streets = action_str.split("/")
    for s_idx, s in enumerate(streets):
        i = 0
        while i < len(s):
            ch = s[i]
            if ch in "fck":
                out.append((s_idx, ch))
                i += 1
            elif ch in "br":  # Accept both Slumbot ('b') and ACPC ('r') for safety.
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                out.append((s_idx, "b" + s[i + 1 : j]))  # normalize to 'b<N>'
                i = j
            else:
                i += 1
    return out


def amount_to_nearest_bucket(delta: int, *, pot_total: int, min_raise: int, max_raise: int) -> int:
    """Map an arbitrary chip raise delta to the closest discrete raise bucket index."""
    best_idx = 0
    best_err = float("inf")
    for idx, frac in enumerate(RAISE_BUCKET_FRACTIONS):
        if frac == MIN_RAISE_BUCKET:
            target = min_raise
        elif frac == ALL_IN_BUCKET:
            target = max_raise
        else:
            target = round(frac * pot_total)
        target = max_raise if max_raise < min_raise else max(min_raise, min(max_raise, target))
        err = abs(int(delta) - int(target))
        if err < best_err:
            best_err = err
            best_idx = idx
    return best_idx


def replay_history(
    state: HUNLState, action_str: str, board_per_street: list[list[int]]
) -> list[tuple[int, int, int, int]]:
    """
    Re-apply Slumbot's action history to our HUNL state machine, revealing board
    cards at each street boundary, and return the public action token list
    `[(player_id, action_type, raise_bucket, stage), ...]` for transformer models.
    """
    tokens = parse_action_tokens(action_str)
    current_street = 0
    history: list[tuple[int, int, int, int]] = []
    for street_idx, tok in tokens:
        # Reveal new board cards if we just crossed a street boundary.
        while current_street < street_idx:
            current_street += 1
            if current_street < len(board_per_street):
                state.reveal_board(board_per_street[current_street])

        p = state.active_player
        stage = state.stage
        if tok == "f":
            history.append((p, const.ACTION_FOLD, 0, stage))
            state.apply(const.ACTION_FOLD)
        elif tok == "k":
            history.append((p, const.ACTION_CHECK, 0, stage))
            state.apply(const.ACTION_CHECK)
        elif tok == "c":
            tc = state.to_call(p)
            if tc == 0:
                history.append((p, const.ACTION_CHECK, 0, stage))
                state.apply(const.ACTION_CHECK)
            else:
                history.append((p, const.ACTION_CALL, 0, stage))
                state.apply(const.ACTION_CALL)
        elif tok.startswith("b") or tok.startswith("r"):
            target_total = int(tok[1:])
            tc = state.to_call(p)
            current_bet = state.bets[p]
            delta = max(0, target_total - current_bet - tc)
            _, min_r, max_r = state.legal_actions()
            pot_total = state.pot + state.bets[0] + state.bets[1]
            bkt = amount_to_nearest_bucket(
                delta, pot_total=pot_total, min_raise=min_r, max_raise=max_r
            )
            history.append((p, const.ACTION_RAISE, bkt, stage))
            state.apply(const.ACTION_RAISE, amount=delta)
        else:
            raise ValueError(f"bad token: {tok!r} in {action_str!r}")

    return history


def format_action_for_slumbot(state: HUNLState, action_type: int, amount: int) -> str:
    """Turn our (action_type, delta) into a Slumbot 'incr' string."""
    if action_type == const.ACTION_FOLD:
        return "f"
    if action_type == const.ACTION_CHECK:
        return "k"
    if action_type == const.ACTION_CALL:
        return "c"
    # RAISE: Slumbot expects the *new total chips this street*, including the call.
    p = state.active_player
    new_total = state.bets[p] + state.to_call(p) + int(amount)
    return f"b{new_total}"


# ---------------------------------------------------------------------------
# ONNX wrapper.
# ---------------------------------------------------------------------------


TRANSFORMER_MAX_TOKENS = 32
TRANSFORMER_STATE_WIDTH = TRANSFORMER_MAX_TOKENS * 4 + 1


def _canonicalize_combos_np(combo_indices: np.ndarray, board: list[int]) -> np.ndarray:
    """Suit-isomorphic canonical `[N, 7]` cards matching `policy_transformer.canonicalize_cards`."""
    n = len(combo_indices)
    cards = np.full((n, 7), -1, dtype=np.int32)
    h0 = COMBO_C0[combo_indices]
    h1 = COMBO_C1[combo_indices]
    r0 = h0 % 13
    r1 = h1 % 13
    swap = (r1 > r0) | ((r1 == r0) & (h1 > h0))
    cards[:, 0] = np.where(swap, h1, h0)
    cards[:, 1] = np.where(swap, h0, h1)
    for i, bc in enumerate(board[:5]):
        cards[:, 2 + i] = int(bc)

    valid = cards >= 0
    ranks = np.where(valid, cards % 13, 0)
    suits = np.where(valid, cards // 13, -1)
    pos = np.arange(7, dtype=np.int32)[None, :, None]
    sids = np.arange(4, dtype=np.int32)[None, None, :]
    matches = valid[:, :, None] & (suits[:, :, None] == sids)
    sentinel = np.broadcast_to(7 + sids, (n, 7, 4))
    first_pos = np.where(matches, pos, sentinel).min(axis=1)
    cmap = (first_pos[:, :, None] > first_pos[:, None, :]).sum(axis=2)
    csuits = np.take_along_axis(cmap, np.clip(suits, 0, 3), axis=1)
    return np.where(valid, csuits * 13 + ranks, -1)


class OnnxPolicy:
    def __init__(
        self,
        model_path: str,
        lstm_hidden: int | None = None,
        deterministic: bool = True,
        intra_threads: int = 0,
    ):
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        if intra_threads > 0:
            sess_opts.intra_op_num_threads = int(intra_threads)
            sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        # Auto-detect state width from ONNX input `h` shape [1, B, H] (129 = Transformer).
        h_width = 384
        for inp in self.session.get_inputs():
            if inp.name == "h" and len(inp.shape) == 3 and isinstance(inp.shape[2], int):
                h_width = int(inp.shape[2])
                break
        if lstm_hidden is not None:
            h_width = int(lstm_hidden)

        self.state_width = h_width
        self.is_transformer = h_width == TRANSFORMER_STATE_WIDTH
        self.deterministic = deterministic
        self.h = np.zeros((1, 1, self.state_width), dtype=np.float32)
        self.c = np.zeros((1, 1, self.state_width), dtype=np.float32)
        # Cache preflop action-likelihood vectors (identical across hands with the same line).
        self._preflop_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def reset_state(self) -> None:
        self.h.fill(0)
        self.c.fill(0)

    def set_transformer_history(self, history: list[tuple[int, int, int, int]]) -> None:
        """Pack up to 32 public-action tokens `(pid, act, bucket, stage)` into `self.h`."""
        if not self.is_transformer:
            return
        self.h.fill(0)
        n = min(len(history), TRANSFORMER_MAX_TOKENS)
        for i in range(n):
            pid, act, bkt, stg = history[i]
            base = i * 4
            self.h[0, 0, base + 0] = float(max(0, min(2, pid + 1)))
            self.h[0, 0, base + 1] = float(max(0, min(const.NUM_ACTIONS, act + 1)))
            self.h[0, 0, base + 2] = float(
                max(0, min(NUM_RAISE_BUCKETS, (bkt if act == const.ACTION_RAISE else 0) + 1))
            )
            self.h[0, 0, base + 3] = float(max(0, min(const.NUM_STAGES, stg + 1)))
        self.h[0, 0, TRANSFORMER_MAX_TOKENS * 4] = float(n)

    def act(self, obs: dict[str, np.ndarray]) -> tuple[int, int]:
        feeds = {
            "cards": obs["cards"].reshape(1, 7).astype(np.int64),
            "scalars": obs["scalars"].reshape(1, 17),
            "action_mask": obs["action_mask"].reshape(1, 4),
            "h": self.h,
            "c": self.c,
        }
        out = self.session.run(None, feeds)
        # Output order matches export: action_logits, raise_logits, value, h_new, c_new
        action_logits = out[0][0]
        raise_logits = out[1][0]
        self.h = out[3]
        self.c = out[4]

        if self.deterministic:
            action_type = int(np.argmax(action_logits))
            raise_bucket = int(np.argmax(raise_logits))
        else:
            # Stochastic. Need to softmax over finite logits only.
            def sample(logits):
                m = np.max(logits)
                e = np.exp(logits - m)
                s = e.sum()
                if not np.isfinite(s) or s <= 0:
                    return int(np.argmax(logits))
                p = e / s
                return int(np.random.choice(len(p), p=p))

            action_type = sample(action_logits)
            raise_bucket = sample(raise_logits)
        return action_type, raise_bucket

    def action_likelihoods_for_combos(
        self,
        *,
        combo_indices: np.ndarray,
        board: list[int],
        scalars_1d: np.ndarray,
        action_mask_1d: np.ndarray,
        history: list[tuple[int, int, int, int]],
        taken_action: int,
        taken_bucket: int,
    ) -> np.ndarray:
        """Compute `P(taken_action, taken_bucket | combo)` for all combos via canonical dedup."""
        is_preflop_full = (
            self.is_transformer and len(board) == 0 and len(combo_indices) == len(COMBO_C0)
        )
        cache_key: tuple[Any, ...] | None = None
        if is_preflop_full:
            cache_key = (
                tuple(np.round(scalars_1d, 5).tolist()),
                tuple(bool(x) for x in action_mask_1d.tolist()),
                tuple(history),
                int(taken_action),
                int(taken_bucket),
            )
            cached = self._preflop_cache.get(cache_key)
            if cached is not None:
                return cached

        if self.is_transformer:
            # Deduplicate suit-isomorphic combos (1326 -> 169 preflop, 1176 -> 344-721 suited).
            canon_cards = _canonicalize_combos_np(combo_indices, board)
            uniq_cards, inv = np.unique(canon_cards, axis=0, return_inverse=True)
            cards = uniq_cards.astype(np.int64)
            n_eval = cards.shape[0]
        else:
            n_eval = len(combo_indices)
            cards = np.full((n_eval, 7), -1, dtype=np.int64)
            cards[:, 0] = COMBO_C0[combo_indices]
            cards[:, 1] = COMBO_C1[combo_indices]
            for i, bc in enumerate(board[:5]):
                cards[:, 2 + i] = int(bc)
            inv = None

        scalars = np.broadcast_to(scalars_1d.reshape(1, 17), (n_eval, 17)).copy()
        mask = np.broadcast_to(action_mask_1d.reshape(1, 4), (n_eval, 4)).copy()

        h_one = np.zeros((1, 1, self.state_width), dtype=np.float32)
        if self.is_transformer:
            m = min(len(history), TRANSFORMER_MAX_TOKENS)
            for i in range(m):
                pid, act, bkt, stg = history[i]
                base = i * 4
                h_one[0, 0, base + 0] = float(max(0, min(2, pid + 1)))
                h_one[0, 0, base + 1] = float(max(0, min(const.NUM_ACTIONS, act + 1)))
                h_one[0, 0, base + 2] = float(
                    max(0, min(NUM_RAISE_BUCKETS, (bkt if act == const.ACTION_RAISE else 0) + 1))
                )
                h_one[0, 0, base + 3] = float(max(0, min(const.NUM_STAGES, stg + 1)))
            h_one[0, 0, TRANSFORMER_MAX_TOKENS * 4] = float(m)
        h = np.broadcast_to(h_one, (1, n_eval, self.state_width)).copy()
        c_ = np.zeros((1, n_eval, self.state_width), dtype=np.float32)

        out = self.session.run(
            None,
            {"cards": cards, "scalars": scalars, "action_mask": mask, "h": h, "c": c_},
        )
        act_logits = np.asarray(out[0], dtype=np.float64)
        act_logits = np.where(np.isfinite(act_logits), act_logits, -1e9)
        act_shift = act_logits - act_logits.max(axis=1, keepdims=True)
        act_exp = np.exp(act_shift)
        act_probs = act_exp / np.maximum(act_exp.sum(axis=1, keepdims=True), 1e-12)
        p_act = act_probs[:, int(taken_action)]

        if int(taken_action) == const.ACTION_RAISE:
            rb_logits = np.asarray(out[1], dtype=np.float64)
            rb_shift = rb_logits - rb_logits.max(axis=1, keepdims=True)
            rb_exp = np.exp(rb_shift)
            rb_probs = rb_exp / np.maximum(rb_exp.sum(axis=1, keepdims=True), 1e-12)
            p_act = p_act * rb_probs[:, int(taken_bucket)]

        res = p_act[inv].astype(np.float32) if inv is not None else p_act.astype(np.float32)
        if cache_key is not None and len(self._preflop_cache) < 512:
            self._preflop_cache[cache_key] = res
        return res


class TorchPolicy:
    """Direct PyTorch checkpoint backend with lossless dynamic PAD history trimming."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        config_path: str | None = None,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        import torch

        from gpu_poker.policy_transformer import PokerTransformerPolicyNet
        from train.export_onnx import _build_model_from_cfg, _load_arch_cfg

        self.torch = torch
        self.device = torch.device(device)
        ckpt_p = Path(checkpoint_path)
        cfg_p = Path(config_path) if config_path else None
        arch_cfg = _load_arch_cfg(ckpt_p, cfg_p)
        net = _build_model_from_cfg(arch_cfg)
        payload = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
        net.load_state_dict(sd)
        self.net = net.to(self.device).eval()
        self.is_transformer = isinstance(self.net, PokerTransformerPolicyNet)
        self.state_width = int(getattr(self.net, "lstm_hidden", TRANSFORMER_STATE_WIDTH))
        self.deterministic = deterministic
        self.h = torch.zeros((1, 1, self.state_width), device=self.device, dtype=torch.float32)
        self.c = torch.zeros((1, 1, self.state_width), device=self.device, dtype=torch.float32)
        self._preflop_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def reset_state(self) -> None:
        self.h.zero_()
        self.c.zero_()

    def _pack_history(self, history: list[tuple[int, int, int, int]]) -> np.ndarray:
        h_one = np.zeros((1, 1, self.state_width), dtype=np.float32)
        if not self.is_transformer:
            return h_one
        n = min(len(history), TRANSFORMER_MAX_TOKENS)
        for i in range(n):
            pid, act, bkt, stg = history[i]
            base = i * 4
            h_one[0, 0, base + 0] = float(max(0, min(2, pid + 1)))
            h_one[0, 0, base + 1] = float(max(0, min(const.NUM_ACTIONS, act + 1)))
            h_one[0, 0, base + 2] = float(
                max(0, min(NUM_RAISE_BUCKETS, (bkt if act == const.ACTION_RAISE else 0) + 1))
            )
            h_one[0, 0, base + 3] = float(max(0, min(const.NUM_STAGES, stg + 1)))
        h_one[0, 0, TRANSFORMER_MAX_TOKENS * 4] = float(n)
        return h_one

    def set_transformer_history(self, history: list[tuple[int, int, int, int]]) -> None:
        if not self.is_transformer:
            return
        self.h = self.torch.from_numpy(self._pack_history(history)).to(self.device)

    def act(self, obs: dict[str, np.ndarray]) -> tuple[int, int]:
        torch = self.torch
        with torch.no_grad():
            cards = torch.from_numpy(obs["cards"].reshape(1, 7).astype(np.int64)).to(self.device)
            scalars = torch.from_numpy(obs["scalars"].reshape(1, 17)).to(self.device)
            mask = torch.from_numpy(obs["action_mask"].reshape(1, 4)).to(self.device)
            out = self.net.forward_step(  # type: ignore[operator]
                cards=cards,
                scalars=scalars,
                action_mask=mask,
                h=self.h,
                c=self.c,
                deterministic=self.deterministic,
            )
            self.h = out.h
            self.c = out.c
            return int(out.action_type[0].item()), int(out.raise_bucket[0].item())

    def action_likelihoods_for_combos(
        self,
        *,
        combo_indices: np.ndarray,
        board: list[int],
        scalars_1d: np.ndarray,
        action_mask_1d: np.ndarray,
        history: list[tuple[int, int, int, int]],
        taken_action: int,
        taken_bucket: int,
    ) -> np.ndarray:
        torch = self.torch
        is_preflop_full = (
            self.is_transformer and len(board) == 0 and len(combo_indices) == len(COMBO_C0)
        )
        cache_key: tuple[Any, ...] | None = None
        if is_preflop_full:
            cache_key = (
                tuple(np.round(scalars_1d, 5).tolist()),
                tuple(bool(x) for x in action_mask_1d.tolist()),
                tuple(history),
                int(taken_action),
                int(taken_bucket),
            )
            cached = self._preflop_cache.get(cache_key)
            if cached is not None:
                return cached

        if self.is_transformer:
            canon_cards = _canonicalize_combos_np(combo_indices, board)
            uniq_cards, inv = np.unique(canon_cards, axis=0, return_inverse=True)
            cards_np = uniq_cards.astype(np.int64)
            n_eval = cards_np.shape[0]
        else:
            n_eval = len(combo_indices)
            cards_np = np.full((n_eval, 7), -1, dtype=np.int64)
            cards_np[:, 0] = COMBO_C0[combo_indices]
            cards_np[:, 1] = COMBO_C1[combo_indices]
            for i, bc in enumerate(board[:5]):
                cards_np[:, 2 + i] = int(bc)
            inv = None

        h_one = self._pack_history(history)
        with torch.no_grad():
            cards_t = torch.from_numpy(cards_np).to(self.device)
            scalars_t = (
                torch.from_numpy(scalars_1d.reshape(1, 17))
                .to(self.device)
                .expand(n_eval, 17)
                .contiguous()
            )
            mask_t = (
                torch.from_numpy(action_mask_1d.reshape(1, 4))
                .to(self.device)
                .expand(n_eval, 4)
                .contiguous()
            )
            h_t = (
                torch.from_numpy(h_one)
                .to(self.device)
                .expand(1, n_eval, self.state_width)
                .contiguous()
            )
            c_t = torch.zeros(
                (1, n_eval, self.state_width), device=self.device, dtype=torch.float32
            )
            act_t = torch.full((n_eval,), int(taken_action), device=self.device, dtype=torch.int64)
            bkt_t = torch.full((n_eval,), int(taken_bucket), device=self.device, dtype=torch.int64)
            eo = self.net.evaluate_step(  # type: ignore[operator]
                cards=cards_t,
                scalars=scalars_t,
                action_mask=mask_t,
                action_type=act_t,
                raise_bucket=bkt_t,
                h=h_t,
                c=c_t,
            )
            p_act = torch.exp(eo.logprob.to(dtype=torch.float32)).cpu().numpy()

        res = p_act[inv].astype(np.float32) if inv is not None else p_act.astype(np.float32)
        if cache_key is not None and len(self._preflop_cache) < 512:
            self._preflop_cache[cache_key] = res
        return res


def compute_river_beliefs(
    *,
    policy: OnnxPolicy | TorchPolicy,
    button: int,
    action_str: str,
    board_per_street: list[list[int]],
    min_weight_ratio: float = 0.02,
    keep_combo: int | None = None,
) -> BeliefTracker:
    """Replay public history through `policy` to compute Bayesian ranges for both seats.

    On postflop streets, combos whose current reach weight is below
    `min_weight_ratio * mean_active_weight` are skipped during batched ONNX
    inference (while keeping `keep_combo`), cutting turn/river inference cost.
    """
    bt = BeliefTracker(smoothing=5e-4)
    st = HUNLState(button=button, hero_seat=0, hole_cards_hero=[0, 1])
    tokens = parse_action_tokens(action_str)
    current_street = 0
    history: list[tuple[int, int, int, int]] = []
    cur_board: list[int] = []

    for street_idx, tok in tokens:
        while current_street < street_idx:
            current_street += 1
            if current_street < len(board_per_street):
                cur_board = board_per_street[current_street]
                st.reveal_board(cur_board)
                bt.mask_board(cur_board)

        p = st.active_player
        stage = st.stage
        obs_pub = encode_obs(st)
        active_combos = unblocked_combo_indices(cur_board)

        if tok == "f":
            act, bkt, delta = const.ACTION_FOLD, 0, 0
        elif tok == "k":
            act, bkt, delta = const.ACTION_CHECK, 0, 0
        elif tok == "c":
            tc = st.to_call(p)
            act = const.ACTION_CHECK if tc == 0 else const.ACTION_CALL
            bkt, delta = 0, 0
        elif tok.startswith("b") or tok.startswith("r"):
            target_total = int(tok[1:])
            tc = st.to_call(p)
            delta = max(0, target_total - st.bets[p] - tc)
            _, min_r, max_r = st.legal_actions()
            pot_total = st.pot + st.bets[0] + st.bets[1]
            bkt = amount_to_nearest_bucket(
                delta, pot_total=pot_total, min_raise=min_r, max_raise=max_r
            )
            act = const.ACTION_RAISE
        else:
            continue

        # On postflop streets, skip combos already pruned to near-zero probability mass.
        eval_combos = active_combos
        if len(cur_board) > 0 and min_weight_ratio > 0.0:
            w = bt.ranges[p][active_combos]
            thresh = float(w.mean()) * min_weight_ratio
            keep = w >= thresh
            if keep_combo is not None:
                keep = keep | (active_combos == int(keep_combo))
            if np.any(keep) and not np.all(keep):
                eval_combos = active_combos[keep]

        lik = policy.action_likelihoods_for_combos(
            combo_indices=eval_combos,
            board=cur_board,
            scalars_1d=obs_pub["scalars"],
            action_mask_1d=obs_pub["action_mask"],
            history=history,
            taken_action=act,
            taken_bucket=bkt,
        )
        bt.update_action(p, lik, active_indices=eval_combos)
        history.append((p, act, bkt, stage))
        st.apply(act, amount=delta)

    # Mask any final board cards revealed after the last action.
    if len(board_per_street) > 3 and board_per_street[3]:
        bt.mask_board(board_per_street[3])
    return bt


# ---------------------------------------------------------------------------
# Slumbot API client.
# ---------------------------------------------------------------------------


def slumbot_new_hand(session: requests.Session, token: str = "") -> dict[str, Any]:
    r = session.post(f"{SLUMBOT_API}/new_hand", json={"token": token}, timeout=15)
    r.raise_for_status()
    return r.json()


def slumbot_act(session: requests.Session, token: str, incr: str) -> dict[str, Any]:
    r = session.post(
        f"{SLUMBOT_API}/act",
        json={"token": token, "incr": incr},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _board_per_street(board_strs: list[str]) -> list[list[int]]:
    """Returns the cumulative board lists indexed by street (0=preflop..3=river)."""
    cards = [str_to_card(s) for s in board_strs]
    return [
        [],  # preflop
        cards[:3],  # flop (if dealt)
        cards[:4],  # turn
        cards[:5],  # river
    ]


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run(
    model_path: str | None = None,
    n_hands: int = 1000,
    deterministic: bool = True,
    verbose: bool = False,
    river_search: bool = False,
    search_iters: int = 100,
    search_max_raises: int = 2,
    intra_threads: int = 0,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
    device: str = "cpu",
) -> None:
    policy: OnnxPolicy | TorchPolicy
    if checkpoint_path is not None:
        policy = TorchPolicy(
            checkpoint_path,
            config_path=config_path,
            device=device,
            deterministic=deterministic,
        )
    elif model_path is not None:
        policy = OnnxPolicy(model_path, deterministic=deterministic, intra_threads=intra_threads)
    else:
        raise ValueError("Provide either --model (ONNX) or --checkpoint (PyTorch)")
    solver = (
        RiverDCFRSolver(bet_fractions=(0.5, 1.0), max_raises=search_max_raises)
        if river_search
        else None
    )
    s = requests.Session()

    chip_total = 0
    hands_played = 0
    errored_hands = 0
    deltas: list[int] = []
    t0 = time.perf_counter()
    token = ""
    river_solves = 0

    for hand_i in range(n_hands):
        data = slumbot_new_hand(s, token)
        if "error_msg" in data or "hole_cards" not in data:
            token = ""
            data = slumbot_new_hand(s, "")
        token = data.get("token", token)
        # Slumbot's `client_pos` indicates which seat the client is in. Empirically
        # (from observed action histories starting with a villain action when
        # client_pos=0): client_pos=0 means hero is BB; client_pos=1 means hero
        # is on the button (SB). Hero is always player 0 in our local state.
        client_pos = int(data.get("client_pos", 0))
        hero_seat = 0
        button = 1 if client_pos == 0 else 0  # 0 = hero on button, 1 = villain on button
        hole = [str_to_card(c) for c in data["hole_cards"]]
        board = data.get("board", [])
        action_str = data.get("action", "")

        state = HUNLState(
            button=button,
            hero_seat=hero_seat,
            hole_cards_hero=hole,
        )
        policy.reset_state()
        hand_errored = False

        try:
            hist = replay_history(state, action_str, _board_per_street(board))
            policy.set_transformer_history(hist)
        except Exception as e:
            print(f"[hand {hand_i}] replay failed on initial state: {e}")
            print(f"  action={action_str!r} board={board} hole={data['hole_cards']}")
            errored_hands += 1
            token = ""
            continue

        # Loop: while it's our turn, act. Then poll Slumbot for next state.
        while not state.done:
            # Slumbot signals end-of-hand by including "winnings". Check FIRST,
            # because the terminal state can have any active_player.
            if "winnings" in data and data.get("winnings") is not None:
                state.done = True
                state.reward_hero = int(data["winnings"])
                break
            # If it's not our turn at this point, something is off — Slumbot
            # should only return state when it's hero's action.
            if state.active_player != hero_seat:
                print(
                    f"[hand {hand_i}] WARNING: replay left active={state.active_player} "
                    f"hero={hero_seat} action={action_str!r} raw={data}"
                )
                hand_errored = True
                token = ""
                break

            obs = encode_obs(state)
            used_search = False
            amount = 0
            if solver is not None and state.stage == const.STAGE_RIVER and len(board) == 5:
                try:
                    bps = _board_per_street(board)
                    hero_cid = combo_id(hole[0], hole[1])
                    bt = compute_river_beliefs(
                        policy=policy,
                        button=button,
                        action_str=action_str,
                        board_per_street=bps,
                        keep_combo=hero_cid,
                    )
                    res = solver.solve(
                        board=bps[3],
                        pot=state.pot,
                        stacks=(state.stacks[0], state.stacks[1]),
                        bets=(state.bets[0], state.bets[1]),
                        active_player=state.active_player,
                        range_p0=bt.ranges[0],
                        range_p1=bt.ranges[1],
                        actions_taken=state.actions_this_street,
                        raises_taken=state.num_raises,
                        last_raise=state.last_raise,
                        big_blind=state.bb_chips,
                        iterations=search_iters,
                        prune_eps=1e-4,
                        keep_combo=hero_cid,
                    )
                    probs = res.action_probs_for_hand(state.hole_cards_hero)
                    probs = probs / max(1e-12, float(probs.sum()))
                    if deterministic:
                        chosen_idx = int(np.argmax(probs))
                    else:
                        chosen_idx = int(np.random.choice(len(probs), p=probs))
                    action_type, amount = res.actions[chosen_idx]
                    used_search = True
                    river_solves += 1
                except Exception as e:
                    if verbose:
                        print(f"  [river_search fallback]: {e}")

            if not used_search:
                action_type, raise_bucket = policy.act(obs)
                mask, min_d, max_d = state.legal_actions()
                if not mask[action_type]:
                    if mask[const.ACTION_CHECK]:
                        action_type = const.ACTION_CHECK
                    elif mask[const.ACTION_CALL]:
                        action_type = const.ACTION_CALL
                    else:
                        action_type = const.ACTION_FOLD

                if action_type == const.ACTION_RAISE:
                    amount = bucket_to_amount(
                        raise_bucket,
                        pot_total=int(obs["pot_total"]),
                        min_raise=int(obs["min_raise"]),
                        max_raise=int(obs["max_raise"]),
                    )
                    if verbose:
                        print(
                            f"  raise: bucket={raise_bucket} -> delta={amount} "
                            f"(pot={int(obs['pot_total'])}, min={min_d}, max={max_d})"
                        )

            incr = format_action_for_slumbot(state, action_type, amount)
            if verbose:
                print(f"  hero acts: {incr}")
            # Apply locally first so our state machine stays in sync, then send.
            state.apply(action_type, amount=amount)
            try:
                data = slumbot_act(s, token, incr)
            except requests.HTTPError as e:
                print(f"[hand {hand_i}] HTTP error from Slumbot: {e}")
                hand_errored = True
                token = ""
                break

            if "error_msg" in data:
                print(
                    f"[hand {hand_i}] Slumbot error on incr={incr!r} "
                    f"(prev_action={action_str!r}): {data['error_msg']}"
                )
                hand_errored = True
                token = ""
                break

            # New state from Slumbot.
            action_str = data.get("action", "")
            board = data.get("board", [])
            if "winnings" in data:
                # Resolve via Slumbot's authoritative winnings (chips for hero).
                delta = int(data["winnings"])
                state.done = True
                state.reward_hero = delta
                # Don't bother re-replaying the final villain actions; we trust Slumbot.
                break
            state = HUNLState(button=button, hero_seat=hero_seat, hole_cards_hero=hole)
            try:
                hist = replay_history(state, action_str, _board_per_street(board))
                policy.set_transformer_history(hist)
            except Exception as e:
                print(f"[hand {hand_i}] mid-hand replay failed: {e}")
                hand_errored = True
                token = ""
                break

        if hand_errored:
            errored_hands += 1
            continue

        # Slumbot's "winnings" field is the chip delta for the client.
        delta = data.get("winnings", state.reward_hero) if data else state.reward_hero
        chip_total += int(delta)
        hands_played += 1
        deltas.append(int(delta))

        if verbose or (hand_i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            mean_chips = chip_total / max(1, hands_played)
            bb_per_100 = (mean_chips / SLUMBOT_BB) * 100.0
            arr = np.array(deltas, dtype=np.float64) / SLUMBOT_BB * 100.0
            se = float(arr.std(ddof=1) / math.sqrt(max(1, len(arr)))) if len(arr) > 1 else 0.0
            print(
                f"[{hand_i + 1}/{n_hands}] elapsed={elapsed:.1f}s "
                f"bb/100={bb_per_100:+.2f} ± {se:.2f}"
            )

    elapsed = time.perf_counter() - t0
    mean_chips = chip_total / max(1, hands_played)
    bb_per_100 = (mean_chips / SLUMBOT_BB) * 100.0
    arr = np.array(deltas, dtype=np.float64) / SLUMBOT_BB * 100.0
    se = float(arr.std(ddof=1) / math.sqrt(max(1, len(arr)))) if len(arr) > 1 else 0.0
    print()
    print(f"=== final ({hands_played} hands, {elapsed:.1f}s) ===")
    print(f"mean chips/hand: {mean_chips:+.2f}")
    ci_lo = bb_per_100 - 1.96 * se
    ci_hi = bb_per_100 + 1.96 * se
    print(f"bb/100:          {bb_per_100:+.3f} ± {se:.3f}  (95% CI: {ci_lo:+.2f} .. {ci_hi:+.2f})")
    if errored_hands:
        print(f"errored hands skipped: {errored_hands}")
    if river_search:
        print(f"river DCFR solves: {river_solves}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None, help="Path to exported .onnx file.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch .pt checkpoint (faster than ONNX for --river-search).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional Hydra YAML config for --checkpoint (e.g. conf/train_rnad.yaml).",
    )
    p.add_argument("--device", type=str, default="cpu", help="Device for --checkpoint (cpu/cuda).")
    p.add_argument("--hands", type=int, default=1000)
    p.add_argument("--stochastic", action="store_true", help="Sample instead of argmax.")
    p.add_argument(
        "--river-search",
        action="store_true",
        help="Solve river spots online with Bayesian belief tracking + DCFR subgame solving.",
    )
    p.add_argument("--search-iters", type=int, default=100, help="DCFR iterations per river solve.")
    p.add_argument("--search-max-raises", type=int, default=2, help="Max raises in river subgame.")
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op CPU threads (0 = runtime default).",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    run(
        model_path=args.model,
        n_hands=int(args.hands),
        deterministic=not args.stochastic,
        verbose=args.verbose,
        river_search=bool(args.river_search),
        search_iters=int(args.search_iters),
        search_max_raises=int(args.search_max_raises),
        intra_threads=int(args.threads),
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
