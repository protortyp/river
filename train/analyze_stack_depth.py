"""Analyze how the bot's strategy varies with stack depth.

Runs the trained ONNX policy through N self-play hands at each of several
starting stack depths (in big blinds), then reports the action distribution,
raise-bucket usage, and how often each player goes all-in.

Use it to sanity-check that randomized-stack training produced a stack-aware
policy (e.g., more all-in / fold play at 30 BB; more 3-bet pots at 200 BB).

Usage:
    uv run python train/analyze_stack_depth.py \\
        --model ~/Downloads/pokergpu_upd539.onnx \\
        --depths 30 50 100 150 200 \\
        --hands 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpu_poker import constants as const  # noqa: E402
from gpu_poker.policy import NUM_RAISE_BUCKETS  # noqa: E402

# Reuse the self-contained Python HUNL state machine + obs encoder we already
# wrote for slumbot_eval (matches Warp env byte-for-byte for the scalar obs).
from train.slumbot_eval import (  # noqa: E402
    HUNLState,
    OnnxPolicy,
    bucket_to_amount,
    encode_obs,
)


def _deal_random_hand(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    """Return (hero_hole, villain_hole, board) using a single uniform deck shuffle."""
    deck = rng.permutation(52)
    return list(deck[0:2]), list(deck[2:4]), list(deck[4:9])


def _play_one_hand(
    *,
    policy_a: OnnxPolicy,  # plays as hero (seat 0)
    policy_b: OnnxPolicy,  # plays as villain (seat 1)
    starting_stack: int,
    sb: int,
    bb: int,
    button: int,
    rng: np.random.Generator,
    stats: dict,
) -> int:
    """Play one hand, accumulate per-seat action stats. Returns hero's chip delta."""
    hole_hero, hole_villain, board = _deal_random_hand(rng)
    state = HUNLState(
        button=button,
        hero_seat=0,
        hole_cards_hero=hole_hero,
        sb=sb,
        bb=bb,
        starting_stack=starting_stack,
    )
    # The state machine only stores hero's hole cards; we need villain's hidden
    # from the policy POV but available at showdown. Stuff them in directly.
    state.hole_cards[1] = list(hole_villain)
    # Reveal whole board upfront in the underlying community_cards list; the
    # encoder masks them by stage anyway.
    state.community_cards = list(board)
    policy_a.reset_state()
    policy_b.reset_state()

    while not state.done:
        actor = policy_a if state.active_player == 0 else policy_b
        # Hide villain's hole cards from policy_a's POV by switching whose hole
        # the obs returns: encode_obs already only reads state.hole_cards[active_player].
        obs = encode_obs(state)
        action_type, raise_bucket = actor.act(obs)
        mask, min_d, max_d = state.legal_actions()
        if not mask[action_type]:
            if mask[const.ACTION_CHECK]:
                action_type = const.ACTION_CHECK
            elif mask[const.ACTION_CALL]:
                action_type = const.ACTION_CALL
            else:
                action_type = const.ACTION_FOLD
        amount = 0
        if action_type == const.ACTION_RAISE:
            amount = bucket_to_amount(
                raise_bucket,
                pot_total=int(obs["pot_total"]),
                min_raise=int(obs["min_raise"]),
                max_raise=int(obs["max_raise"]),
            )

        # Per-actor stats.
        is_a = state.active_player == 0
        bucket = stats["A"] if is_a else stats["B"]
        bucket["total"] += 1
        bucket[["fold", "check", "call", "raise"][action_type]] += 1
        if action_type == const.ACTION_RAISE:
            bucket["raise_buckets"][raise_bucket] += 1
            # All-in if delta equals the player's all-in amount.
            p = state.active_player
            tc = state.to_call(p)
            if amount >= state.stacks[p] - tc:
                bucket["allin_raises"] += 1

        state.apply(action_type, amount=amount)

    # If the hand reached showdown (we dealt all 5 board cards), evaluate.
    if state.winner is None:
        # Hand ended in fold; state._end_hand already set winner.
        pass
    # In our simple loop, fold sets state.winner. Both-all-in just terminates
    # without a winner (we didn't run the showdown evaluator here). For per-depth
    # *strategy* stats this doesn't matter — we only need the action counts.
    # Reward to hero (seat 0): use the chip delta if state.winner is set, else 0.
    if state.winner == 0:
        return state.reward_hero
    if state.winner == 1:
        return state.reward_hero
    # Both-all-in without showdown resolution: use treys to evaluate.
    return _evaluate_showdown(state)


def _evaluate_showdown(state: HUNLState) -> int:
    """Resolve a both-all-in showdown using treys. Returns hero chip delta."""
    try:
        from treys import Card, Evaluator  # type: ignore[import-untyped]
    except ImportError:
        return 0  # unable to evaluate; treat as push

    def _to_treys(c: int) -> int:
        # Our encoding: rank = c % 13 (0='2', 12='A'); suit = c // 13 (0=c, 1=d, 2=h, 3=s)
        rank_chars = "23456789TJQKA"
        suit_chars = "cdhs"
        return Card.new(rank_chars[c % 13] + suit_chars[c // 13])

    ev = Evaluator()
    board = [_to_treys(c) for c in state.community_cards if c >= 0]
    if len(board) < 5:
        # Pre-flop all-in: random-roll the remaining cards.
        used = set(state.community_cards) | set(state.hole_cards[0]) | set(state.hole_cards[1])
        remaining = [c for c in range(52) if c not in used]
        rng = np.random.default_rng()
        extra = rng.choice(remaining, size=5 - len(board), replace=False).tolist()
        board += [_to_treys(int(c)) for c in extra]
    h0 = [_to_treys(c) for c in state.hole_cards[0]]
    h1 = [_to_treys(c) for c in state.hole_cards[1]]
    rank0 = ev.evaluate(board, h0)
    rank1 = ev.evaluate(board, h1)
    # Lower rank wins in treys.
    pot = state.pot + state.bets[0] + state.bets[1]
    if rank0 < rank1:
        return pot - state.starting_stack + (state.starting_stack - state.stacks[0])
    if rank1 < rank0:
        return -(pot - state.starting_stack + (state.starting_stack - state.stacks[0]))
    return 0


def _empty_stats() -> dict:
    return {
        "A": {
            "total": 0,
            "fold": 0,
            "check": 0,
            "call": 0,
            "raise": 0,
            "allin_raises": 0,
            "raise_buckets": [0] * NUM_RAISE_BUCKETS,
        },
        "B": {
            "total": 0,
            "fold": 0,
            "check": 0,
            "call": 0,
            "raise": 0,
            "allin_raises": 0,
            "raise_buckets": [0] * NUM_RAISE_BUCKETS,
        },
    }


def _fmt_row(stats: dict, depth_bb: int) -> str:
    s = stats["A"]  # We just look at the policy when it's seat A.
    n = max(1, s["total"])
    fold = s["fold"] / n * 100
    check = s["check"] / n * 100
    call = s["call"] / n * 100
    raise_pct = s["raise"] / n * 100
    raises = max(1, s["raise"])
    allin_of_raises = s["allin_raises"] / raises * 100
    bucket_pcts = [round(b / raises * 100, 1) for b in s["raise_buckets"]]
    return (
        f"  {depth_bb:>4d} BB  | "
        f"f={fold:5.1f}% k={check:5.1f}% c={call:5.1f}% r={raise_pct:5.1f}% | "
        f"all-in-of-raises={allin_of_raises:5.1f}% | "
        f"buckets={bucket_pcts}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--depths", type=int, nargs="+", default=[30, 50, 100, 150, 200])
    p.add_argument("--hands", type=int, default=500)
    p.add_argument("--sb", type=int, default=50)
    p.add_argument("--bb", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    # Load two policies sharing the same ONNX session would force a single LSTM
    # state buffer; instantiate two so each side has its own h, c.
    policy_a = OnnxPolicy(args.model, deterministic=not args.stochastic)
    policy_b = OnnxPolicy(args.model, deterministic=not args.stochastic)

    print(f"model: {args.model}")
    print(f"{args.hands} hands per depth, blinds {args.sb}/{args.bb}")
    print()
    print("  depth   | actions                                | raise-size composition")
    print("  --------+----------------------------------------+--------------------------------")

    for depth_bb in args.depths:
        starting_stack = depth_bb * args.bb
        rng = np.random.default_rng(args.seed + depth_bb)
        stats = _empty_stats()
        for hand_i in range(args.hands):
            button = hand_i % 2  # alternate button so both seats see SB and BB
            _play_one_hand(
                policy_a=policy_a,
                policy_b=policy_b,
                starting_stack=starting_stack,
                sb=args.sb,
                bb=args.bb,
                button=button,
                rng=rng,
                stats=stats,
            )
        print(_fmt_row(stats, depth_bb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
