"""
Generate (game_state -> observation) fixtures by driving the real WarpPokerEnv.

The Warp encoder is the ground truth: a JS/TS port must reproduce its outputs
bit-for-bit, or the bot will misread the game state in the browser.

This script runs N parallel envs through K random-action hands and writes one
fixture per `env.step()` (or initial `reset()`), with both the game state and
the resulting Warp obs tensors. The TS encoder test loads this JSON, runs the
TS encoder on each state, and asserts equality against the recorded obs.

Usage:
    uv run python train/dump_obs_fixtures.py --out web/tests/fixtures.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpu_poker import constants as const  # noqa: E402
from gpu_poker.env import WarpPokerEnv  # noqa: E402


def _to_numpy(wp_array) -> np.ndarray:
    """warp.array -> numpy via torch (zero-copy where possible)."""
    import warp as wp

    return wp.to_torch(wp_array).cpu().numpy()


def extract_state(env: WarpPokerEnv) -> list[dict]:
    """Materialize the per-env game state into plain JSON-serializable dicts."""
    s = env.state

    stage = _to_numpy(s.stage)
    button = _to_numpy(s.button)
    active_player = _to_numpy(s.active_player)
    stacks = _to_numpy(s.stacks)  # [N, 2]
    bets = _to_numpy(s.bets)  # [N, 2]
    pot = _to_numpy(s.pot)
    hole_cards = _to_numpy(s.hole_cards)  # [N, 2, 2]
    community = _to_numpy(s.community_cards)  # [N, 5]
    last_raise = _to_numpy(s.last_raise)
    num_raises = _to_numpy(s.num_raises)
    last_aggressor = _to_numpy(s.last_aggressor)
    actions_this_street = _to_numpy(s.actions_this_street)
    last_action_type = _to_numpy(s.last_action_type)
    last_action_amount = _to_numpy(s.last_action_amount)
    last_action_was_raise = _to_numpy(s.last_action_was_raise)
    cfg_starting_stack = _to_numpy(s.cfg_starting_stack)
    cfg_big_blind = _to_numpy(s.cfg_big_blind)

    out = []
    for i in range(env.num_envs):
        out.append(
            {
                "stage": int(stage[i]),
                "button": int(button[i]),
                "active_player": int(active_player[i]),
                "stacks": [int(stacks[i, 0]), int(stacks[i, 1])],
                "bets": [int(bets[i, 0]), int(bets[i, 1])],
                "pot": int(pot[i]),
                "hole_cards": [
                    [int(hole_cards[i, 0, 0]), int(hole_cards[i, 0, 1])],
                    [int(hole_cards[i, 1, 0]), int(hole_cards[i, 1, 1])],
                ],
                "community_cards": [int(community[i, j]) for j in range(5)],
                "last_raise": int(last_raise[i]),
                "num_raises": int(num_raises[i]),
                "last_aggressor": int(last_aggressor[i]),
                "actions_this_street": int(actions_this_street[i]),
                "last_action_type": int(last_action_type[i]),
                "last_action_amount": int(last_action_amount[i]),
                "last_action_was_raise": int(last_action_was_raise[i]),
                "cfg_starting_stack": int(cfg_starting_stack[i]),
                "cfg_big_blind": int(cfg_big_blind[i]),
            }
        )
    return out


def extract_obs(obs_dict: dict[str, torch.Tensor]) -> list[dict]:
    cards = obs_dict["cards"].cpu().numpy()  # [N, 7] int32
    scalars = obs_dict["scalars"].cpu().numpy()  # [N, 17] f32
    mask = obs_dict["action_mask"].cpu().numpy()  # [N, 4] bool
    min_raise = obs_dict["min_raise"].cpu().numpy()  # [N]
    max_raise = obs_dict["max_raise"].cpu().numpy()  # [N]
    out = []
    n = cards.shape[0]
    for i in range(n):
        out.append(
            {
                "cards": [int(x) for x in cards[i]],
                # Round to 9 decimal places to keep JSON small but well above f32 precision.
                "scalars": [round(float(x), 9) for x in scalars[i]],
                "action_mask": [bool(x) for x in mask[i]],
                "min_raise": int(min_raise[i]),
                "max_raise": int(max_raise[i]),
            }
        )
    return out


def sample_actions(
    obs_dict: dict[str, torch.Tensor], rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick a uniformly-random legal action per env, with a random raise size."""
    mask = obs_dict["action_mask"].cpu().numpy()  # [N, 4]
    min_r = obs_dict["min_raise"].cpu().numpy()
    max_r = obs_dict["max_raise"].cpu().numpy()
    n = mask.shape[0]

    actions = np.zeros(n, dtype=np.int32)
    amounts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        legal_idx = np.where(mask[i])[0]
        if len(legal_idx) == 0:
            actions[i] = const.ACTION_FOLD
            continue
        a = int(rng.choice(legal_idx))
        actions[i] = a
        if a == const.ACTION_RAISE:
            lo, hi = int(min_r[i]), int(max_r[i])
            if hi >= lo and hi > 0:
                amounts[i] = int(rng.integers(lo, hi + 1))
            else:
                amounts[i] = hi  # all-in only
    return torch.from_numpy(actions), torch.from_numpy(amounts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=200, help="Number of env.step() iterations.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--starting-stack", type=int, default=1000)
    p.add_argument("--big-blind", type=int, default=10)
    p.add_argument("--small-blind", type=int, default=5)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = WarpPokerEnv(
        num_envs=args.num_envs,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        device="cpu",
    )

    fixtures = []

    # Snapshot at reset.
    obs = env.reset()
    states = extract_state(env)
    obs_records = extract_obs(obs)
    for s, o in zip(states, obs_records, strict=True):
        fixtures.append({"state": s, "obs": o})

    for _step_i in range(args.steps):
        actions, amounts = sample_actions(obs, rng)
        obs = env.step(actions, amounts)
        states = extract_state(env)
        obs_records = extract_obs(obs)
        for s, o in zip(states, obs_records, strict=True):
            fixtures.append({"state": s, "obs": o})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "starting_stack": args.starting_stack,
            "big_blind": args.big_blind,
            "small_blind": args.small_blind,
            "fixtures": len(fixtures),
            "constants": {
                "ACTION_FOLD": const.ACTION_FOLD,
                "ACTION_CHECK": const.ACTION_CHECK,
                "ACTION_CALL": const.ACTION_CALL,
                "ACTION_RAISE": const.ACTION_RAISE,
                "STAGE_PREFLOP": const.STAGE_PREFLOP,
                "STAGE_FLOP": const.STAGE_FLOP,
                "STAGE_TURN": const.STAGE_TURN,
                "STAGE_RIVER": const.STAGE_RIVER,
                "INVALID_CARD": const.INVALID_CARD,
                "STARTING_STACK_DEFAULT": const.STARTING_STACK,
                "BIG_BLIND_DEFAULT": const.BIG_BLIND,
            },
        },
        "fixtures": fixtures,
    }
    args.out.write_text(json.dumps(payload))
    size_kb = args.out.stat().st_size / 1024
    print(f"[fixtures] wrote {len(fixtures)} fixtures to {args.out} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
