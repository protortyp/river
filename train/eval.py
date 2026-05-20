import torch

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import (
    NUM_RAISE_BUCKETS,
    PokerPolicyNet,
    raise_bucket_to_amount,
    raise_frac_to_amount,
)
from train import bots


def _street_from_scalars(scalars: torch.Tensor) -> torch.Tensor:
    # `scalars[...,7]` is stage normalized to [0,1] by dividing by 5.0.
    # Map to int stage in [0..5], then clamp to poker streets [0..3].
    stage = (scalars[..., 7] * 5.0).round().to(dtype=torch.int64)
    return stage.clamp_(0, 3)


def _init_action_stats(device: torch.device) -> dict[int, dict[str, torch.Tensor]]:
    out: dict[int, dict[str, torch.Tensor]] = {}
    for s in range(4):
        out[s] = {
            "count": torch.zeros((), device=device, dtype=torch.int64),
            "fold": torch.zeros((), device=device, dtype=torch.int64),
            "check": torch.zeros((), device=device, dtype=torch.int64),
            "call": torch.zeros((), device=device, dtype=torch.int64),
            "raise": torch.zeros((), device=device, dtype=torch.int64),
            "rf_sum": torch.zeros((), device=device, dtype=torch.float32),
            "rf_min": torch.full((), float("inf"), device=device, dtype=torch.float32),
            "rf_max": torch.full((), float("-inf"), device=device, dtype=torch.float32),
        }
    return out


def _finalize_action_stats(stats: dict[int, dict[str, torch.Tensor]]) -> dict[str, float]:
    out: dict[str, float] = {}
    names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
    for s, nm in names.items():
        st = stats[s]
        denom = max(1, int(st["count"].item()))
        out[f"{nm}/pct_fold"] = float(st["fold"].item()) / denom
        out[f"{nm}/pct_check"] = float(st["check"].item()) / denom
        out[f"{nm}/pct_call"] = float(st["call"].item()) / denom
        out[f"{nm}/pct_raise"] = float(st["raise"].item()) / denom
        if int(st["raise"].item()) > 0:
            rf_mean = st["rf_sum"] / st["raise"].to(torch.float32)
            out[f"{nm}/raise_frac_mean"] = float(rf_mean.item())
            out[f"{nm}/raise_frac_min"] = float(st["rf_min"].item())
            out[f"{nm}/raise_frac_max"] = float(st["rf_max"].item())
        else:
            out[f"{nm}/raise_frac_mean"] = 0.0
            out[f"{nm}/raise_frac_min"] = 0.0
            out[f"{nm}/raise_frac_max"] = 0.0
    return out


@torch.no_grad()
def eval_vs_bot(
    *,
    env: WarpPokerEnv,
    policy: PokerPolicyNet,
    bot_name: str,
    steps: int = 0,
    target_hands: int = 0,
    max_steps: int = 0,
    learner_seat: int = 0,
    deterministic: bool = True,
    track_actions: bool = False,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Evaluate a policy vs a scripted bot with a chosen learner seat.

    Returns metrics in bb units where possible. When `seed` is set, stochastic
    bots (random_aggressive, loose_aggressive) use a private torch.Generator
    seeded from `seed` so the eval is reproducible without touching the global
    RNG.
    """
    device = torch.device(env.device)
    obs = env.reset()

    h0, c0 = PokerPolicyNet.init_state(env.num_envs, policy.lstm_hidden, device)
    bot_generator: torch.Generator | None = None
    if seed is not None:
        bot_generator = torch.Generator(device=device)
        bot_generator.manual_seed(int(seed))
    bot_fn = bots.get_bot_fn(bot_name, generator=bot_generator)

    bb_won = torch.zeros((), device=device, dtype=torch.float32)
    hands = torch.zeros((), device=device, dtype=torch.int64)
    stats = _init_action_stats(device) if track_actions else None
    learner_seat = int(learner_seat)
    if learner_seat not in (0, 1):
        raise ValueError(f"learner_seat must be 0 or 1, got {learner_seat}")

    if target_hands <= 0 and steps <= 0:
        raise ValueError("Provide either `steps>0` or `target_hands>0`")
    if max_steps <= 0:
        max_steps = steps if steps > 0 else 10_000_000

    it = 0
    while it < max_steps:
        pid = obs["player_id"].to(dtype=torch.int64)
        is_learner = pid.eq(learner_seat)

        action_type = torch.empty((env.num_envs,), device=device, dtype=torch.int32)
        amounts = torch.zeros((env.num_envs,), device=device, dtype=torch.int32)

        # Learner action when it's learner's turn.
        if is_learner.any():
            out = policy.forward_step(
                cards=obs["cards"][is_learner],
                scalars=obs["scalars"][is_learner],
                action_mask=obs["action_mask"][is_learner],
                h=h0[:, is_learner, :],
                c=c0[:, is_learner, :],
                terminated=None,
                deterministic=deterministic,
            )
            h0[:, is_learner, :] = out.h
            c0[:, is_learner, :] = out.c

            action_type[is_learner] = out.action_type.to(dtype=torch.int32)
            learner_amounts = raise_bucket_to_amount(
                raise_bucket=out.raise_bucket,
                pot_total=obs["pot_total"][is_learner],
                min_raise=obs["min_raise"][is_learner],
                max_raise=obs["max_raise"][is_learner],
            )
            amounts[is_learner] = learner_amounts

            if stats is not None:
                st = _street_from_scalars(obs["scalars"][is_learner])
                a = out.action_type.to(dtype=torch.int64)
                # Accumulate by street. (We track bucket idx as a proxy for the
                # raise-frac stats; the dashboard keys still use 'raise_frac' for
                # backward-compat with older eval consumers.)
                bucket_proxy = out.raise_bucket.to(dtype=torch.float32) / max(
                    1, NUM_RAISE_BUCKETS - 1
                )
                for sid in range(4):
                    m = st.eq(sid)
                    if not m.any():
                        continue
                    ss = stats[sid]
                    ss["count"] += m.sum().to(dtype=torch.int64)
                    ss["fold"] += (a[m].eq(c.ACTION_FOLD)).sum().to(dtype=torch.int64)
                    ss["check"] += (a[m].eq(c.ACTION_CHECK)).sum().to(dtype=torch.int64)
                    ss["call"] += (a[m].eq(c.ACTION_CALL)).sum().to(dtype=torch.int64)
                    rmask = a[m].eq(c.ACTION_RAISE)
                    ss["raise"] += rmask.sum().to(dtype=torch.int64)
                    if rmask.any():
                        rf = bucket_proxy[m][rmask].to(dtype=torch.float32)
                        ss["rf_sum"] += rf.sum()
                        ss["rf_min"] = torch.minimum(ss["rf_min"], rf.min())
                        ss["rf_max"] = torch.maximum(ss["rf_max"], rf.max())

        # Bot action when it's bot's turn.
        if (~is_learner).any():
            a_bot, rf_bot = bot_fn(
                {
                    "action_mask": obs["action_mask"][~is_learner],
                    "min_raise": obs["min_raise"][~is_learner],
                    "max_raise": obs["max_raise"][~is_learner],
                }
            )
            action_type[~is_learner] = a_bot
            bot_amounts = raise_frac_to_amount(
                raise_frac=rf_bot,
                min_raise=obs["min_raise"][~is_learner],
                max_raise=obs["max_raise"][~is_learner],
            )
            amounts[~is_learner] = bot_amounts
        amounts = torch.where(action_type.eq(c.ACTION_RAISE), amounts, torch.zeros_like(amounts))

        next_obs = env.step(action_type, amounts)

        # Terminal reward is P0 reward in chips; convert to learner bb with per-env bb.
        term = next_obs["terminated"].to(dtype=torch.bool)
        if term.any():
            bb = next_obs["big_blind"].to(dtype=torch.float32)
            bb_term = (next_obs["rewards"].to(dtype=torch.float32) / bb)[term]
            bb_won = bb_won + bb_term.sum() if learner_seat == 0 else bb_won - bb_term.sum()
            hands = hands + term.sum().to(dtype=torch.int64)

            # Reset agent memory at hand boundaries.
            z = torch.zeros_like(h0[:, term, :])
            h0[:, term, :] = z
            c0[:, term, :] = z

        obs = next_obs
        it += 1

        if steps > 0 and it >= steps:
            break
        if target_hands > 0 and hands.item() >= target_hands:
            break

    bb_per_hand = (bb_won / hands.to(dtype=torch.float32)).item() if hands.item() > 0 else 0.0
    bb_per_100 = bb_per_hand * 100.0
    out = {
        "bb_per_hand": float(bb_per_hand),
        "bb_per_100": float(bb_per_100),
        "hands": float(hands.item()),
    }
    if stats is not None:
        out.update({f"actions/{k}": v for k, v in _finalize_action_stats(stats).items()})
    return out


@torch.no_grad()
def eval_vs_bot_seat_swap(
    *,
    env: WarpPokerEnv,
    policy: PokerPolicyNet,
    bot_name: str,
    target_hands: int,
    max_steps: int = 0,
    deterministic: bool = False,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Evaluate vs a bot using both seat assignments and average from learner perspective.
    """
    seat0_seed = None if seed is None else int(seed)
    seat1_seed = None if seed is None else int(seed) + 1
    m0 = eval_vs_bot(
        env=env,
        policy=policy,
        bot_name=bot_name,
        target_hands=target_hands,
        max_steps=max_steps,
        learner_seat=0,
        deterministic=deterministic,
        seed=seat0_seed,
    )
    m1 = eval_vs_bot(
        env=env,
        policy=policy,
        bot_name=bot_name,
        target_hands=target_hands,
        max_steps=max_steps,
        learner_seat=1,
        deterministic=deterministic,
        seed=seat1_seed,
    )
    bb_avg = 0.5 * (float(m0["bb_per_hand"]) + float(m1["bb_per_hand"]))
    hands_total = int(m0["hands"]) + int(m1["hands"])
    return {
        "bb_per_hand": float(bb_avg),
        "bb_per_100": float(bb_avg * 100.0),
        "hands": float(hands_total),
        "bb_per_hand_p0": float(m0["bb_per_hand"]),
        "bb_per_hand_p1": float(m1["bb_per_hand"]),
    }


@torch.no_grad()
def eval_vs_snapshot_mix(
    *,
    env: WarpPokerEnv,
    policy: PokerPolicyNet,
    snapshots: list[PokerPolicyNet],
    snap_weights: torch.Tensor,
    target_hands: int,
    max_steps: int = 0,
    learner_seat: int = 0,
    deterministic: bool = False,
) -> dict[str, float]:
    """
    Evaluate a policy vs an opponent sampled from a snapshot mixture.

    Opponent policy is sampled per-hand (on `terminated`) for each env independently.
    """
    if not snapshots:
        raise ValueError("snapshots must be non-empty")
    if target_hands <= 0:
        raise ValueError("target_hands must be > 0")

    # All snapshot policies must share the LSTM hidden size with the learner so
    # the per-env (h_opp, c_opp) tensors can be indexed by opp_idx without a
    # per-snapshot reshape. A silently mis-sized snapshot would feed the wrong
    # state into pol.forward_step and either crash or produce garbage actions.
    expected_hidden = int(policy.lstm_hidden)
    for i, snap in enumerate(snapshots):
        snap_hidden = int(snap.lstm_hidden)
        if snap_hidden != expected_hidden:
            raise ValueError(
                "All snapshots must share lstm_hidden with the learner "
                f"(learner={expected_hidden}); snapshots[{i}].lstm_hidden={snap_hidden}"
            )

    device = torch.device(env.device)
    obs = env.reset()

    learner_seat = int(learner_seat)
    if learner_seat not in (0, 1):
        raise ValueError(f"learner_seat must be 0 or 1, got {learner_seat}")

    # Normalize weights (allow passing unnormalized PFSP weights).
    w = snap_weights.to(device=device, dtype=torch.float32)
    if w.numel() != len(snapshots):
        raise ValueError(f"snap_weights must have shape [{len(snapshots)}], got {tuple(w.shape)}")
    w = w.clamp_min(0.0)
    w = w / w.sum().clamp_min(1e-12)

    opp_idx = torch.multinomial(w, num_samples=env.num_envs, replacement=True).to(dtype=torch.int64)

    h_learner, c_learner = PokerPolicyNet.init_state(env.num_envs, policy.lstm_hidden, device)
    h_opp, c_opp = PokerPolicyNet.init_state(env.num_envs, policy.lstm_hidden, device)

    bb_won = torch.zeros((), device=device, dtype=torch.float32)
    hands = torch.zeros((), device=device, dtype=torch.int64)

    if max_steps <= 0:
        max_steps = 10_000_000

    it = 0
    while it < max_steps:
        pid = obs["player_id"].to(dtype=torch.int64)
        is_learner = pid.eq(learner_seat)

        action_type = torch.empty((env.num_envs,), device=device, dtype=torch.int32)
        raise_bucket = torch.zeros((env.num_envs,), device=device, dtype=torch.int64)

        if is_learner.any():
            out_learner = policy.forward_step(
                cards=obs["cards"][is_learner],
                scalars=obs["scalars"][is_learner],
                action_mask=obs["action_mask"][is_learner],
                h=h_learner[:, is_learner, :],
                c=c_learner[:, is_learner, :],
                terminated=None,
                deterministic=deterministic,
            )
            h_learner[:, is_learner, :] = out_learner.h
            c_learner[:, is_learner, :] = out_learner.c
            action_type[is_learner] = out_learner.action_type.to(dtype=torch.int32)
            raise_bucket[is_learner] = out_learner.raise_bucket

        if (~is_learner).any():
            idx = torch.nonzero((~is_learner), as_tuple=False).squeeze(-1)
            # Group by opponent snapshot id to avoid per-env python dispatch.
            for sid in torch.unique(opp_idx[idx]).tolist():
                sid_i = int(sid)
                m = opp_idx[idx].eq(sid_i)
                if not m.any():
                    continue
                sub = idx[m]
                pol = snapshots[sid_i]
                out_opp = pol.forward_step(
                    cards=obs["cards"][sub],
                    scalars=obs["scalars"][sub],
                    action_mask=obs["action_mask"][sub],
                    h=h_opp[:, sub, :],
                    c=c_opp[:, sub, :],
                    terminated=None,
                    deterministic=deterministic,
                )
                h_opp[:, sub, :] = out_opp.h
                c_opp[:, sub, :] = out_opp.c
                action_type[sub] = out_opp.action_type.to(dtype=torch.int32)
                raise_bucket[sub] = out_opp.raise_bucket

        amounts = raise_bucket_to_amount(
            raise_bucket=raise_bucket,
            pot_total=obs["pot_total"],
            min_raise=obs["min_raise"],
            max_raise=obs["max_raise"],
        )
        amounts = torch.where(action_type.eq(c.ACTION_RAISE), amounts, torch.zeros_like(amounts))

        next_obs = env.step(action_type, amounts)
        term = next_obs["terminated"].to(dtype=torch.bool)
        if term.any():
            bb = next_obs["big_blind"].to(dtype=torch.float32)
            bb_term = (next_obs["rewards"].to(dtype=torch.float32) / bb)[term]
            bb_won = bb_won + bb_term.sum() if learner_seat == 0 else bb_won - bb_term.sum()
            hands = hands + term.sum().to(dtype=torch.int64)

            # Reset memories and resample opponent for new hands.
            z_learner = torch.zeros_like(h_learner[:, term, :])
            h_learner[:, term, :] = z_learner
            c_learner[:, term, :] = z_learner
            z_opp = torch.zeros_like(h_opp[:, term, :])
            h_opp[:, term, :] = z_opp
            c_opp[:, term, :] = z_opp
            opp_idx[term] = torch.multinomial(
                w, num_samples=int(term.sum().item()), replacement=True
            )

        obs = next_obs
        it += 1
        if hands.item() >= target_hands:
            break

    bb_per_hand = (bb_won / hands.to(dtype=torch.float32)).item() if hands.item() > 0 else 0.0
    return {
        "bb_per_hand": float(bb_per_hand),
        "bb_per_100": float(bb_per_hand * 100.0),
        "hands": float(hands.item()),
    }


@torch.no_grad()
def eval_vs_snapshot_mix_seat_swap(
    *,
    env: WarpPokerEnv,
    policy: PokerPolicyNet,
    snapshots: list[PokerPolicyNet],
    snap_weights: torch.Tensor,
    target_hands: int,
    max_steps: int = 0,
    deterministic: bool = False,
) -> dict[str, float]:
    m0 = eval_vs_snapshot_mix(
        env=env,
        policy=policy,
        snapshots=snapshots,
        snap_weights=snap_weights,
        target_hands=target_hands,
        max_steps=max_steps,
        learner_seat=0,
        deterministic=deterministic,
    )
    m1 = eval_vs_snapshot_mix(
        env=env,
        policy=policy,
        snapshots=snapshots,
        snap_weights=snap_weights,
        target_hands=target_hands,
        max_steps=max_steps,
        learner_seat=1,
        deterministic=deterministic,
    )
    bb_avg = 0.5 * (float(m0["bb_per_hand"]) + float(m1["bb_per_hand"]))
    hands_total = int(m0["hands"]) + int(m1["hands"])
    return {
        "bb_per_hand": float(bb_avg),
        "bb_per_100": float(bb_avg * 100.0),
        "hands": float(hands_total),
        "bb_per_hand_p0": float(m0["bb_per_hand"]),
        "bb_per_hand_p1": float(m1["bb_per_hand"]),
    }


@torch.no_grad()
def eval_vs_policy(
    *,
    env: WarpPokerEnv,
    policy_p0: PokerPolicyNet,
    policy_p1: PokerPolicyNet,
    target_hands: int,
    max_steps: int = 0,
    deterministic: bool = True,
) -> dict[str, float]:
    """
    Evaluate policy_p0 as seat P0 vs policy_p1 as seat P1.
    """
    device = torch.device(env.device)
    obs = env.reset()

    h0, c0 = PokerPolicyNet.init_state(env.num_envs, policy_p0.lstm_hidden, device)
    h1, c1 = PokerPolicyNet.init_state(env.num_envs, policy_p1.lstm_hidden, device)

    bb_won = torch.zeros((), device=device, dtype=torch.float32)
    hands = torch.zeros((), device=device, dtype=torch.int64)

    if target_hands <= 0:
        raise ValueError("target_hands must be > 0")
    if max_steps <= 0:
        max_steps = 10_000_000

    it = 0
    while it < max_steps:
        pid = obs["player_id"].to(dtype=torch.int64)
        is_p0 = pid.eq(0)

        action_type = torch.empty((env.num_envs,), device=device, dtype=torch.int32)
        raise_bucket = torch.zeros((env.num_envs,), device=device, dtype=torch.int64)

        if is_p0.any():
            out0 = policy_p0.forward_step(
                cards=obs["cards"][is_p0],
                scalars=obs["scalars"][is_p0],
                action_mask=obs["action_mask"][is_p0],
                h=h0[:, is_p0, :],
                c=c0[:, is_p0, :],
                terminated=None,
                deterministic=deterministic,
            )
            h0[:, is_p0, :] = out0.h
            c0[:, is_p0, :] = out0.c
            action_type[is_p0] = out0.action_type.to(dtype=torch.int32)
            raise_bucket[is_p0] = out0.raise_bucket

        if (~is_p0).any():
            out1 = policy_p1.forward_step(
                cards=obs["cards"][~is_p0],
                scalars=obs["scalars"][~is_p0],
                action_mask=obs["action_mask"][~is_p0],
                h=h1[:, ~is_p0, :],
                c=c1[:, ~is_p0, :],
                terminated=None,
                deterministic=deterministic,
            )
            h1[:, ~is_p0, :] = out1.h
            c1[:, ~is_p0, :] = out1.c
            action_type[~is_p0] = out1.action_type.to(dtype=torch.int32)
            raise_bucket[~is_p0] = out1.raise_bucket

        amounts = raise_bucket_to_amount(
            raise_bucket=raise_bucket,
            pot_total=obs["pot_total"],
            min_raise=obs["min_raise"],
            max_raise=obs["max_raise"],
        )
        amounts = torch.where(action_type.eq(c.ACTION_RAISE), amounts, torch.zeros_like(amounts))

        next_obs = env.step(action_type, amounts)

        term = next_obs["terminated"].to(dtype=torch.bool)
        if term.any():
            bb = next_obs["big_blind"].to(dtype=torch.float32)
            bb_term = (next_obs["rewards"].to(dtype=torch.float32) / bb)[term]
            bb_won = bb_won + bb_term.sum()
            hands = hands + term.sum().to(dtype=torch.int64)

            z0 = torch.zeros_like(h0[:, term, :])
            h0[:, term, :] = z0
            c0[:, term, :] = z0
            z1 = torch.zeros_like(h1[:, term, :])
            h1[:, term, :] = z1
            c1[:, term, :] = z1

        obs = next_obs
        it += 1
        if hands.item() >= target_hands:
            break

    bb_per_hand = (bb_won / hands.to(dtype=torch.float32)).item() if hands.item() > 0 else 0.0
    bb_per_100 = bb_per_hand * 100.0
    return {
        "bb_per_hand": float(bb_per_hand),
        "bb_per_100": float(bb_per_100),
        "hands": float(hands.item()),
    }
