"""Tests for SOTA architecture upgrades: suit canonicalization, CTDE critic,
fused step+obs kernel, duplicate-deck eval, R-NaD runner, and exploitability."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from train.config import TrainConfig
from train.eval import eval_vs_policy_duplicate
from train.exploitability import measure_exploitability
from train.train import run

from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy_transformer import (
    PokerTransformerPolicyNet,
    canonicalize_cards,
)


def test_canonicalize_cards_invariant_across_all_24_suit_permutations_and_hole_swap():
    """All 24 suit permutations and both hole-card orders map to the exact same canonical cards."""
    # Ah(38), Kd(24) on Qh(36), 7c(5), 2s(39), turn=-1, river=-1.
    base_ranks = [12, 11, 10, 5, 0, -1, -1]
    base_suits = [2, 1, 2, 0, 3, -1, -1]

    rows = []
    for perm in itertools.permutations([0, 1, 2, 3]):
        cards = [
            perm[s] * 13 + r if r >= 0 else -1 for r, s in zip(base_ranks, base_suits, strict=True)
        ]
        rows.append(cards)
        # Also include swapped hole-card order.
        swapped = [cards[1], cards[0], *cards[2:]]
        rows.append(swapped)

    batch = torch.tensor(rows, dtype=torch.int64)
    canon = canonicalize_cards(batch)
    # Every single one of the 48 isomorphic representations must be bit-identical.
    for i in range(1, canon.shape[0]):
        assert torch.equal(canon[0], canon[i]), f"Mismatch at row {i}: {canon[0]} vs {canon[i]}"
    # Unrevealed slots stay -1.
    assert canon[0, 5].item() == -1
    assert canon[0, 6].item() == -1


def test_transformer_actor_invariant_to_suit_permutation_and_opp_cards():
    """Actor logits must be suit-permutation invariant and strictly independent of opp_cards."""
    torch.manual_seed(0)
    net = PokerTransformerPolicyNet(
        scalar_dim=17, d_model=64, n_heads=4, n_layers=2, ffn_dim=128
    ).eval()

    # Ah Kd on Qh 7c 2s vs As Kc on Qs 7d 2h (swapped hole order too).
    c1 = torch.tensor([[2 * 13 + 12, 1 * 13 + 11, 2 * 13 + 10, 0 * 13 + 5, 3 * 13 + 0, -1, -1]])
    c2 = torch.tensor([[0 * 13 + 11, 3 * 13 + 12, 3 * 13 + 10, 1 * 13 + 5, 2 * 13 + 0, -1, -1]])
    scalars = torch.randn((1, 17))
    mask = torch.ones((1, 4), dtype=torch.bool)
    h, c_ = PokerTransformerPolicyNet.init_state(1, 0, torch.device("cpu"))

    opp_a = torch.tensor([[0, 1]], dtype=torch.int32)
    opp_b = torch.tensor([[50, 51]], dtype=torch.int32)

    with torch.no_grad():
        out1_a = net.forward_step(
            cards=c1,
            scalars=scalars,
            action_mask=mask,
            h=h,
            c=c_,
            deterministic=True,
            opp_cards=opp_a,
        )
        out1_b = net.forward_step(
            cards=c1,
            scalars=scalars,
            action_mask=mask,
            h=h,
            c=c_,
            deterministic=True,
            opp_cards=opp_b,
        )
        out2_no_opp = net.forward_step(
            cards=c2, scalars=scalars, action_mask=mask, h=h, c=c_, deterministic=True
        )

    # Actor must ignore opp_cards (no information leakage) and be invariant to suit permutation.
    assert torch.equal(out1_a.action_type, out1_b.action_type)
    assert torch.allclose(out1_a.logprob, out1_b.logprob, atol=1e-6)
    assert torch.allclose(out1_a.logprob, out2_no_opp.logprob, atol=1e-5)
    # Omniscient critic must actually depend on opp_cards.
    assert not torch.allclose(out1_a.value, out1_b.value, atol=1e-5)
    # Decentralized value head must remain invariant to opp_cards.
    assert out1_a.decentralized_value is not None and out1_b.decentralized_value is not None
    assert torch.allclose(out1_a.decentralized_value, out1_b.decentralized_value, atol=1e-6)


def test_env_opp_cards_matches_non_acting_seat_and_seed_rng_reproduces():
    env = WarpPokerEnv(num_envs=16, starting_stack=1000, small_blind=5, big_blind=10, device="cpu")
    seeds = np.arange(100, 116, dtype=np.uint32)
    buttons = np.zeros(16, dtype=np.int32)

    env.seed_rng(seeds, buttons=buttons)
    obs1 = env.reset()
    cards1 = obs1["cards"].clone()
    opp1 = obs1["opp_cards"].clone()

    # opp_cards must be valid distinct cards from hero hole cards.
    for i in range(16):
        hero_set = {int(cards1[i, 0]), int(cards1[i, 1])}
        opp_set = {int(opp1[i, 0]), int(opp1[i, 1])}
        assert len(hero_set | opp_set) == 4
        assert all(0 <= x < 52 for x in opp_set)

    # Re-seeding reproduces identical deals bit-for-bit.
    env.seed_rng(seeds, buttons=buttons)
    obs2 = env.reset()
    assert torch.equal(cards1, obs2["cards"])
    assert torch.equal(opp1, obs2["opp_cards"])


def test_duplicate_deck_eval_identical_deterministic_policy_is_exact_zero():
    env = WarpPokerEnv(num_envs=32, starting_stack=1000, small_blind=5, big_blind=10, device="cpu")
    net = PokerTransformerPolicyNet(
        scalar_dim=17, d_model=32, n_heads=4, n_layers=1, ffn_dim=64
    ).eval()
    res = eval_vs_policy_duplicate(
        env=env,
        policy_a=net,
        policy_b=net,
        target_hands=128,
        max_steps=2000,
        deterministic=True,
        seed=7,
    )
    assert res["hands"] >= 256
    assert abs(res["bb_per_hand"]) < 1e-6


def test_rnad_runner_cpu_smoke(tmp_path):
    cfg = TrainConfig(
        device="cpu",
        num_envs=64,
        rollout_steps=8,
        total_env_steps=64 * 8 * 2,
        num_updates=0,
        lr=3e-4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.005,
        max_grad_norm=1.0,
        card_embed_dim=16,
        mlp_dim=32,
        torso_layers=1,
        lstm_hidden=32,
        head_layers=0,
        head_dim=None,
        num_epochs=1,
        minibatch_size=128,
        seed=0,
        log_every_env_steps=1,
        log_every_seconds=1.0,
        checkpoint_dir=str(tmp_path / "rnad_ckpt"),
        save_every_env_steps=512,
        resume_from="",
        eval_every=2,
        eval_hands=64,
        eval_max_steps=2000,
        league_enabled=False,
        league_eta_selfplay=1.0,
        league_bot_initial_score=0.0,
        league_eval_bots=False,
        league_eval_bots_every=100,
        league_snapshot_pool=0,
        league_snapshot_every_updates=0,
        league_eval_every_updates=0,
        league_eval_hands=0,
        league_eval_subset=0,
        league_eval_deterministic=False,
        league_eval_seat_swap=True,
        league_pfsp_temperature=1.0,
        league_pfsp_epsilon=0.0,
        league_rollout_snapshot_k=0,
        selfplay_eval_enabled=False,
        selfplay_eval_every_updates=0,
        selfplay_eval_hands=0,
        selfplay_eval_subset=0,
        selfplay_eval_deterministic=False,
        selfplay_eval_seat_swap=True,
        timing_sync=False,
        compile_policy=False,
        use_amp=False,
        amp_dtype="bfloat16",
        use_cuda_graph=False,
        policy_backbone="transformer",
        transformer_d_model=32,
        transformer_n_heads=4,
        transformer_n_layers=1,
        transformer_ffn_dim=64,
        rnad_enabled=True,
        rnad_eta=0.2,
        rnad_reg_update_every=1,
        use_omniscient_critic=True,
        mlflow_enabled=False,
    )
    res = run(cfg)
    assert res.env_steps >= 64 * 8 * 2
    assert res.opt_steps > 0
    assert "vs_init_bb_per_100" in res.eval


def test_exploitability_cpu_smoke():
    target = PokerTransformerPolicyNet(scalar_dim=17, d_model=32, n_heads=4, n_layers=1, ffn_dim=64)
    res = measure_exploitability(
        target_net=target,
        device="cpu",
        num_envs=32,
        rollout_steps=8,
        updates=2,
        eval_hands=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        ffn_dim=64,
        seed=0,
    )
    assert res["hands"] >= 128
    assert np.isfinite(res["exploitability_bb_per_100"])


def test_dynamic_history_trim_matches_full_sequence():
    """Trimming trailing PAD history slots must match full-32-token evaluation."""
    torch.manual_seed(9)
    net = PokerTransformerPolicyNet(
        scalar_dim=17, d_model=32, n_heads=4, n_layers=2, ffn_dim=64
    ).eval()
    cards = torch.randint(-1, 52, (4, 7), dtype=torch.int32)
    scalars = torch.randn((4, 17))
    tokens, lengths = PokerTransformerPolicyNet.init_raw_state(4, torch.device("cpu"))
    tokens[:, 0] = torch.tensor([1, 4, 3, 1], dtype=torch.int32)
    tokens[:, 1] = torch.tensor([2, 3, 1, 1], dtype=torch.int32)
    lengths.fill_(2)

    with torch.no_grad():
        lat_trimmed, _ = net.encode_state(
            cards=cards, scalars=scalars, tokens=tokens, lengths=lengths
        )
        tokens_full = torch.cat([tokens, tokens[:1]], dim=0)
        lengths_full = torch.cat([lengths, torch.tensor([32], dtype=torch.int32)], dim=0)
        cards_full = torch.cat([cards, cards[:1]], dim=0)
        scalars_full = torch.cat([scalars, scalars[:1]], dim=0)
        lat_full, _ = net.encode_state(
            cards=cards_full, scalars=scalars_full, tokens=tokens_full, lengths=lengths_full
        )
    assert torch.allclose(lat_trimmed, lat_full[:4], atol=1e-5)
