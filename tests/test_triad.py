"""Tests for the AlphaStar triad controller.

This file grows incrementally as the T1-T11 implementation lands. As of
T1 (controller skeleton), only the construction-time invariants are
testable; per-role behavior tests (sampling, mutate, snapshot triggers,
gradient isolation) land alongside their respective tasks.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from train.league import OpponentMeta
from train.train import TrainConfig
from train.triad import (
    OPP_LIVE_MAIN,
    OPP_SELF,
    OPP_SNAPSHOT,
    ROLE_LE,
    ROLE_MAIN,
    ROLE_ME,
    ROLE_TAG_LE_HISTORICAL,
    ROLE_TAG_MAIN_CURRENT,
    ROLE_TAG_MAIN_HISTORICAL,
    ROLE_TAG_ME_HISTORICAL,
    ROLES,
    TriadController,
    _compute_env_partition,
    _forgotten_main_historical,
    _pfsp_sample_indices,
    _role_specs_from_cfg,
    _snapshot_indices_by_role,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _base_cfg(**overrides) -> TrainConfig:
    """Construct a CPU-friendly TrainConfig with triad enabled by default.

    Keeps fields tiny (small num_envs / rollout / minibatch) so tests
    run in <1s on CPU.
    """
    defaults = {
        "device": "cpu",
        "num_envs": 512,
        "rollout_steps": 8,
        "total_env_steps": 512 * 8,
        "num_updates": 0,
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 1.0,
        "card_embed_dim": 16,
        "mlp_dim": 32,
        "torso_layers": 1,
        "lstm_hidden": 32,
        "head_layers": 0,
        "head_dim": None,
        "num_epochs": 1,
        "minibatch_size": 64,
        "seed": 0,
        "log_every_env_steps": 1,
        "log_every_seconds": 1.0,
        "checkpoint_dir": "checkpoints",
        "save_every_env_steps": 0,
        "resume_from": "",
        "eval_every": 0,
        "eval_hands": 0,
        "eval_max_steps": 0,
        "league_enabled": False,
        "league_eta_selfplay": 1.0,
        "league_bot_initial_score": 0.0,
        "league_eval_bots": True,
        "league_eval_bots_every": 1,
        "league_snapshot_pool": 0,
        "league_snapshot_every_updates": 0,
        "league_eval_every_updates": 0,
        "league_eval_hands": 0,
        "league_eval_subset": 0,
        "league_eval_deterministic": True,
        "league_eval_seat_swap": True,
        "league_pfsp_temperature": 1.0,
        "league_pfsp_epsilon": 0.0,
        "league_rollout_snapshot_k": 0,
        "selfplay_eval_enabled": False,
        "selfplay_eval_every_updates": 0,
        "selfplay_eval_hands": 0,
        "selfplay_eval_subset": 0,
        "selfplay_eval_deterministic": True,
        "selfplay_eval_seat_swap": True,
        "timing_sync": False,
        "compile_policy": False,
        "use_amp": False,
        "amp_dtype": "bfloat16",
        "use_cuda_graph": False,
        "policy_backbone": "transformer",
        "transformer_d_model": 32,
        "transformer_n_heads": 4,
        "transformer_n_layers": 1,
        "transformer_ffn_dim": 64,
        "triad_enabled": True,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_dummy_net_factory(cfg: TrainConfig, scalar_dim: int = 17, device=None):
    """Returns a callable that builds a fresh tiny transformer policy.

    Mirrors what train.py's _make_policy does, scaled down. Lives here
    to keep test_triad.py free of train.py imports beyond TrainConfig
    (avoiding heavy dep load for fast unit tests).
    """
    from gpu_poker.policy_transformer import PokerTransformerPolicyNet

    dev = device or torch.device(cfg.device)

    def _factory():
        return PokerTransformerPolicyNet(
            scalar_dim=scalar_dim,
            d_model=cfg.transformer_d_model,
            n_heads=cfg.transformer_n_heads,
            n_layers=cfg.transformer_n_layers,
            ffn_dim=cfg.transformer_ffn_dim,
        ).to(dev)

    return _factory


# --------------------------------------------------------------------------- #
# T1: RoleSpec + env partition + TriadController.__init__
# --------------------------------------------------------------------------- #


def test_role_specs_have_correct_mutate_kinds():
    """Sanity: each role gets the right mutate behaviour from cfg."""
    cfg = _base_cfg()
    specs = _role_specs_from_cfg(cfg)
    assert specs[ROLE_MAIN].mutate_kind == "never"
    assert specs[ROLE_ME].mutate_kind == "reset_to_main"
    assert specs[ROLE_LE].mutate_kind == "random_with_prob"
    assert specs[ROLE_LE].mutate_prob == pytest.approx(0.5)


def test_role_specs_resolve_hparam_overrides():
    """Per-role hparam fields override global; None inherits global."""
    cfg = _base_cfg(triad_lr_me=1e-5, triad_entropy_le=0.05)
    specs = _role_specs_from_cfg(cfg)
    # ME LR overridden
    assert specs[ROLE_ME].lr == pytest.approx(1e-5)
    # M LR inherits global
    assert specs[ROLE_MAIN].lr == pytest.approx(cfg.lr)
    # LE entropy overridden
    assert specs[ROLE_LE].entropy_coef == pytest.approx(0.05)
    # ME entropy inherits global
    assert specs[ROLE_ME].entropy_coef == pytest.approx(cfg.entropy_coef)


def test_env_partition_contiguous_and_exact():
    """Slices partition [0, num_envs) exactly with no gaps or overlaps."""
    cfg = _base_cfg(num_envs=1000)
    specs = _role_specs_from_cfg(cfg)
    part = _compute_env_partition(cfg.num_envs, specs)
    main_lo, main_hi = part[ROLE_MAIN]
    me_lo, me_hi = part[ROLE_ME]
    le_lo, le_hi = part[ROLE_LE]
    assert main_lo == 0
    assert main_hi == me_lo
    assert me_hi == le_lo
    assert le_hi == cfg.num_envs
    # Roughly the requested fractions (LE absorbs slack).
    assert (main_hi - main_lo) == 600
    assert (me_hi - me_lo) == 200
    assert (le_hi - le_lo) == 200


def test_env_partition_absorbs_int_truncation_into_le():
    """When fractions don't yield whole envs, LE gets the remainder."""
    cfg = _base_cfg(num_envs=999)
    specs = _role_specs_from_cfg(cfg)
    part = _compute_env_partition(cfg.num_envs, specs)
    main_n = part[ROLE_MAIN][1] - part[ROLE_MAIN][0]
    me_n = part[ROLE_ME][1] - part[ROLE_ME][0]
    le_n = part[ROLE_LE][1] - part[ROLE_LE][0]
    assert main_n == int(999 * 0.6)  # 599
    assert me_n == int(999 * 0.2)  # 199
    assert le_n == 999 - main_n - me_n  # 201 (absorbs slack)
    assert main_n + me_n + le_n == 999


def test_triad_controller_init_creates_three_independent_nets():
    """T1 core invariant: 3 nets allocated, distinct objects, correct shapes."""
    cfg = _base_cfg()
    device = torch.device("cpu")
    factory = _make_dummy_net_factory(cfg, scalar_dim=17, device=device)
    triad = TriadController(cfg=cfg, scalar_dim=17, device=device, make_net=factory)

    # All three roles present.
    assert set(triad.roles.keys()) == set(ROLES)

    # Nets are distinct instances (not the same object reused).
    nets = [triad.roles[r].net for r in ROLES]
    assert len({id(n) for n in nets}) == 3

    # Each has its own optimizer pointing at its own parameters.
    for role in ROLES:
        rs = triad.roles[role]
        opt_params = {id(p) for g in rs.opt.param_groups for p in g["params"]}
        net_params = {id(p) for p in rs.net.parameters()}
        assert opt_params == net_params, f"{role} optimizer param set differs from net"


def test_triad_controller_init_per_role_buffers_and_state_shapes():
    """Buffers and hidden-state tensors are sized per the env slice."""
    cfg = _base_cfg(num_envs=512)
    device = torch.device("cpu")
    factory = _make_dummy_net_factory(cfg, scalar_dim=17, device=device)
    triad = TriadController(cfg=cfg, scalar_dim=17, device=device, make_net=factory)

    expected_slice_n = {
        ROLE_MAIN: int(512 * 0.6),  # 307
        ROLE_ME: int(512 * 0.2),  # 102
        ROLE_LE: 512 - int(512 * 0.6) - int(512 * 0.2),  # 103
    }
    for role in ROLES:
        rs = triad.roles[role]
        slice_n = expected_slice_n[role]
        assert rs.slice_n == slice_n
        # Buffer has T rollout_steps, N envs, scalar_dim, hidden.
        assert rs.buf.cards.shape == (cfg.rollout_steps, slice_n, 7)
        assert rs.buf.scalars.shape == (cfg.rollout_steps, slice_n, 17)
        assert rs.buf.h.shape == (cfg.rollout_steps, slice_n, rs.net.lstm_hidden)
        # Per-seat hidden state.
        assert rs.h0.shape == (1, slice_n, rs.net.lstm_hidden)
        assert rs.c0.shape == (1, slice_n, rs.net.lstm_hidden)
        assert rs.h1.shape == (1, slice_n, rs.net.lstm_hidden)
        assert rs.c1.shape == (1, slice_n, rs.net.lstm_hidden)
        # Hidden states zero-initialized.
        assert torch.all(rs.h0 == 0)


def test_triad_controller_env_slice_is_contiguous_python_slice():
    """env_slice() returns slices that exactly cover [0, num_envs)."""
    cfg = _base_cfg(num_envs=512)
    device = torch.device("cpu")
    factory = _make_dummy_net_factory(cfg, scalar_dim=17, device=device)
    triad = TriadController(cfg=cfg, scalar_dim=17, device=device, make_net=factory)
    main_s = triad.env_slice(ROLE_MAIN)
    me_s = triad.env_slice(ROLE_ME)
    le_s = triad.env_slice(ROLE_LE)
    assert main_s.start == 0
    assert main_s.stop == me_s.start
    assert me_s.stop == le_s.start
    assert le_s.stop == cfg.num_envs


def test_triad_controller_init_from_resume_loads_identical_weights():
    """D8: single .pt resume fans out to all 3 nets identically."""
    cfg = _base_cfg()
    device = torch.device("cpu")
    factory = _make_dummy_net_factory(cfg, scalar_dim=17, device=device)
    triad = TriadController(cfg=cfg, scalar_dim=17, device=device, make_net=factory)

    # Build a fake checkpoint from one of the nets (any net's state_dict).
    src_state = triad.roles[ROLE_MAIN].net.state_dict()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save({"model": src_state, "env_steps": 0, "update": 0}, tmp.name)
        triad.init_from_resume(tmp.name)

    # All three nets now have the same parameters as the checkpoint.
    for role in ROLES:
        rs_state = triad.roles[role].net.state_dict()
        for k in src_state:
            assert torch.equal(rs_state[k], src_state[k]), f"{role}/{k} mismatch"


def test_triad_stubs_raise_not_implemented():
    """Documented stubs raise NotImplementedError -- catches accidental
    silent fall-through during integration of later tasks.
    """
    cfg = _base_cfg()
    device = torch.device("cpu")
    factory = _make_dummy_net_factory(cfg, scalar_dim=17, device=device)
    triad = TriadController(cfg=cfg, scalar_dim=17, device=device, make_net=factory)
    for fn in (
        # sample_opponents implemented in T3 (covered by sampling tests).
        # ppo_update / ppo_update_role implemented in T4.
        # maybe_snapshot / mutate_after_snapshot implemented in T5.
        lambda: triad.forward_step(ROLE_MAIN),
        lambda: triad.schedule_async_eval(0),
    ):
        with pytest.raises(NotImplementedError):
            fn()


def test_role_state_is_mutable_dataclass():
    """RoleState fields are mutable (controller updates eval_wr,
    snapshot_count, etc. in place during training).
    """
    cfg = _base_cfg()
    device = torch.device("cpu")
    factory = _make_dummy_net_factory(cfg, scalar_dim=17, device=device)
    triad = TriadController(cfg=cfg, scalar_dim=17, device=device, make_net=factory)
    rs = triad.role(ROLE_ME)
    # snapshot_count should be incrementable
    rs.snapshot_count += 1
    assert triad.role(ROLE_ME).snapshot_count == 1
    # eval_wr_vs_targets is a dict, can be written
    rs.eval_wr_vs_targets["main_current"] = 0.42
    assert triad.role(ROLE_ME).eval_wr_vs_targets["main_current"] == pytest.approx(0.42)


def test_role_spec_is_frozen():
    """RoleSpec is immutable (it's resolved config, not state)."""
    cfg = _base_cfg()
    specs = _role_specs_from_cfg(cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        specs[ROLE_MAIN].lr = 1.0  # type: ignore[misc]


def test_unknown_role_raises_keyerror():
    cfg = _base_cfg()
    device = torch.device("cpu")
    factory = _make_dummy_net_factory(cfg, scalar_dim=17, device=device)
    triad = TriadController(cfg=cfg, scalar_dim=17, device=device, make_net=factory)
    with pytest.raises(KeyError):
        triad.role("not_a_role")
    with pytest.raises(KeyError):
        triad.env_slice("not_a_role")


# --------------------------------------------------------------------------- #
# T3: sample_opponents helpers + per-role distributions
# --------------------------------------------------------------------------- #


def _make_pool(
    *,
    n_main_historical: int = 0,
    n_le_historical: int = 0,
    n_me_historical: int = 0,
    include_current_main: bool = False,
    include_bots: int = 0,
    score_value: float = 0.0,
    main_historical_start_env_steps: int = 1_000,
) -> tuple[list[OpponentMeta], torch.Tensor]:
    """Build a synthetic opponent pool for sampling tests.

    Returns ``(metas, scores)`` where ``scores`` is a 1-D float tensor
    aligned with ``metas``. All snapshots are tagged with their role; bots
    are interleaved at the head. ``main_historical_start_env_steps`` makes
    "older vs newer" deterministic for forgotten-set tests.
    """
    metas: list[OpponentMeta] = []
    for i in range(include_bots):
        metas.append(OpponentMeta(id=f"bot_{i}", kind="bot"))
    if include_current_main:
        metas.append(
            OpponentMeta(
                id="snap_main_current",
                kind="snapshot",
                agent_role=ROLE_TAG_MAIN_CURRENT,
                env_steps=10_000_000,
                update=0,
            )
        )
    for i in range(n_main_historical):
        metas.append(
            OpponentMeta(
                id=f"snap_main_hist_{i}",
                kind="snapshot",
                agent_role=ROLE_TAG_MAIN_HISTORICAL,
                env_steps=main_historical_start_env_steps * (i + 1),
                update=i,
            )
        )
    for i in range(n_le_historical):
        metas.append(
            OpponentMeta(
                id=f"snap_le_hist_{i}",
                kind="snapshot",
                agent_role=ROLE_TAG_LE_HISTORICAL,
                env_steps=500_000 * (i + 1),
                update=i,
            )
        )
    for i in range(n_me_historical):
        metas.append(
            OpponentMeta(
                id=f"snap_me_hist_{i}",
                kind="snapshot",
                agent_role=ROLE_TAG_ME_HISTORICAL,
                env_steps=300_000 * (i + 1),
                update=i,
            )
        )
    scores = torch.full((len(metas),), float(score_value), dtype=torch.float32)
    return metas, scores


def _make_triad(cfg: TrainConfig | None = None, **overrides) -> TriadController:
    cfg = cfg or _base_cfg(**overrides)
    device = torch.device("cpu")
    factory = _make_dummy_net_factory(cfg, scalar_dim=17, device=device)
    return TriadController(cfg=cfg, scalar_dim=17, device=device, make_net=factory)


def test_snapshot_indices_by_role_excludes_bots_and_wrong_roles():
    metas, _ = _make_pool(n_main_historical=3, n_le_historical=2, include_bots=2)
    main_idx = _snapshot_indices_by_role(metas, {ROLE_TAG_MAIN_HISTORICAL})
    assert len(main_idx) == 3
    assert all(metas[i].agent_role == ROLE_TAG_MAIN_HISTORICAL for i in main_idx)
    le_idx = _snapshot_indices_by_role(metas, {ROLE_TAG_LE_HISTORICAL})
    assert len(le_idx) == 2
    # Bots never returned regardless of role filter.
    assert _snapshot_indices_by_role(metas, {"bot"}) == []


def test_forgotten_main_historical_returns_older_half():
    metas, _ = _make_pool(n_main_historical=10)
    all_main = _snapshot_indices_by_role(metas, {ROLE_TAG_MAIN_HISTORICAL})
    forgotten = _forgotten_main_historical(metas, all_main, fraction=0.5)
    assert len(forgotten) == 5
    # All forgotten env_steps strictly less than any non-forgotten.
    forgotten_max = max(metas[i].env_steps or 0 for i in forgotten)
    nonforgotten = set(all_main) - set(forgotten)
    nonforgotten_min = min(metas[i].env_steps or 0 for i in nonforgotten)
    assert forgotten_max < nonforgotten_min


def test_pfsp_sample_indices_returns_none_on_empty_pool():
    scores = torch.zeros((0,), dtype=torch.float32)
    assert (
        _pfsp_sample_indices(
            [],
            scores,
            n_samples=10,
            mode="hard",
            temperature=1.0,
            epsilon=0.0,
            q=2.0,
            device=torch.device("cpu"),
        )
        is None
    )


def test_pfsp_sample_indices_only_returns_indices_in_pool():
    metas, scores = _make_pool(n_main_historical=5, include_bots=3)
    pool = _snapshot_indices_by_role(metas, {ROLE_TAG_MAIN_HISTORICAL})
    picks = _pfsp_sample_indices(
        pool,
        scores,
        n_samples=200,
        mode="hard",
        temperature=1.0,
        epsilon=1e-3,
        q=2.0,
        device=torch.device("cpu"),
    )
    assert picks is not None
    assert picks.shape == (200,)
    assert set(picks.tolist()).issubset(set(pool))


def test_sample_opponents_rejects_unknown_role():
    triad = _make_triad()
    metas, scores = _make_pool()
    with pytest.raises(KeyError):
        triad.sample_opponents("not_a_role", 10, opponent_metas=metas, opponent_scores=scores)


def test_sample_opponents_zero_slice_returns_empty():
    triad = _make_triad()
    metas, scores = _make_pool()
    out = triad.sample_opponents(ROLE_MAIN, 0, opponent_metas=metas, opponent_scores=scores)
    assert out["opp_kind"].shape == (0,)
    assert out["opp_id"].shape == (0,)


def test_sample_opponents_main_mix_distribution():
    """M's bucket draw matches cfg fractions within 3% over 30k samples."""
    torch.manual_seed(0)
    cfg = _base_cfg(
        triad_main_mix_self=0.35,
        triad_main_mix_pfsp=0.50,
        triad_main_mix_forgotten=0.15,
    )
    triad = _make_triad(cfg)
    # Pool with enough main_historical so both pfsp + forgotten buckets
    # produce real snapshots (not the fall-back-to-self path).
    metas, scores = _make_pool(n_main_historical=20, include_current_main=True)
    n = 30_000
    out = triad.sample_opponents(ROLE_MAIN, n, opponent_metas=metas, opponent_scores=scores)
    kinds = out["opp_kind"]
    p_self = float(kinds.eq(OPP_SELF).float().mean())
    p_snap = float(kinds.eq(OPP_SNAPSHOT).float().mean())
    # OPP_SNAPSHOT covers both whole-league and forgotten buckets.
    assert abs(p_self - 0.35) < 0.03, f"self={p_self}"
    assert abs(p_snap - 0.65) < 0.03, f"snap={p_snap}"


def test_sample_opponents_main_empty_pool_all_selfplay():
    """No snapshots in pool -> M falls back to self-play on every env."""
    triad = _make_triad()
    metas, scores = _make_pool()  # empty
    out = triad.sample_opponents(ROLE_MAIN, 1024, opponent_metas=metas, opponent_scores=scores)
    assert torch.all(out["opp_kind"] == OPP_SELF)


def test_sample_opponents_me_targets_live_main_above_threshold():
    """When ME's eval WR vs current main >= triad_me_curriculum_threshold,
    all envs go OPP_LIVE_MAIN regardless of historical pool contents.
    """
    cfg = _base_cfg(triad_me_curriculum_threshold=0.30)
    triad = _make_triad(cfg)
    triad.role(ROLE_ME).eval_wr_vs_targets["main_current"] = 0.42
    metas, scores = _make_pool(n_main_historical=5, include_current_main=True)
    out = triad.sample_opponents(ROLE_ME, 500, opponent_metas=metas, opponent_scores=scores)
    assert torch.all(out["opp_kind"] == OPP_LIVE_MAIN)


def test_sample_opponents_me_curriculum_fallback_below_threshold():
    """WR < threshold -> ME draws from main_historical via PFSP-var."""
    cfg = _base_cfg(triad_me_curriculum_threshold=0.30)
    triad = _make_triad(cfg)
    triad.role(ROLE_ME).eval_wr_vs_targets["main_current"] = 0.10
    metas, scores = _make_pool(n_main_historical=5)
    main_hist_idx = set(_snapshot_indices_by_role(metas, {ROLE_TAG_MAIN_HISTORICAL}))
    out = triad.sample_opponents(ROLE_ME, 500, opponent_metas=metas, opponent_scores=scores)
    assert torch.all(out["opp_kind"] == OPP_SNAPSHOT)
    assert set(out["opp_id"].tolist()).issubset(main_hist_idx)


def test_sample_opponents_me_no_historicals_degenerates_to_live_main():
    """WR below threshold + empty historical pool -> fallback to live Main
    so ME still has something to train against.
    """
    cfg = _base_cfg(triad_me_curriculum_threshold=0.99)  # force curriculum branch
    triad = _make_triad(cfg)
    metas, scores = _make_pool(include_current_main=True)  # no historicals
    out = triad.sample_opponents(ROLE_ME, 200, opponent_metas=metas, opponent_scores=scores)
    assert torch.all(out["opp_kind"] == OPP_LIVE_MAIN)


def test_sample_opponents_le_uses_full_league_excluding_le_historicals():
    """LE's pool excludes league_exploiter historicals (no self-training)."""
    triad = _make_triad()
    metas, scores = _make_pool(
        n_main_historical=4,
        n_le_historical=3,
        n_me_historical=2,
        include_current_main=True,
    )
    le_hist_idx = set(_snapshot_indices_by_role(metas, {ROLE_TAG_LE_HISTORICAL}))
    out = triad.sample_opponents(ROLE_LE, 500, opponent_metas=metas, opponent_scores=scores)
    assert torch.all(out["opp_kind"] == OPP_SNAPSHOT)
    assert not le_hist_idx.intersection(out["opp_id"].tolist()), (
        "LE should never sample its own historicals"
    )
    # Pool size: 1 (current main) + 4 (main_hist) + 2 (me_hist) = 7.
    expected_pool = set(
        _snapshot_indices_by_role(
            metas, {ROLE_TAG_MAIN_CURRENT, ROLE_TAG_MAIN_HISTORICAL, ROLE_TAG_ME_HISTORICAL}
        )
    )
    assert set(out["opp_id"].tolist()).issubset(expected_pool)


def test_sample_opponents_le_empty_pool_all_selfplay():
    triad = _make_triad()
    metas, scores = _make_pool(n_le_historical=3)  # only LE historicals
    out = triad.sample_opponents(ROLE_LE, 200, opponent_metas=metas, opponent_scores=scores)
    assert torch.all(out["opp_kind"] == OPP_SELF)


def test_sample_opponents_output_device_and_dtype():
    triad = _make_triad()
    metas, scores = _make_pool(n_main_historical=3, include_current_main=True)
    out = triad.sample_opponents(ROLE_MAIN, 64, opponent_metas=metas, opponent_scores=scores)
    assert out["opp_kind"].dtype == torch.int64
    assert out["opp_id"].dtype == torch.int64
    assert out["opp_kind"].device == triad.device
    assert out["opp_id"].device == triad.device


# --------------------------------------------------------------------------- #
# T4: ppo_update with gradient isolation
# --------------------------------------------------------------------------- #


def _fill_role_buffer_with_synthetic_data(triad: TriadController, role: str) -> None:
    """Populate role.buf with plausible PPO inputs.

    Cards/scalars/action_mask/action_type/raise_bucket/old_logprob/value
    are randomized so the PPO loss is non-trivial; h/c are left at zero
    (a valid initial transformer packed state -- length=0 token table).
    """
    rs = triad.role(role)
    buf = rs.buf
    g = torch.Generator(device="cpu").manual_seed(hash(role) & 0xFFFFFFFF)
    buf.cards.copy_(torch.randint(0, 52, buf.cards.shape, generator=g, dtype=torch.int32))
    buf.scalars.copy_(torch.randn(buf.scalars.shape, generator=g, dtype=torch.float32))
    buf.action_mask.fill_(True)
    buf.action_type.copy_(
        torch.randint(0, 4, buf.action_type.shape, generator=g, dtype=torch.int64)
    )
    buf.raise_bucket.copy_(
        torch.randint(0, 7, buf.raise_bucket.shape, generator=g, dtype=torch.int64)
    )
    buf.logprob.copy_(0.1 * torch.randn(buf.logprob.shape, generator=g, dtype=torch.float32))
    buf.value.copy_(0.1 * torch.randn(buf.value.shape, generator=g, dtype=torch.float32))
    # h/c are torch.empty() from RolloutBuffer.__init__ -- uninitialized memory
    # could decode into out-of-vocab embedding indices once unpacked. Zero them
    # to a valid "empty packed transformer state" (all PAD tokens, length=0).
    buf.h.zero_()
    buf.c.zero_()


def _snapshot_all_params(triad: TriadController) -> dict[str, dict[str, torch.Tensor]]:
    return {
        role: {k: v.detach().clone() for k, v in triad.role(role).net.state_dict().items()}
        for role in ROLES
    }


def _params_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def test_ppo_update_role_changes_only_target_net_params():
    """Running PPO on ME updates ME's params but leaves M and LE byte-identical."""
    cfg = _base_cfg(num_envs=512, rollout_steps=4, minibatch_size=32, num_epochs=1, lr=1e-2)
    triad = _make_triad(cfg)
    _fill_role_buffer_with_synthetic_data(triad, ROLE_ME)
    before = _snapshot_all_params(triad)

    rs_me = triad.role(ROLE_ME)
    returns = torch.randn((cfg.rollout_steps, rs_me.slice_n), dtype=torch.float32)
    advantages = torch.randn((cfg.rollout_steps, rs_me.slice_n), dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

    metrics = triad.ppo_update_role(ROLE_ME, returns=returns, advantages=advantages)
    after = _snapshot_all_params(triad)

    assert not _params_equal(before[ROLE_ME], after[ROLE_ME]), "ME params unchanged"
    assert _params_equal(before[ROLE_MAIN], after[ROLE_MAIN]), "M params leaked"
    assert _params_equal(before[ROLE_LE], after[ROLE_LE]), "LE params leaked"

    assert metrics["mb_count"] > 0
    for k in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
        assert k in metrics
        assert torch.isfinite(torch.tensor(metrics[k])).item()


def test_ppo_update_main_does_not_touch_me_or_le():
    """Symmetric isolation check for the M role."""
    cfg = _base_cfg(num_envs=512, rollout_steps=4, minibatch_size=32, num_epochs=1, lr=1e-2)
    triad = _make_triad(cfg)
    _fill_role_buffer_with_synthetic_data(triad, ROLE_MAIN)
    before = _snapshot_all_params(triad)

    rs = triad.role(ROLE_MAIN)
    returns = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
    advantages = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    triad.ppo_update_role(ROLE_MAIN, returns=returns, advantages=advantages)
    after = _snapshot_all_params(triad)

    assert not _params_equal(before[ROLE_MAIN], after[ROLE_MAIN])
    assert _params_equal(before[ROLE_ME], after[ROLE_ME])
    assert _params_equal(before[ROLE_LE], after[ROLE_LE])


def test_ppo_update_role_respects_train_mask():
    """All-False train_mask collapses policy/value losses to 0 (entropy term
    still fires); loss must remain finite.
    """
    cfg = _base_cfg(num_envs=512, rollout_steps=4, minibatch_size=32, num_epochs=1, lr=1e-3)
    triad = _make_triad(cfg)
    _fill_role_buffer_with_synthetic_data(triad, ROLE_LE)
    rs = triad.role(ROLE_LE)
    returns = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
    advantages = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    mask = torch.zeros((cfg.rollout_steps, rs.slice_n), dtype=torch.bool)
    metrics = triad.ppo_update_role(
        ROLE_LE, returns=returns, advantages=advantages, train_mask=mask
    )
    assert metrics["policy_loss"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["value_loss"] == pytest.approx(0.0, abs=1e-6)


def test_ppo_update_role_rejects_wrong_shape():
    cfg = _base_cfg(num_envs=512, rollout_steps=4, minibatch_size=32, num_epochs=1)
    triad = _make_triad(cfg)
    _fill_role_buffer_with_synthetic_data(triad, ROLE_MAIN)
    rs = triad.role(ROLE_MAIN)
    wrong_t = torch.randn((cfg.rollout_steps + 1, rs.slice_n), dtype=torch.float32)
    adv = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
    with pytest.raises(ValueError, match="returns shape"):
        triad.ppo_update_role(ROLE_MAIN, returns=wrong_t, advantages=adv)


def test_ppo_update_role_uses_per_role_entropy_coef():
    """Different entropy_coef -> different loss for an otherwise-identical run.

    Uses lr=0 so the net stays put across the two runs -- only the entropy
    coefficient affects the reported total loss.
    """

    def _run_with_entropy(coef: float) -> float:
        torch.manual_seed(0)
        cfg = _base_cfg(
            num_envs=512,
            rollout_steps=4,
            minibatch_size=32,
            num_epochs=1,
            lr=0.0,
            triad_entropy_me=coef,
        )
        triad = _make_triad(cfg)
        _fill_role_buffer_with_synthetic_data(triad, ROLE_ME)
        rs = triad.role(ROLE_ME)
        returns = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
        adv = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
        m = triad.ppo_update_role(ROLE_ME, returns=returns, advantages=adv)
        return m["loss"]

    loss_low = _run_with_entropy(0.0)
    loss_high = _run_with_entropy(1.0)
    # Higher entropy coef -> larger -entropy_coef*entropy subtraction -> smaller loss.
    assert loss_high < loss_low, f"expected loss_high<loss_low, got {loss_high} vs {loss_low}"


def test_ppo_update_runs_all_three_roles():
    """The convenience wrapper updates every role in one call."""
    cfg = _base_cfg(num_envs=512, rollout_steps=4, minibatch_size=32, num_epochs=1, lr=1e-2)
    triad = _make_triad(cfg)
    for r in ROLES:
        _fill_role_buffer_with_synthetic_data(triad, r)
    before = _snapshot_all_params(triad)

    returns_by_role = {}
    adv_by_role = {}
    for r in ROLES:
        rs = triad.role(r)
        ret = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
        adv = torch.randn((cfg.rollout_steps, rs.slice_n), dtype=torch.float32)
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
        returns_by_role[r] = ret
        adv_by_role[r] = adv

    metrics = triad.ppo_update(
        returns_by_role=returns_by_role,
        advantages_by_role=adv_by_role,
    )
    after = _snapshot_all_params(triad)
    for r in ROLES:
        assert not _params_equal(before[r], after[r]), f"{r} params unchanged"
        assert r in metrics
        assert metrics[r]["mb_count"] > 0


# --------------------------------------------------------------------------- #
# T5: maybe_snapshot + mutate_after_snapshot
# --------------------------------------------------------------------------- #


def test_maybe_snapshot_main_periodic_fires_at_multiples_of_k():
    cfg = _base_cfg(triad_main_snapshot_every=5)
    triad = _make_triad(cfg)
    fired_at = [
        u
        for u in range(0, 21)
        if triad.maybe_snapshot(u, env_steps=u * 1000)[ROLE_MAIN].should_snapshot
    ]
    # update=0 does NOT fire (we require update > 0 to avoid snapshotting random init).
    assert fired_at == [5, 10, 15, 20]


def test_maybe_snapshot_main_tags_as_main_historical():
    cfg = _base_cfg(triad_main_snapshot_every=2)
    triad = _make_triad(cfg)
    dec = triad.maybe_snapshot(4, env_steps=4000)[ROLE_MAIN]
    assert dec.should_snapshot is True
    assert dec.agent_role_tag == "main_historical"
    assert dec.reason == "periodic"


def test_maybe_snapshot_me_below_threshold_does_not_fire():
    cfg = _base_cfg(triad_me_promote_wr=0.70)
    triad = _make_triad(cfg)
    triad.role(ROLE_ME).eval_wr_vs_targets["main_current"] = 0.50
    dec = triad.maybe_snapshot(10, env_steps=10000)[ROLE_ME]
    assert dec.should_snapshot is False
    assert dec.reason == "below_wr_threshold"


def test_maybe_snapshot_me_at_or_above_threshold_fires():
    cfg = _base_cfg(triad_me_promote_wr=0.70)
    triad = _make_triad(cfg)
    triad.role(ROLE_ME).eval_wr_vs_targets["main_current"] = 0.70
    dec = triad.maybe_snapshot(10, env_steps=10000)[ROLE_ME]
    assert dec.should_snapshot is True
    assert dec.agent_role_tag == "main_exploiter"
    assert dec.reason == "wr_threshold"


def test_maybe_snapshot_me_no_eval_data_defers():
    triad = _make_triad()
    # No eval populated yet.
    dec = triad.maybe_snapshot(10, env_steps=10000)[ROLE_ME]
    assert dec.should_snapshot is False
    assert dec.reason == "no_eval_data"


def test_maybe_snapshot_le_uses_min_wr_across_non_le_targets():
    cfg = _base_cfg(triad_le_promote_wr=0.70)
    triad = _make_triad(cfg)
    # One target below threshold blocks the snapshot.
    triad.role(ROLE_LE).eval_wr_vs_targets.update(
        {
            "main_current": 0.80,
            "main_historical_0": 0.65,  # below
            "main_historical_1": 0.75,
        }
    )
    dec = triad.maybe_snapshot(10, env_steps=10000)[ROLE_LE]
    assert dec.should_snapshot is False
    assert dec.reason == "below_wr_threshold"


def test_maybe_snapshot_le_ignores_le_historical_keys():
    cfg = _base_cfg(triad_le_promote_wr=0.70)
    triad = _make_triad(cfg)
    triad.role(ROLE_LE).eval_wr_vs_targets.update(
        {
            "main_current": 0.80,
            "main_historical_0": 0.75,
            # An LE-self eval should be ignored entirely (LE doesn't need to
            # beat itself to "graduate").
            "le_historical_0": 0.10,
        }
    )
    dec = triad.maybe_snapshot(10, env_steps=10000)[ROLE_LE]
    assert dec.should_snapshot is True
    assert dec.agent_role_tag == "league_exploiter"


def test_maybe_snapshot_le_no_eval_data_defers():
    triad = _make_triad()
    dec = triad.maybe_snapshot(10, env_steps=10000)[ROLE_LE]
    assert dec.should_snapshot is False
    assert dec.reason == "no_eval_data"


def test_mutate_after_snapshot_main_is_noop():
    """M never mutates -- its params must be byte-identical after."""
    triad = _make_triad()
    before = {k: v.clone() for k, v in triad.role(ROLE_MAIN).net.state_dict().items()}
    applied = triad.mutate_after_snapshot(ROLE_MAIN)
    after = triad.role(ROLE_MAIN).net.state_dict()
    assert applied == "noop"
    for k in before:
        assert torch.equal(before[k], after[k]), f"M param {k} drifted"
    assert triad.role(ROLE_MAIN).snapshot_count == 1


def test_mutate_after_snapshot_me_copies_main_params():
    triad = _make_triad()
    # Force divergence between M and ME so the test isn't a no-op tautology.
    with torch.no_grad():
        for p in triad.role(ROLE_ME).net.parameters():
            p.add_(0.5)
    assert not _params_equal(
        triad.role(ROLE_MAIN).net.state_dict(),
        triad.role(ROLE_ME).net.state_dict(),
    )
    applied = triad.mutate_after_snapshot(ROLE_ME)
    assert applied == "reset_to_main"
    # ME now byte-identical to M.
    assert _params_equal(
        triad.role(ROLE_MAIN).net.state_dict(),
        triad.role(ROLE_ME).net.state_dict(),
    )


def test_mutate_after_snapshot_le_random_with_prob_zero_is_noop():
    cfg = _base_cfg(triad_le_reset_prob=0.0)
    triad = _make_triad(cfg)
    before = {k: v.clone() for k, v in triad.role(ROLE_LE).net.state_dict().items()}
    for _ in range(10):
        applied = triad.mutate_after_snapshot(ROLE_LE)
        assert applied == "noop"
    after = triad.role(ROLE_LE).net.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k])


def test_mutate_after_snapshot_le_random_with_prob_one_always_reinits():
    cfg = _base_cfg(triad_le_reset_prob=1.0)
    triad = _make_triad(cfg)
    before = {k: v.clone() for k, v in triad.role(ROLE_LE).net.state_dict().items()}
    applied = triad.mutate_after_snapshot(ROLE_LE)
    after = triad.role(ROLE_LE).net.state_dict()
    assert applied == "random_reinit"
    # Net has changed (fresh random init -> not byte-equal to prior state).
    assert not all(torch.equal(before[k], after[k]) for k in before)


def test_mutate_after_snapshot_le_random_with_prob_half_distribution():
    """Over 500 calls with prob=0.5, ~half should reset."""
    cfg = _base_cfg(triad_le_reset_prob=0.5)
    triad = _make_triad(cfg)
    torch.manual_seed(0)
    n = 500
    resets = sum(triad.mutate_after_snapshot(ROLE_LE) == "random_reinit" for _ in range(n))
    frac = resets / n
    assert 0.40 < frac < 0.60, f"reset fraction {frac:.3f} not near 0.5"


def test_mutate_after_snapshot_increments_snapshot_count():
    triad = _make_triad()
    assert triad.role(ROLE_MAIN).snapshot_count == 0
    for i in range(1, 4):
        triad.mutate_after_snapshot(ROLE_MAIN)
        assert triad.role(ROLE_MAIN).snapshot_count == i


# --------------------------------------------------------------------------- #
# maybe_force_reset_me: AlphaStar §3 stalling-fix for ME
# --------------------------------------------------------------------------- #


def test_maybe_force_reset_me_disabled_returns_none():
    """Default cfg.triad_me_force_reset_every=0 => feature is a no-op."""
    triad = _make_triad()  # default _base_cfg has the field at 0
    # Make ME diverge so we'd notice if a reset accidentally fired.
    with torch.no_grad():
        for p in triad.role(ROLE_ME).net.parameters():
            p.add_(0.5)
    me_before = {k: v.clone() for k, v in triad.role(ROLE_ME).net.state_dict().items()}
    # Even at absurdly large updates, disabled means None.
    assert triad.maybe_force_reset_me(update=10_000) is None
    me_after = triad.role(ROLE_ME).net.state_dict()
    for k in me_before:
        assert torch.equal(me_before[k], me_after[k]), f"ME param {k} drifted while disabled"


def test_maybe_force_reset_me_below_threshold_returns_none():
    """Within the timeout window: no reset."""
    cfg = _base_cfg(triad_me_force_reset_every=50)
    triad = _make_triad(cfg)
    # last_snapshot_update starts at -1, so updates_since = update + 1.
    # update=48 => since=49 < 50 => no fire.
    assert triad.maybe_force_reset_me(update=48) is None


def test_maybe_force_reset_me_fires_and_copies_main_params():
    cfg = _base_cfg(triad_me_force_reset_every=50)
    triad = _make_triad(cfg)
    # Force divergence between M and ME so this isn't a no-op tautology.
    with torch.no_grad():
        for p in triad.role(ROLE_ME).net.parameters():
            p.add_(0.5)
    assert not _params_equal(
        triad.role(ROLE_MAIN).net.state_dict(),
        triad.role(ROLE_ME).net.state_dict(),
    )
    # last_snapshot_update=-1, so update=49 -> since=50 -> fires.
    result = triad.maybe_force_reset_me(update=49)
    assert result is not None
    assert result["kind"] == "reset_to_main_forced"
    assert result["updates_since"] == 50
    # ME byte-identical to Main after the reset.
    assert _params_equal(
        triad.role(ROLE_MAIN).net.state_dict(),
        triad.role(ROLE_ME).net.state_dict(),
    )
    # Timer reset: next call within the window doesn't refire.
    assert triad.maybe_force_reset_me(update=50) is None
    assert triad.maybe_force_reset_me(update=98) is None
    # ...but at update=99 (since=50 again), it fires.
    assert triad.maybe_force_reset_me(update=99) is not None


def test_maybe_force_reset_me_does_not_touch_main_or_le():
    """Force-reset only writes to ME's net; M and LE must stay unchanged."""
    cfg = _base_cfg(triad_me_force_reset_every=50)
    triad = _make_triad(cfg)
    main_before = {k: v.clone() for k, v in triad.role(ROLE_MAIN).net.state_dict().items()}
    le_before = {k: v.clone() for k, v in triad.role(ROLE_LE).net.state_dict().items()}
    triad.maybe_force_reset_me(update=100)
    main_after = triad.role(ROLE_MAIN).net.state_dict()
    le_after = triad.role(ROLE_LE).net.state_dict()
    for k in main_before:
        assert torch.equal(main_before[k], main_after[k]), f"Main param {k} drifted"
    for k in le_before:
        assert torch.equal(le_before[k], le_after[k]), f"LE param {k} drifted"


def test_maybe_force_reset_me_does_not_bump_snapshot_count():
    """A forced reset is NOT a snapshot. snapshot_count must not change."""
    cfg = _base_cfg(triad_me_force_reset_every=50)
    triad = _make_triad(cfg)
    assert triad.role(ROLE_ME).snapshot_count == 0
    triad.maybe_force_reset_me(update=100)
    assert triad.role(ROLE_ME).snapshot_count == 0
