from __future__ import annotations

from pathlib import Path

import torch
from train.league import UnifiedOpponentPool


def test_unified_pool_init_with_bots(tmp_path: Path) -> None:
    pool_dir = tmp_path / "league_pool"
    bot_names = ["bot_a", "bot_b"]
    pool = UnifiedOpponentPool(pool_dir=pool_dir, bot_names=bot_names)

    # Check if index file is created on load if missing
    opponents = pool.load_index()
    assert len(opponents) == 2
    assert {o.id for o in opponents} == {"bot_a", "bot_b"}
    assert all(o.kind == "bot" for o in opponents)
    assert Path(pool.index_path).exists()


def test_unified_pool_add_snapshot_and_update(tmp_path: Path) -> None:
    pool_dir = tmp_path / "league_pool"
    pool = UnifiedOpponentPool(pool_dir=pool_dir, bot_names=["bot_a"])

    m = torch.nn.Linear(4, 3)
    meta = pool.add_snapshot(model=m, env_steps=100, update=1, agent_role="learner")

    assert Path(meta.id).exists()
    assert meta.kind == "snapshot"
    assert meta.env_steps == 100

    opponents = pool.load_index()
    assert len(opponents) == 2  # 1 bot + 1 snapshot

    pool.update_eval(opponent_id=meta.id, bb_per_hand=0.1, hands=100)
    opponents = pool.load_index()
    snap = next(o for o in opponents if o.id == meta.id)
    assert snap.bb_per_hand == 0.1
    assert snap.hands == 100

    pool.update_eval(opponent_id="bot_a", bb_per_hand=-0.5, hands=200)
    opponents = pool.load_index()
    bot = next(o for o in opponents if o.id == "bot_a")
    assert bot.bb_per_hand == -0.5
    assert bot.hands == 200


def test_unified_pool_prune_snapshots_preserves_bots(tmp_path: Path) -> None:
    pool_dir = tmp_path / "league_pool"
    pool = UnifiedOpponentPool(pool_dir=pool_dir, bot_names=["bot_a", "bot_b"])
    m = torch.nn.Linear(1, 1)

    # Add many snapshots
    for i in range(10):
        pool.add_snapshot(model=m, env_steps=i * 100, update=i, agent_role="learner")

    opponents_before = pool.load_index()
    assert len(opponents_before) == 12  # 2 bots + 10 snapshots

    # Prune keeping only recent 2
    pool.prune_snapshots(keep_recent=2, keep_exponential=False)

    opponents_after = pool.load_index()
    bots = [o for o in opponents_after if o.kind == "bot"]
    snapshots = [o for o in opponents_after if o.kind == "snapshot"]

    assert len(bots) == 2
    assert len(snapshots) == 2
    assert snapshots[0].env_steps == 800
    assert snapshots[1].env_steps == 900


def test_unified_pool_persistence(tmp_path: Path) -> None:
    pool_dir = tmp_path / "league_pool"
    pool = UnifiedOpponentPool(pool_dir=pool_dir, bot_names=["bot_a"])
    pool.load_index()  # Initialize

    # Simulate restart by creating new pool instance on same dir
    pool2 = UnifiedOpponentPool(pool_dir=pool_dir, bot_names=["bot_a", "bot_b"])
    opponents = pool2.load_index()

    # Should contain both old bot_a and new bot_b
    assert len(opponents) == 2
    assert {o.id for o in opponents} == {"bot_a", "bot_b"}


def test_snapshots_from_different_roles_at_same_step_do_not_collide(tmp_path: Path) -> None:
    pool = UnifiedOpponentPool(pool_dir=tmp_path / "pool", bot_names=[])
    model = torch.nn.Linear(2, 2)
    a = pool.add_snapshot(model=model, env_steps=10, update=1, agent_role="main_historical")
    b = pool.add_snapshot(model=model, env_steps=10, update=1, agent_role="main_exploiter")
    assert a.id != b.id
    assert Path(a.id).exists() and Path(b.id).exists()
