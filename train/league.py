from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SnapshotMeta:
    path: str
    env_steps: int
    update: int
    # Estimated performance of the *current learner* vs this snapshot (from the learner's view).
    # Positive means learner wins in expectation.
    bb_per_hand: float | None = None
    hands: int | None = None
    # Optional seat-specific evals for debugging seat bias.
    # - `bb_per_hand_p0`: learner is P0, snapshot is P1 (P0 bb/hand; i.e., learner bb/hand).
    # - `bb_per_hand_p1`: learner is P1, snapshot is P0 (already in learner view).
    #   (I.e., learner bb/hand == -P0 bb/hand.)
    bb_per_hand_p0: float | None = None
    bb_per_hand_p1: float | None = None


class LeaguePool:
    """
    Disk-backed snapshot pool with a small JSON index for metadata.

    This is intentionally lightweight (no Elo/TrueSkill yet). It's enough to:
    - persist snapshots across restarts
    - store per-snapshot eval metrics for PFSP-style sampling
    """

    def __init__(self, *, pool_dir: Path) -> None:
        self.pool_dir = pool_dir
        self.index_path = pool_dir / "index.json"
        self.pool_dir.mkdir(parents=True, exist_ok=True)

    def load_index(self) -> list[SnapshotMeta]:
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text())
        items = data.get("snapshots", [])
        out: list[SnapshotMeta] = []
        for it in items:
            out.append(SnapshotMeta(**it))
        return out

    def save_index(self, snapshots: list[SnapshotMeta]) -> None:
        payload = {"snapshots": [asdict(s) for s in snapshots]}
        self.index_path.write_text(json.dumps(payload, indent=2) + "\n")

    def add_snapshot(
        self,
        *,
        model: torch.nn.Module,
        env_steps: int,
        update: int,
    ) -> SnapshotMeta:
        path = self.pool_dir / f"snap_env{env_steps}_upd{update}.pt"
        torch.save(
            {
                "env_steps": int(env_steps),
                "update": int(update),
                "model": model.state_dict(),
            },
            path,
        )
        meta = SnapshotMeta(path=str(path), env_steps=int(env_steps), update=int(update))
        snaps = self.load_index()
        snaps.append(meta)
        self.save_index(snaps)
        return meta

    def update_eval(
        self,
        *,
        path: str,
        bb_per_hand: float,
        hands: int,
        bb_per_hand_p0: float | None = None,
        bb_per_hand_p1: float | None = None,
    ) -> None:
        snaps = self.load_index()
        changed = False
        for i, s in enumerate(snaps):
            if s.path == path:
                snaps[i] = SnapshotMeta(
                    path=s.path,
                    env_steps=s.env_steps,
                    update=s.update,
                    bb_per_hand=float(bb_per_hand),
                    hands=int(hands),
                    bb_per_hand_p0=float(bb_per_hand_p0) if bb_per_hand_p0 is not None else None,
                    bb_per_hand_p1=float(bb_per_hand_p1) if bb_per_hand_p1 is not None else None,
                )
                changed = True
                break
        if changed:
            self.save_index(snaps)

    def prune(
        self,
        *,
        keep_recent: int,
        keep_exponential: bool,
    ) -> None:
        """
        Prune the disk pool to avoid unbounded growth.

        Retention:
        - Keep the most recent `keep_recent` snapshots (by env_steps/update).
        - Optionally keep a sparse set of older snapshots, approximately exponentially spaced
          by keeping at most 1 snapshot per `floor(log2(env_steps))` bucket.
        """
        snaps = self.load_index()
        if not snaps:
            return

        snaps_sorted = sorted(snaps, key=lambda m: (m.env_steps, m.update))
        keep_recent = max(0, int(keep_recent))
        keep: dict[str, SnapshotMeta] = {}

        recent = snaps_sorted[-keep_recent:] if keep_recent > 0 else []
        for s in recent:
            keep[s.path] = s

        if keep_exponential:
            buckets: dict[int, SnapshotMeta] = {}
            for s in snaps_sorted[: max(0, len(snaps_sorted) - len(recent))]:
                # Bucket by log2(env_steps). env_steps==0 is bucket 0.
                b = int(max(0, int(s.env_steps)).bit_length() - 1)
                prev = buckets.get(b)
                if prev is None or (s.env_steps, s.update) > (prev.env_steps, prev.update):
                    buckets[b] = s
            for s in buckets.values():
                keep[s.path] = s

        # Delete unkept files and write new index.
        kept_paths = set(keep.keys())
        for s in snaps_sorted:
            if s.path in kept_paths:
                continue
            # Best-effort: leave it on disk but drop from index.
            with contextlib.suppress(OSError):
                Path(s.path).unlink(missing_ok=True)

        new_index = sorted(keep.values(), key=lambda m: (m.env_steps, m.update))
        self.save_index(new_index)


@dataclass(frozen=True)
class OpponentMeta:
    """Metadata for any opponent (bot or snapshot) in the unified pool."""

    id: str  # Bot name or snapshot path
    kind: str  # "bot" or "snapshot"
    agent_role: str | None = None  # e.g. "main", "main_historical", "league_exploiter"
    # Snapshot-specific fields
    env_steps: int | None = None
    update: int | None = None
    # PFSP evaluation results (used for both)
    bb_per_hand: float | None = None
    hands: int | None = None
    bb_per_hand_p0: float | None = None
    bb_per_hand_p1: float | None = None


class UnifiedOpponentPool:
    """
    Manages a unified pool of opponents (bots + snapshots) with PFSP weighting.
    Bots are registered at init, snapshots are added dynamically.
    """

    def __init__(self, *, pool_dir: Path, bot_names: list[str]) -> None:
        self.pool_dir = pool_dir
        self.index_path = pool_dir / "unified_index.json"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.bot_names = list(bot_names)

    def load_index(self) -> list[OpponentMeta]:
        """Load index, ensuring bots are always present."""
        if not self.index_path.exists():
            # Initialize with bots only
            opponents = [OpponentMeta(id=name, kind="bot") for name in self.bot_names]
            self.save_index(opponents)
            return opponents

        data = json.loads(self.index_path.read_text())
        opponents = [OpponentMeta(**item) for item in data.get("opponents", [])]

        # Ensure all bots are present (in case new bots were added)
        existing_bot_ids = {o.id for o in opponents if o.kind == "bot"}
        new_bots = []
        for name in self.bot_names:
            if name not in existing_bot_ids:
                new_bots.append(OpponentMeta(id=name, kind="bot"))

        # Prepend new bots so they are available; order is mainly for display.
        if new_bots:
            opponents = new_bots + opponents

        return opponents

    def save_index(self, opponents: list[OpponentMeta]) -> None:
        payload = {"opponents": [asdict(o) for o in opponents]}
        self.index_path.write_text(json.dumps(payload, indent=2) + "\n")

    def add_snapshot(
        self, *, model: torch.nn.Module, env_steps: int, update: int, agent_role: str
    ) -> OpponentMeta:
        """Add a new snapshot to the pool."""
        path = self.pool_dir / f"snap_env{env_steps}_upd{update}.pt"
        torch.save(
            {
                "env_steps": int(env_steps),
                "update": int(update),
                "model": model.state_dict(),
            },
            path,
        )

        meta = OpponentMeta(
            id=str(path),
            kind="snapshot",
            agent_role=agent_role,
            env_steps=int(env_steps),
            update=int(update),
        )
        opponents = self.load_index()
        opponents.append(meta)
        self.save_index(opponents)
        return meta

    def update_eval(
        self,
        *,
        opponent_id: str,
        bb_per_hand: float,
        hands: int,
        bb_per_hand_p0: float | None = None,
        bb_per_hand_p1: float | None = None,
    ) -> None:
        """Update evaluation results for any opponent (bot or snapshot)."""
        opponents = self.load_index()
        changed = False
        for i, o in enumerate(opponents):
            if o.id == opponent_id:
                opponents[i] = OpponentMeta(
                    id=o.id,
                    kind=o.kind,
                    agent_role=o.agent_role,
                    env_steps=o.env_steps,
                    update=o.update,
                    bb_per_hand=float(bb_per_hand),
                    hands=int(hands),
                    bb_per_hand_p0=float(bb_per_hand_p0) if bb_per_hand_p0 is not None else None,
                    bb_per_hand_p1=float(bb_per_hand_p1) if bb_per_hand_p1 is not None else None,
                )
                changed = True
                break
        if changed:
            self.save_index(opponents)

    def update_role(self, *, opponent_id: str, new_role: str) -> None:
        """Update the role of an existing opponent."""
        opponents = self.load_index()
        changed = False
        for i, o in enumerate(opponents):
            if o.id == opponent_id:
                opponents[i] = OpponentMeta(
                    id=o.id,
                    kind=o.kind,
                    agent_role=new_role,
                    env_steps=o.env_steps,
                    update=o.update,
                    bb_per_hand=o.bb_per_hand,
                    hands=o.hands,
                    bb_per_hand_p0=o.bb_per_hand_p0,
                    bb_per_hand_p1=o.bb_per_hand_p1,
                )
                changed = True
                break
        if changed:
            self.save_index(opponents)

    def get_bots(self) -> list[OpponentMeta]:
        return [o for o in self.load_index() if o.kind == "bot"]

    def get_snapshots(self) -> list[OpponentMeta]:
        return [o for o in self.load_index() if o.kind == "snapshot"]

    def prune_snapshots(self, *, keep_recent: int, keep_exponential: bool) -> None:
        """Prune old snapshots (bots are never pruned)."""
        opponents = self.load_index()
        bots = [o for o in opponents if o.kind == "bot"]
        snapshots = [o for o in opponents if o.kind == "snapshot"]

        if not snapshots:
            return

        # Sort snapshots by age (env_steps, update)
        # Some manually added or corrupted snapshots might be missing env_steps.
        # We assume strict typing here as per add_snapshot.
        snapshots_sorted = sorted(snapshots, key=lambda m: (m.env_steps or 0, m.update or 0))

        keep_recent = max(0, int(keep_recent))
        keep: dict[str, OpponentMeta] = {}

        recent = snapshots_sorted[-keep_recent:] if keep_recent > 0 else []
        for s in recent:
            keep[s.id] = s

        if keep_exponential:
            buckets: dict[int, OpponentMeta] = {}
            for s in snapshots_sorted[: max(0, len(snapshots_sorted) - len(recent))]:
                # Bucket by log2(env_steps). env_steps==0 is bucket 0.
                val = max(0, int(s.env_steps or 0))
                b = int(val.bit_length() - 1)
                prev = buckets.get(b)
                # Keep the NEWER snapshot in the bucket
                if prev is None or (s.env_steps or 0, s.update or 0) > (
                    prev.env_steps or 0,
                    prev.update or 0,
                ):
                    buckets[b] = s
            for s in buckets.values():
                keep[s.id] = s

        # Delete unkept files and write new index.
        kept_paths = set(keep.keys())
        for s in snapshots_sorted:
            if s.id in kept_paths:
                continue
            # Best-effort: leave it on disk but drop from index.
            with contextlib.suppress(OSError):
                Path(s.id).unlink(missing_ok=True)

        kept_snapshots = sorted(keep.values(), key=lambda m: (m.env_steps or 0, m.update or 0))

        # Save back bots + kept snapshots
        self.save_index(bots + kept_snapshots)


def pfsp_weights_from_bb_per_hand(
    bb_per_hand: torch.Tensor,
    *,
    temperature: float,
    epsilon: float,
    mode: str = "hard",
    q: float = 2.0,
) -> torch.Tensor:
    """
    Convert a bb/hand estimate to PFSP-style sampling weights.

    Two AlphaStar-style priority functions are supported (Vinyals et al. 2019):

    - mode="hard" (default, main-agent training): f_hard(p) = (1 - p)^q.
      Concentrates mass on opponents the learner *loses to*. This is what
      AlphaStar's main agent and league exploiters use. q=2 by default.

    - mode="var" (curriculum / exploiters): f_var(p) = p*(1 - p).
      Concentrates on opponents at ~50% win prob — useful for an exploiter
      trying to climb against opponents currently near parity. AlphaStar's
      main exploiters fell back to f_var when win_prob < 20%.

    Both apply an additive epsilon floor for exploration / numerical stability.

    The previous default (f_var) was the wrong fit for a main agent: combined
    with NFSP-style training against scripted bots the learner already crushes,
    it weighted everything except the easiest bots and gave a near-uniform
    rollout share over the opponent pool, amplifying the exploiter trap.
    """
    t = max(1e-6, float(temperature))
    p = torch.sigmoid(bb_per_hand / t)
    w = p * (1.0 - p) if mode == "var" else (1.0 - p).clamp_min(0.0).pow(float(q))
    if epsilon > 0:
        w = w + float(epsilon)
    return w


def pfsp_weights_for_exploiter(
    bb_per_hand_exploiter: torch.Tensor, *, temperature: float, epsilon: float
) -> torch.Tensor:
    """
    Monotone PFSP variant for the exploiter.

    Emphasizes opponents the exploiter *currently loses to*: the weight is
    `sigmoid(-bb_exploiter / T)`, which is near 1 when the exploiter is losing
    badly and decays toward 0 when it dominates. This is the right signal for
    a counter-strategy trainer — easy wins generate little gradient.
    """
    t = max(1e-6, float(temperature))
    w = torch.sigmoid(-bb_per_hand_exploiter / t)
    if epsilon > 0:
        w = w + float(epsilon)
    return w
