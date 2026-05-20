# Session Handoff (2025-12-12) — Self-play Snapshot Eval

This snapshot captures the state after adding a **stronger evaluation signal**: periodic evaluation of the current
policy vs the **league snapshot pool** (seat-swapped).

## Why

Bot evals are useful but can miss:
- plateaus that only show up vs non-trivial opponents
- self-play cycling / overfitting to a narrow style

So we add an eval-only “selfplay vs snapshots” signal.

## What was added

- `train/train.py`
  - New Hydra config knobs in `TrainConfig`:
    - `selfplay_eval_enabled`
    - `selfplay_eval_every_updates`
    - `selfplay_eval_hands`
    - `selfplay_eval_subset`
    - `selfplay_eval_deterministic`
    - `selfplay_eval_seat_swap`
  - Periodic evaluation vs a subset of `league_snapshot_pool` (always includes freshest few).
  - Logs to TensorBoard:
    - `selfplay/bb_per_hand_mean`
    - `selfplay/bb_per_hand_min`
    - `selfplay/bb_per_hand_max`
    - `selfplay/vs_latest_bb_per_hand`
    - plus a few counters (`selfplay/eval_opponents`, `selfplay/eval_hands_total`)

- `conf/train.yaml`
  - Default `selfplay_eval_*` values added (enabled by default, but requires snapshots to exist).

- `train/query_training.py`
  - Prints a “selfplay/snapshot” section if the new tags exist.

## Notes / gotchas

- Self-play snapshot eval requires:
  - `league_enabled=true`
  - `league_snapshot_pool > 0`
  - and snapshots being created (`league_snapshot_every_updates > 0`)
- Seat swap is enabled by default to reduce HU SB/BB bias.

## Commands

- Train: `uv run python train/train.py`
- Query: `uv run python train/query_training.py`
