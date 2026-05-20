# Session Handoff (2025-12-12) — League/PFSP

This is an archival snapshot of the current system state after adding league + PFSP.
The canonical “latest” handoff is `docs/session_handoff.md`.

## What Changed Recently (high level)

- Added a disk-backed snapshot pool: `checkpoints/league_pool/index.json` + `snap_*.pt`.
- Added PFSP-style opponent sampling and diagnostics.
- Fixed seat bias by evaluating snapshots with seat swap and averaging from learner perspective.
- Added eval subset selection (always include freshest few).
- Added pruning (recent dense + exponential sparse) with configurable cadence.

## Key Config Knobs (conf/train.yaml)

League:
- `league_enabled`
- `league_eta_selfplay`, `league_bot_prob`
- `league_snapshot_pool`, `league_snapshot_every_updates`

Snapshot eval:
- `league_eval_every_updates`, `league_eval_hands`
- `league_eval_subset`, `league_eval_deterministic`, `league_eval_seat_swap`

PFSP:
- `league_pfsp_temperature`, `league_pfsp_epsilon`

Retention:
- `league_pool_keep_recent`, `league_pool_keep_exponential`
- `league_prune_every_snapshot_adds`

## TensorBoard Signals

Core PPO:
- `train/loss`, `train/value_loss`, `train/entropy`, `train/approx_kl`, `train/clipfrac`, `train/explained_variance`

Eval:
- `eval/<bot>_bb_per_hand`, `eval/<bot>_bb_per_100`, `eval/<bot>_hands`

League/PFSP:
- `league/snapshots_filled`, `league/snapshot_bb_per_hand_{mean,min,max}`
- `league/pfsp_weight_{min,max,entropy,ess}`

## Testing

- `tests/test_league_pool.py` covers snapshot add/update, pruning behavior, PFSP weight peak.
- Trainer smoke tests cover both normal and league modes.
