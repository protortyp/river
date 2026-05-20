# Triad / Training Follow-ups

Captured during the G (AlphaStar triad) implementation push. None of
these block the first V100 triad run; they are quality/observability
improvements to land after the initial run produces telemetry.

## T7 — Proper async eval

The current `_run_impl_triad` populates `eval_wr_vs_targets["main_current"]`
for ME and LE from a sigmoid-mapped rollout reward proxy. Good enough to
gate snapshot triggers, but it conflates "winning vs the assigned mix of
opponents" with "winning vs the actual target" (current Main / non-LE
league). Replace with:

- A background worker (CUDA stream) that periodically (every
  `triad_eval_every_updates`) plays `triad_eval_hands` hands of:
  - ME vs current Main net (writes `me.eval_wr_vs_targets["main_current"]`)
  - LE vs each non-LE snapshot in the pool (writes per-target keys
    under `le.eval_wr_vs_targets["main_current"]`, `"main_historical_<id>"`,
    `"main_exploiter_<id>"`).
- Eval should use `triad_eval_hands` per target and `eval_vs_policy`
  (from `train.eval`) with seat-swap averaging.

Until this lands, ME/LE may snapshot earlier or later than the AlphaStar
recipe specifies because the proxy reward is a noisy signal.

## T8 — TB logging polish

Currently TB writes per-role under `main/`, `me/`, `le/` namespaces but
with simplified metrics:

- Missing: per-role explained_variance, action distribution stats,
  per-snapshot PFSP weight summaries, league pool size, opponent-kind
  breakdown ("what fraction of M's envs played each kind of opponent").
- Missing: cross-role comparison panels (e.g. `compare/snapshot_count`).
- Missing: GPU util / mem under `system/` (already in `_run_impl`; lift
  the same code).

## T9 — Per-role resume

`TriadController.init_from_resume(path)` currently fans a single .pt out
to all three roles. To resume mid-run (where M / ME / LE have diverged),
we need three-file resume:

- Save: `triad_checkpoint(env_steps, update)` writes
  `triad_main_envX_updY.pt`, `triad_me_envX_updY.pt`,
  `triad_le_envX_updY.pt` plus an `index.json` with role/env_steps/update
  metadata.
- Resume: detect a 3-file checkpoint via the index and load per-role
  weights; fall back to single-file fan-out if only one .pt is given.
- Also persist per-role `snapshot_count`, `last_snapshot_update`,
  `parent_main_id`, and `eval_wr_vs_targets` so the snapshot triggers
  fire on the right cadence after resume.

## T11 — Additional integration tests

The CPU smoke test in `tests/test_train_script.py::test_train_script_cpu_triad_smoke`
exercises the full pipeline end-to-end. Missing more focused checks:

- After a snapshot fires, the next rollout step samples that snapshot
  with non-zero probability (verifies pool sync between controller and
  rollout loop).
- ME `eval_wr_vs_targets["main_current"]` increases after several
  updates against a beatable Main (sanity check on the eval proxy).
- M's selfplay fraction in opp_kind histogram matches
  `triad_main_mix_self` within 5% over 100 rollouts.

## Cross-cutting

- **MLflow integration** (separate from triad): instrument
  `_run_impl_triad` to log per-role / per-update metrics. User is
  setting up the MLflow tracking server separately. Defer until the
  server is up.
- **Forgotten subset definition**: currently `_forgotten_main_historical`
  returns the older half of `main_historical` snapshots by env_steps.
  Refine to a per-snapshot "last_sampled_update" tracker if the
  catastrophic-forgetting diagnostic shows the proxy is too coarse.
- **`opt_steps` accounting**: `_run_impl_triad` reports
  `sum(mb_count)` across all roles; `_run_impl` reports the single
  net's opt step count. They are not directly comparable -- consider a
  `TrainResult.per_role_metrics` field instead.
- **Transformer ONNX export + web inference parity** (separate task):
  the web client still uses the LSTM ONNX. Port `train/export_onnx.py`
  to the transformer backbone and add a parity test.
