# Agent Guide (pokergpu)

This repository's primary goal is a **gym-like poker environment that runs almost entirely on GPU** (Warp + CUDA) to maximize simulation throughput for RL.

## Maintainer notes

Infrastructure specifics — the MLflow server address, the remote GPU
box, deploy targets — and the internal research journal (`wiki/`) are
kept out of version control. Maintainers keep them in a gitignored
`.PRIVATE_AGENTS.md`; this public guide stays generic.

## Ground Rules

- **Prefer GPU-only execution**: avoid CPU round-trips in hot paths (no `.cpu()`, `.numpy()`, or host synchronizations unless explicitly needed for benchmarking or tests).
- **Keep correctness guardrails strong**:
  - chip conservation
  - uncalled bet handling
  - split pots
  - legal action masks/bounds consistent with action application
  - reliable episode boundaries via `episode_id` + `terminated` (auto-reset may still occur)
- **Measure performance continuously**: any meaningful change to env stepping, observations, legal actions, or sampling should be validated with benchmarks.

## Current Interfaces (keep in mind)

- Per-env blinds/stack are supported via `WarpPokerEnv.reset_with_config(...)` and are used for auto-reset as well.
- `obs["scalars"]` currently has shape `(N, 17)`:
  - indices `0..14`: core + public history features
  - index `15`: `big_blind / starting_stack`
  - index `16`: `effective_stack_bb / 200` (clipped to `[0,1]`)

## Reference: prior prototype

There is a prior prototype (a Rust env + Python training stack) that is useful as *inspiration* but should not be copied blindly.

Key takeaways worth preserving in this repo:

- **Per-player RNN state**: maintain separate `(h, c)` per environment *and* per player, and select the correct state based on the acting player each step. This avoids mixing “SB memory” with “BB memory” and is critical for multi-step betting sequences.
- **Store RNN state in the rollout buffer**: snapshot `(h, c)` at action time and store in the rollout buffer so PPO updates can evaluate actions without replaying the whole LSTM over the rollout (big speed win, memory tradeoff).
- **NFSP-lite snapshot opponents (optional early stabilizer)**: periodically save the current policy to a small pool and sample opponents from the pool with probability `(1-eta)` to reduce nonstationarity vs pure self-play.
- **Evaluation bots**: simple scripted opponents (calling station, random aggressive, etc.) are extremely useful for regression tests and fast eval loops.

Important lessons from the prototype:

- Getting PPO to reliably beat even simple opponents requires careful **training/eval protocol** (rollouts, advantage estimation, action masking correctness, reward scaling) and can fail silently without good evals.

## MLflow Experiment Tracking

All `_run_impl_triad` runs push params/metrics/artifacts to a self-hosted
MLflow server. Browser users log in via GitHub OAuth; programmatic
clients (training code, MCP, mlflow CLI) authenticate via HTTP basic
auth against an htpasswd entry.

### Credentials

Credentials live **only** in the gitignored repo-root `.env` — nothing
sensitive is committed. `.env.example` is the committed template; copy
it and fill in real values:

```bash
cp .env.example .env   # then edit .env
```

`.env` keys the code reads:

```
MLFLOW_TRACKING_URI=https://<your-mlflow-host>
MLFLOW_TRACKING_USERNAME=api
MLFLOW_TRACKING_PASSWORD=<token>
```

`train/mlflow_config.py` loads `.env` into the process environment on
import (via `python-dotenv`); variables already set in the environment
override `.env`. `configure_mlflow()` raises `MLflowNotConfiguredError` if
`MLFLOW_TRACKING_URI` is unset. The `api:<token>` pair is an entry in
`oauth2-proxy/htpasswd` on the hosting server — rotate by regenerating
the entry server-side and updating `.env`.

### How training pushes data

`_run_impl_triad` calls into `train/mlflow_logging.py`. The integration
is **on by default** (`cfg.mlflow_enabled: true`) and **fault-tolerant**:
any HTTP/network failure logs a single warning and latches the run into
`disabled_after_error` — training never crashes because MLflow is down.

Disable per-run with `mlflow_enabled=false` (smoke tests, offline work).

Hook points:

- `mlf.start_run(cfg)` once at training start. Logs all TrainConfig
  fields as params, sets tags (git provenance, hardware, hostname,
  Neptune-style labels), uploads the resolved config as a JSON artifact.
- `mlf.log_metrics(run, dict, step=env_steps)` every `do_log` tick —
  ONE batched HTTP call carrying ~92 metrics. NEVER call once per
  metric in a hot loop; build the dict first.
- `mlf.log_snapshot_event(run, ...)` on every snapshot trigger.
- `mlf.end_run(run, status="completed")` at clean exit.

### What gets logged per tick

| Group | Keys |
|---|---|
| Global | `train/overall_sps`, `train/update` |
| System | `system/gpu_util_pct`, `system/gpu_mem_pct` (via pynvml) |
| PPO (×3 roles) | `<role>/{loss,policy_loss,value_loss,entropy,approx_kl,approx_kl_abs,clipfrac,mb_count}` |
| Snapshots (×3) | `<role>/snapshot_count`, `<role>/updates_since_last_snapshot` |
| Eval (×3) | `<role>/eval_wr/main_current` (today: sigmoid proxy from rollout reward; T7 will replace with real per-target WRs) |
| Opp mix (×3) | `<role>/opp_kind/{self,bot,snapshot,live_main}_pct` |
| Actions (×3) | `<role>/action/{fold,check,call,raise}_pct` |
| Raise dist (×3) | `<role>/raise_bucket/b0_pct` … `b6_pct` |
| Rewards (×3) | `<role>/terminal_reward_mean`, `<role>/terminal_hands_count` |
| League | `league/pool_size_{total,main_historical,main_exploiter,league_exploiter}` |
| Triad | `triad/me_curriculum_active` (1 when ME falls back to PFSP-var) |

### Tags (Neptune-style)

Three TrainConfig fields:

- `mlflow_tags: ["triad", "v100", "exp42"]` — each becomes a searchable
  `label.<name> = "true"` MLflow tag. Filter in the UI:
  `tags."label.v100" = "true"`.
- `mlflow_notes: "explore higher entropy on ME"` — markdown-rendered
  description on the run page.
- `mlflow_extra_tags: {"hypothesis": "raise more"}` — arbitrary k/v
  tags merged in after the auto-tags.

Auto-tags set on every run: `git_sha`, `git_branch`, `git_dirty`,
`git_commit_msg`, `gpu_name`, `gpu_capability`, `cuda_version`,
`torch_version`, `python_version`, `hostname`, `train_mode`,
`policy_backbone`, `final_status` (set at clean exit).

### Running with custom MLflow metadata

```bash
uv run python train/train.py --config-name=train_triad_v100 \
  mlflow_run_name=ablation-no-forgotten \
  'mlflow_tags=[ablation,no-forgotten,v100]' \
  'mlflow_notes="Testing triad_main_mix_forgotten=0 vs default 0.15"' \
  triad_main_mix_forgotten=0.0 triad_main_mix_pfsp=0.65
```

### Querying from code / scripts

Always go through `train.mlflow_config.configure_mlflow()` so the URI +
credentials are picked up uniformly:

```python
from train.mlflow_config import configure_mlflow
import mlflow

configure_mlflow(experiment="pokergpu-triad")
runs = mlflow.search_runs(
    experiment_names=["pokergpu-triad"],
    filter_string='tags."label.v100" = "true" AND tags.final_status = "completed"',
    order_by=["start_time DESC"],
    max_results=10,
)
print(runs[["run_id", "tags.mlflow.runName", "metrics.main/loss"]])
```

### When you change the metric set

If you add or rename a metric in `_run_impl_triad`, also:

1. Update the table above (else this guide drifts from reality).
2. Note that **MLflow can't backfill old runs** — once a metric exists,
   it exists for all future runs but missing-from-old-runs forever. Add
   liberally; remove rarely.

### Checkpoint uploads (opt-in)

Off by default. `cfg.mlflow_upload_checkpoints` selects the policy:

| Value | What gets uploaded |
|---|---|
| `none` (default) | Nothing. .pt files stay on local disk. |
| `final` | Main net's weights at clean run exit, as `checkpoints/main/final/triad_main_final_envX_updY.pt`. Single ~13MB upload. |
| `periodic` | Above + every `save_every_env_steps` tick under `checkpoints/main/periodic/`. |
| `all` | Above + every triad snapshot (M periodic, ME/LE on WR-promotion) under `checkpoints/snapshots/<role>/`. |

Override per run:

```bash
uv run python train/train.py --config-name=train_triad_v100 \
  mlflow_upload_checkpoints=final
```

Trade-off: at the production transformer size each .pt is ~13MB, so a
`periodic`/`all` 4B-env-step run can push ~3GB into MinIO. Stay on
`final` unless you specifically need intermediate weights or full pool
provenance from MLflow alone. Local-disk .pt files are unaffected by
this knob -- they're always written.

### Don't log to MLflow

- Per-minibatch metrics — only per-update aggregates.
- TensorBoard event files — TB stays separate (single-run drill-down vs
  MLflow's cross-run comparison).
- Optimizer state alone — already bundled inside the checkpoint .pt by
  `save_checkpoint`; no separate upload needed.

### Failure handling rule

`train/mlflow_logging.py` is the only file allowed to call `mlflow.*`
directly. `_run_impl_triad` MUST go through the wrappers there. If you
need a new operation (e.g. `log_artifact`), add a wrapper that catches
broad `Exception` and latches `run.disabled_after_error = True`.
**Telemetry must never crash training.**

## Keep Docs Up To Date

Whenever you change performance-critical code paths or the training loop plumbing, update:
- `docs/performance_and_roadmap.md`
- `docs/session_handoff.md`

## Benchmarks (Required)

Run and record results (JSON is gitignored):
- Env benchmark: `uv run python bench/run.py`
- Training-style smoke: `uv run python train/torchrl_smoke.py`
- PPO training v0: `uv run python train/train.py`
- League training (NFSP-lite opponent sampling): `uv run python train/train.py league_enabled=true`
- Checkpoint resume: `uv run python train/train.py resume_from=checkpoints/<ckpt>.pt`
- Eval harness (policy vs bots): `uv run python -c "from train.eval import eval_vs_bot; from gpu_poker.env import WarpPokerEnv; from gpu_poker.policy import PokerPolicyNet; env=WarpPokerEnv(num_envs=1024, device='cuda:0'); obs=env.reset(); net=PokerPolicyNet(scalar_dim=int(obs['scalars'].shape[1])); print(eval_vs_bot(env=env, policy=net, bot_name='calling_station', steps=2000))"`

If benchmark output schema changes, update the docs accordingly.

## Code Quality

Before finishing a change:
- `ruff check .`
- `ruff format`
- `uv run pytest`
- `uv run ty check`

Note: in some sandboxed setups `uv run ruff format` can fail due to access to `~/.cache/uv`; using `./.venv/bin/ruff format` is acceptable in that case.
