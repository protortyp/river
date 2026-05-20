"""MLflow telemetry helpers for ``_run_impl_triad``.

Lives in its own module so train.py stays focused on the training
loop. All public functions accept an ``MLflowRun`` handle (returned by
``start_run``) and degrade to no-ops when MLflow is disabled or the
server is unreachable.

Failure policy (D-pattern: graceful degradation):
  * If ``mlflow_enabled=False`` -> every helper is a no-op (zero work).
  * If a logging call raises -> log a single WARNING and disable
    further MLflow calls for the remainder of this training run.
    Failures of a telemetry sink must never bring down the trainer.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from train.train import TrainConfig

logger = logging.getLogger(__name__)

# Action-type integer constants mirror gpu_poker.constants but are
# duplicated here so this module doesn't need to import the env layer
# just for telemetry. Kept in sync via the action_distribution test.
ACT_FOLD = 0
ACT_CHECK = 1
ACT_CALL = 2
ACT_RAISE = 3
NUM_RAISE_BUCKETS = 7


@dataclass
class MLflowRun:
    """Handle returned by ``start_run``. Carries the active run object
    and the kill-switch flag that ``_safe`` consults.

    Treat as opaque from train.py; only pass it back into the helpers
    in this module.
    """

    # ``mlflow.ActiveRun`` or None when disabled. Stored as Any to avoid
    # forcing an mlflow import when the user runs with mlflow_enabled=False.
    active: Any = None
    enabled: bool = False
    # Set True the first time a logging call raises so subsequent calls
    # bail out fast. Resets only on the next training run.
    disabled_after_error: bool = False


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def start_run(cfg: TrainConfig) -> MLflowRun:
    """Open an MLflow run (or return a no-op handle).

    Configures the tracking URI from ``train.mlflow_config``, logs all
    TrainConfig fields as params, sets git / hardware tags, and uploads
    the resolved config as an artifact. Returns the ``MLflowRun`` handle
    even when disabled so the caller can pass it through unconditionally.
    """
    if not cfg.mlflow_enabled:
        return MLflowRun(enabled=False)
    try:
        from train.mlflow_config import configure_mlflow

        configure_mlflow(experiment=cfg.mlflow_experiment)
        import mlflow

        run_name = cfg.mlflow_run_name or _auto_run_name(cfg)
        active = mlflow.start_run(run_name=run_name)

        # Params: all TrainConfig fields. Values longer than MLflow's
        # 500-char param limit are stringified-and-truncated; collections
        # (lists, dicts) are JSON-encoded.
        from dataclasses import asdict

        cfg_dict = asdict(cfg)
        params = {k: _coerce_param(v) for k, v in cfg_dict.items()}
        # MLflow log_params accepts up to 100 keys per call; chunk if needed.
        for chunk in _chunk_dict(params, 100):
            mlflow.log_params(chunk)

        # Tags: git provenance + hardware + run-shape + user-supplied.
        # Neptune-style flat labels become `label.<name> = "true"` so
        # they're searchable individually in the UI.
        label_tags = {f"label.{t}": "true" for t in (cfg.mlflow_tags or [])}
        # User extras win over auto-tags via dict-merge order.
        extra = dict(cfg.mlflow_extra_tags or {})
        tags = {
            **_git_provenance(),
            **_hardware_tags(cfg),
            "train_mode": "triad" if cfg.triad_enabled else "single_net",
            "policy_backbone": cfg.policy_backbone,
            "hostname": socket.gethostname(),
            **label_tags,
            **extra,
        }
        if cfg.mlflow_notes:
            # mlflow.note.content is the system tag the UI renders as the
            # run description (markdown supported).
            tags["mlflow.note.content"] = cfg.mlflow_notes
        mlflow.set_tags(tags)

        # Artifact: the resolved config as JSON so we can reproduce
        # exactly without depending on hydra's overrides being recorded.
        with tempfile.NamedTemporaryFile("w", suffix="_train_config.json", delete=False) as f:
            json.dump(cfg_dict, f, indent=2, default=str)
            f_path = f.name
        try:
            mlflow.log_artifact(f_path, artifact_path="config")
        finally:
            os.unlink(f_path)

        logger.info(
            "mlflow: run started at %s (experiment=%s, run_name=%s)",
            mlflow.get_tracking_uri(),
            cfg.mlflow_experiment or "default",
            run_name,
        )
        return MLflowRun(active=active, enabled=True)

    except Exception as e:  # noqa: BLE001 - intentional broad catch
        logger.warning(
            "mlflow: failed to start run (%s: %s); disabling for this training run",
            type(e).__name__,
            e,
        )
        return MLflowRun(enabled=False)


def end_run(run: MLflowRun, *, status: str) -> None:
    """Close the MLflow run, tagging a final_status (completed / interrupted / crashed)."""
    if not run.enabled or run.disabled_after_error:
        return
    try:
        import mlflow

        mlflow.set_tag("final_status", status)
        mlflow.end_run()
    except Exception as e:  # noqa: BLE001
        logger.warning("mlflow: end_run failed (%s: %s)", type(e).__name__, e)


# --------------------------------------------------------------------------- #
# Metric / tag emission helpers
# --------------------------------------------------------------------------- #


def log_metrics(run: MLflowRun, metrics: dict[str, float], *, step: int) -> None:
    """Batched metric log. Single HTTP call regardless of dict size."""
    if not run.enabled or run.disabled_after_error or not metrics:
        return
    try:
        import mlflow

        # Filter NaN/Inf -- MLflow rejects them and aborts the batch.
        clean = {k: float(v) for k, v in metrics.items() if v is not None and _is_finite(v)}
        if clean:
            mlflow.log_metrics(clean, step=int(step))
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "mlflow: log_metrics failed (%s: %s); disabling for this run",
            type(e).__name__,
            e,
        )
        run.disabled_after_error = True


def set_tag(run: MLflowRun, key: str, value: str) -> None:
    if not run.enabled or run.disabled_after_error:
        return
    try:
        import mlflow

        mlflow.set_tag(key, value)
    except Exception as e:  # noqa: BLE001
        logger.warning("mlflow: set_tag failed (%s: %s)", type(e).__name__, e)
        run.disabled_after_error = True


def log_checkpoint(
    run: MLflowRun,
    path,
    *,
    artifact_path: str,
) -> None:
    """Upload a single .pt checkpoint file as an MLflow artifact.

    ``artifact_path`` is the in-run namespace (e.g. ``checkpoints/main``
    or ``checkpoints/snapshots``). Wrapped in the same safe-fail policy
    as every other helper here: upload failures degrade the run to
    ``disabled_after_error`` but never raise to the caller.

    NOTE: ``mlflow.log_artifact`` blocks until the upload completes. For
    long-running training this can stall the loop by O(file_size /
    upstream_bandwidth). At ~13MB per role / per call over the typical
    home-server link this is sub-second; if a future change pushes
    multi-GB artifacts (e.g. full optimizer state), revisit and consider
    spawning a background thread.
    """
    if not run.enabled or run.disabled_after_error:
        return
    try:
        import mlflow

        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "mlflow: log_artifact(%s) failed (%s: %s); disabling for this run",
            path,
            type(e).__name__,
            e,
        )
        run.disabled_after_error = True


def log_snapshot_event(
    run: MLflowRun,
    *,
    role: str,
    snapshot_count: int,
    env_steps: int,
    update: int,
    mutate_applied: str,
) -> None:
    """Record a per-role snapshot promotion. Visible as a UI tag plus a
    monotone metric so it shows up on the timeline chart.
    """
    set_tag(
        run,
        f"snapshot/{role}/{snapshot_count}",
        f"env{env_steps}/upd{update}/{mutate_applied}",
    )
    log_metrics(run, {f"{role}/snapshot_count": float(snapshot_count)}, step=env_steps)


# --------------------------------------------------------------------------- #
# Per-tick metric collectors (compose the per-role dict the trainer logs)
# --------------------------------------------------------------------------- #


def opp_kind_distribution(opp_kind_slice: torch.Tensor) -> dict[str, float]:
    """Fraction of envs in a role's slice assigned to each opponent kind.

    Returns ``{opp_kind/self_pct, opp_kind/bot_pct, opp_kind/snapshot_pct,
    opp_kind/live_main_pct}`` (sums to ~1.0). Empty slice -> all zeros.
    """
    n = int(opp_kind_slice.numel())
    if n == 0:
        return {
            "opp_kind/self_pct": 0.0,
            "opp_kind/bot_pct": 0.0,
            "opp_kind/snapshot_pct": 0.0,
            "opp_kind/live_main_pct": 0.0,
        }
    flat = opp_kind_slice.flatten()
    return {
        "opp_kind/self_pct": float(flat.eq(0).float().mean()),
        "opp_kind/bot_pct": float(flat.eq(1).float().mean()),
        "opp_kind/snapshot_pct": float(flat.eq(2).float().mean()),
        "opp_kind/live_main_pct": float(flat.eq(3).float().mean()),
    }


def action_distribution(action_type: torch.Tensor, learner_mask: torch.Tensor) -> dict[str, float]:
    """Action-type fractions averaged over learner-actor steps only.

    Mirrors train.py's _action_stats_masked but returns the four
    {fold/check/call/raise}_pct values keyed for MLflow.
    """
    mask = learner_mask.to(dtype=torch.bool)
    total = float(mask.sum().item())
    if total <= 0:
        return {
            "action/fold_pct": 0.0,
            "action/check_pct": 0.0,
            "action/call_pct": 0.0,
            "action/raise_pct": 0.0,
        }
    at = action_type[mask]
    return {
        "action/fold_pct": float(at.eq(ACT_FOLD).float().mean()),
        "action/check_pct": float(at.eq(ACT_CHECK).float().mean()),
        "action/call_pct": float(at.eq(ACT_CALL).float().mean()),
        "action/raise_pct": float(at.eq(ACT_RAISE).float().mean()),
    }


def raise_bucket_distribution(
    raise_bucket: torch.Tensor,
    action_type: torch.Tensor,
    learner_mask: torch.Tensor,
) -> dict[str, float]:
    """Per-bucket fraction of LEARNER raise actions (b0_pct..b6_pct).

    Each value is conditional on action==RAISE, so they sum to ~1.0
    over the raise subset (not over all actions). Empty -> all zeros.
    """
    mask = learner_mask.to(dtype=torch.bool) & action_type.eq(ACT_RAISE)
    total = float(mask.sum().item())
    out = {f"raise_bucket/b{b}_pct": 0.0 for b in range(NUM_RAISE_BUCKETS)}
    if total <= 0:
        return out
    rb = raise_bucket[mask]
    for b in range(NUM_RAISE_BUCKETS):
        out[f"raise_bucket/b{b}_pct"] = float(rb.eq(b).float().mean())
    return out


def system_stats(device: str) -> dict[str, float]:
    """GPU utilization + memory percentages via pynvml if available."""
    out: dict[str, float] = {}
    try:
        import pynvml  # type: ignore[import-untyped]

        if not device.startswith("cuda"):
            return out
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        out["system/gpu_util_pct"] = float(util.gpu)
        out["system/gpu_mem_pct"] = float(100.0 * mem.used / mem.total)
    except Exception:  # noqa: BLE001
        # pynvml not installed or NVML failed -- silently skip.
        pass
    return out


# --------------------------------------------------------------------------- #
# Internal: provenance + utility
# --------------------------------------------------------------------------- #


def _auto_run_name(cfg: TrainConfig) -> str:
    sha = _git_sha_short() or "nosha"
    ts = time.strftime("%Y%m%d-%H%M%S")
    mode = "triad" if cfg.triad_enabled else "single"
    return f"{mode}-{cfg.policy_backbone}-{ts}-{sha}"


def _git_provenance() -> dict[str, str]:
    """SHA / branch / dirty status. Empty dict if git fails."""
    out: dict[str, str] = {}
    try:
        out["git_sha"] = _run_git(["rev-parse", "HEAD"]) or ""
        out["git_branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or ""
        dirty = _run_git(["status", "--porcelain"])
        out["git_dirty"] = "true" if dirty else "false"
        msg = _run_git(["log", "-1", "--pretty=%s"])
        if msg:
            out["git_commit_msg"] = msg[:250]
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in out.items() if v}


def _git_sha_short() -> str:
    sha = _run_git(["rev-parse", "--short", "HEAD"])
    return sha or ""


def _run_git(args: list[str]) -> str:
    try:
        r = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _hardware_tags(cfg: TrainConfig) -> dict[str, str]:
    tags: dict[str, str] = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        try:
            tags["gpu_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            tags["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
            tags["cuda_version"] = torch.version.cuda or "unknown"
        except Exception:  # noqa: BLE001
            pass
    return tags


def _coerce_param(v: Any) -> str:
    """MLflow params must be strings <=500 chars. Stringify everything,
    JSON-encode lists/dicts/None, truncate aggressively."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        s = "" if v is None else str(v)
    else:
        try:
            s = json.dumps(v, default=str)
        except Exception:  # noqa: BLE001
            s = str(v)
    return s[:500]


def _chunk_dict(d: dict, n: int) -> Iterable[dict]:
    items = list(d.items())
    for i in range(0, len(items), n):
        yield dict(items[i : i + n])


def _is_finite(v: Any) -> bool:
    try:
        f = float(v)
        return not (f != f or f == float("inf") or f == float("-inf"))
    except (TypeError, ValueError):
        return False
