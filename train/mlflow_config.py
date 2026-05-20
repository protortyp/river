"""MLflow tracking server configuration for pokergpu training.

Credentials and the tracking-server address are read **entirely from the
environment** -- nothing sensitive is committed to the repo.

The gitignored repo-root ``.env`` file is loaded into the process
environment on import (via ``python-dotenv``). ``.env.example`` is the
committed template listing the expected keys: copy it to ``.env`` and
fill in real values, or provide the variables some other way (CI,
container env, shell profile).

Resolution order (highest priority first):
  1. Variables already present in the process environment -- so CI /
     ad-hoc runs can override without touching ``.env``.
  2. The repo-root ``.env`` file.

The MLflow client reads ``MLFLOW_TRACKING_URI`` /
``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD`` from the
environment at request time; once ``.env`` is loaded those are in place.
``configure_mlflow()`` additionally points the client at the URI and
selects the experiment.

To rotate the API token: regenerate the htpasswd entry on the hosting
server (see hosting-setup/README) and update ``.env``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load the gitignored repo-root .env into os.environ. load_dotenv does
# NOT override variables already set in the process environment, so
# explicit env / CI overrides win (see resolution order above). It is a
# safe no-op when .env is absent.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# Default experiment name -- not sensitive, safe to keep in source.
# Per-run overrides go through the ``mlflow.set_experiment`` call site
# in _run_impl_triad.
DEFAULT_EXPERIMENT: str = "pokergpu-triad"


class MLflowNotConfiguredError(RuntimeError):
    """Raised when no MLflow tracking URI is configured.

    ``train.mlflow_logging.start_run`` catches this and degrades to a
    no-op, so training never depends on telemetry being configured.
    """


def tracking_uri() -> str:
    """Return the configured MLflow tracking URI.

    Reads ``MLFLOW_TRACKING_URI`` from the environment (populated from
    ``.env`` on import). Raises :class:`MLflowNotConfiguredError` if it
    is unset or empty. Makes no network call and does not import
    ``mlflow`` -- cheap enough for a reachability probe.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        raise MLflowNotConfiguredError(
            "MLFLOW_TRACKING_URI is not set. Copy .env.example to .env and "
            "fill in the MLflow values, or export the variables directly."
        )
    return uri


def configure_mlflow(*, experiment: str | None = None) -> str:
    """Point the MLflow client at the configured server and return its URI.

    Resolves the tracking URI via :func:`tracking_uri` (which raises
    :class:`MLflowNotConfiguredError` when unset). ``MLFLOW_TRACKING_USERNAME``
    / ``_PASSWORD`` are read by the MLflow client itself and are optional
    here (a server without HTTP Basic Auth needs neither).

    Imports ``mlflow`` lazily so test code paths that never touch
    MLflow don't pay the import cost.
    """
    uri = tracking_uri()

    import mlflow  # type: ignore[import-untyped]

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or DEFAULT_EXPERIMENT)
    return uri
