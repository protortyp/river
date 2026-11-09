"""Helpers shared by the benchmark scripts."""

import sys
from pathlib import Path

import torch
import warp as wp

# Scripts run as `python bench/x.py` have only `bench/` on sys.path, so the
# repo root is added to make the `train` package importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    wp.synchronize()
