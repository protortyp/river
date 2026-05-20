#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- 1. Find Project Root ---
# Get the directory where the script is located. This makes the script runnable from anywhere.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# The project root is one level above the 'train' directory where this script lives.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
echo "Found Project Root: $PROJECT_ROOT"

# --- 2. Find Latest Run & Checkpoint ---
# Find the latest Hydra run directory relative to the project root.
LATEST_RUN_DIR=$(ls -td "$PROJECT_ROOT"/train/outputs/*/* 2>/dev/null | head -n 1)
if [ -z "$LATEST_RUN_DIR" ] || [ ! -d "$LATEST_RUN_DIR" ]; then
  echo "Error: Could not find any previous training run in '$PROJECT_ROOT/train/outputs/'."
  exit 1
fi
echo "Resuming in directory: $LATEST_RUN_DIR"

# Find the latest checkpoint file in the global league directory.
LATEST_CKPT=$(ls -t "$PROJECT_ROOT"/global_league/checkpoints/*.pt 2>/dev/null | head -n 1)
if [ -z "$LATEST_CKPT" ] || [ ! -f "$LATEST_CKPT" ]; then
  echo "Error: Could not find any checkpoint (.pt) file in '$PROJECT_ROOT/global_league/checkpoints/'."
  exit 1
fi
echo "Using checkpoint: $LATEST_CKPT"
echo "---"

# --- 3. Execute Continue Command ---
echo "Starting continued training run..."
# Change to the project root so that all paths for hydra/uv resolve correctly.
cd "$PROJECT_ROOT"
PROJECT_ROOT="$PROJECT_ROOT" uv run train/train.py \
    compile_policy=true \
    resume_from="$LATEST_CKPT" \
    hydra.run.dir="$LATEST_RUN_DIR"
