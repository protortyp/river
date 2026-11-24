set dotenv-load := true

default:
    @just --list

# Lint and type-check
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check

# Auto-format Python code
fmt:
    uv run ruff format .

# Run Python test suite
test:
    uv run pytest

# Run web encoder, engine, and ONNX parity tests
test-web:
    cd web && npm test

# Run GPU environment throughput benchmark
bench:
    uv run python bench/run.py

# Train with default Hydra config (override with args, e.g. `just train triad_enabled=true`)
train *args:
    uv run python train/train.py {{args}}

# Export a checkpoint to ONNX for browser or Slumbot inference
export-onnx ckpt out="web/static/model.onnx":
    uv run python train/export_onnx.py --checkpoint {{ckpt}} --out {{out}}

# Evaluate an ONNX model against Slumbot
eval-slumbot model hands="1000":
    uv run python train/slumbot_eval.py --model {{model}} --hands {{hands}}
