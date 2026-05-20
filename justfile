# Connection details (SSH target + remote path) come from the gitignored
# .env -- see .env.example for the keys.
set dotenv-load := true

# List available recipes (runs when `just` is invoked with no arguments).
default:
    @just --list

# Lint + type-check: `ruff check` then `ty check`.
check:
    uv run ruff check .
    uv run ty check

# Auto-format the codebase in place.
fmt:
    uv run ruff format .

# Run the test suite.
test:
    uv run pytest

# Run the GPU environment benchmark.
bench:
    uv run python bench/run.py

# Deploy the web UI to production.
deploy:
    ssh -i ~/.ssh/id_ed25519 -o IdentityAgent=none "$DEPLOY_SSH_TARGET" "cd $DEPLOY_SERVER_PATH && git pull && ./deploy_pokergpu.sh"
