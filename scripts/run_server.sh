#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$PROJECT_DIR/upstream-openpi"

export OPENPI_DATA_HOME="$PROJECT_DIR/cache/openpi"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

cd "$OPENPI_DIR"
exec uv run scripts/serve_policy.py --env LIBERO --port "${PI05_PORT:-8000}"
