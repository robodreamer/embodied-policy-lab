#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BACKEND="${BACKEND:-libero}"
PORT="${PI05_PORT:-8000}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

case "$BACKEND" in
  libero)
    OPENPI_DIR="$PROJECT_DIR/upstream-openpi"
    export OPENPI_DATA_HOME="$PROJECT_DIR/cache/openpi"
    cd "$OPENPI_DIR"
    exec uv run scripts/serve_policy.py --env LIBERO --port "$PORT"
    ;;
  robocasa)
    OPENPI_DIR="$PROJECT_DIR/upstream-robocasa-openpi"
    PYTHON="$OPENPI_DIR/.venv/bin/python"
    CHECKPOINT="${ROBOCASA_CHECKPOINT:-$PROJECT_DIR/cache/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
    CONFIG="${ROBOCASA_POLICY_CONFIG:-pi05_pretrain_human300}"
    if [[ ! -x "$PYTHON" ]]; then
      echo "RoboCasa is not set up. Run ./scripts/setup_robocasa.sh first." >&2
      exit 1
    fi
    if [[ ! -f "$CHECKPOINT/assets/norm_stats.json" ]] || \
      [[ ! -f "$CHECKPOINT/params/_METADATA" ]]; then
      echo "The RoboCasa π0.5 checkpoint is incomplete at $CHECKPOINT" >&2
      echo "Download it with: ROBOCASA_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_robocasa.sh" >&2
      exit 1
    fi
    export OPENPI_DATA_HOME="$PROJECT_DIR/cache/robocasa-openpi"
    exec "$PYTHON" "$PROJECT_DIR/showcase/serve_robocasa_policy.py" \
      --port "$PORT" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT"
    ;;
  *)
    echo "Unsupported backend: $BACKEND (choose libero or robocasa)" >&2
    exit 2
    ;;
esac
