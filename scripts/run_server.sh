#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BACKEND="${BACKEND:-libero}"
MODEL="${MODEL:-pi05}"
PORT="${POLICY_PORT:-${PI05_PORT:-8000}}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

case "$BACKEND" in
  libero)
    if [[ "$MODEL" != "pi05" ]]; then
      echo "Model $MODEL does not support LIBERO in this repository." >&2
      exit 2
    fi
    OPENPI_DIR="$PROJECT_DIR/upstream-openpi"
    export OPENPI_DATA_HOME="$PROJECT_DIR/cache/openpi"
    cd "$OPENPI_DIR"
    exec uv run scripts/serve_policy.py --env LIBERO --port "$PORT"
    ;;
  robocasa)
    case "$MODEL" in
      pi05)
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
      groot-n1.5)
        GROOT_DIR="$PROJECT_DIR/upstream-robocasa-groot"
        PYTHON="$GROOT_DIR/.venv/bin/python"
        CHECKPOINT="${GROOT_CHECKPOINT:-$PROJECT_DIR/cache/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000}"
        if [[ ! -x "$PYTHON" ]]; then
          echo "GR00T is not set up. Run ./scripts/setup_groot.sh first." >&2
          exit 1
        fi
        if [[ ! -f "$CHECKPOINT/config.json" ]] || \
          [[ ! -f "$CHECKPOINT/model.safetensors.index.json" ]] || \
          [[ ! -f "$CHECKPOINT/experiment_cfg/metadata.json" ]]; then
          echo "The RoboCasa GR00T checkpoint is incomplete at $CHECKPOINT" >&2
          echo "Download it with: GROOT_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_groot.sh" >&2
          exit 1
        fi
        export NO_ALBUMENTATIONS_UPDATE=1
        exec "$PYTHON" "$PROJECT_DIR/showcase/serve_groot_policy.py" \
          --port "$PORT" \
          --checkpoint "$CHECKPOINT"
        ;;
      *)
        echo "Unsupported RoboCasa model: $MODEL (choose pi05 or groot-n1.5)" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unsupported backend: $BACKEND (choose libero or robocasa)" >&2
    exit 2
    ;;
esac
