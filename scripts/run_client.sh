#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$PROJECT_DIR/upstream-openpi"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-1}"
SEED="${SEED:-7}"
PORT="${PI05_PORT:-8000}"

export LIBERO_CONFIG_PATH="$PROJECT_DIR/config/libero"
export PYTHONPATH="$OPENPI_DIR/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

cd "$OPENPI_DIR"
exec examples/libero/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port "$PORT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task "$TRIALS_PER_TASK" \
  --args.seed "$SEED" \
  --args.video-out-path "$PROJECT_DIR/videos/$TASK_SUITE"
