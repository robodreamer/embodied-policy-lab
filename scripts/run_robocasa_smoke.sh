#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/upstream-robocasa-openpi/.venv/bin/python"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/robocasa-smoke/$TIMESTAMP}"

if [[ ! -x "$PYTHON" ]]; then
  echo "RoboCasa is not set up. Run ./scripts/setup_robocasa.sh first." >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$PYTHON" "$PROJECT_DIR/showcase/robocasa_smoke.py" \
  --task-set "${ROBOCASA_TASK_SET:-atomic_seen}" \
  --task-id "${ROBOCASA_TASK_ID:-0}" \
  --split "${ROBOCASA_SPLIT:-target}" \
  --seed "${SEED:-7}" \
  --steps "${ROBOCASA_SMOKE_STEPS:-5}" \
  --output-dir "$OUTPUT_DIR"
