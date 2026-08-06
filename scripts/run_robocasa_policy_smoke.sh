#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$PROJECT_DIR/upstream-robocasa-openpi"
PYTHON="$OPENPI_DIR/.venv/bin/python"
CHECKPOINT="${ROBOCASA_CHECKPOINT:-$PROJECT_DIR/cache/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
CONFIG="${ROBOCASA_POLICY_CONFIG:-pi05_pretrain_human300}"
PORT="${PI05_PORT:-8000}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/robocasa-policy-smoke/$TIMESTAMP}"

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

mkdir -p "$OUTPUT_DIR"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "${GPU_PID:-}" ]]; then
    kill "$GPU_PID" 2>/dev/null || true
    wait "$GPU_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

port_ready() {
  "$PYTHON" - "$PORT" <<'PY'
import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{int(sys.argv[1])}/healthz", timeout=0.25
    ) as response:
        raise SystemExit(response.status != 200)
except (OSError, urllib.error.URLError):
    raise SystemExit(1)
PY
}

if command -v ollama >/dev/null 2>&1; then
  while read -r model _; do
    [[ -n "$model" ]] && ollama stop "$model" >/dev/null 2>&1 || true
  done < <(ollama ps 2>/dev/null | tail -n +2)
fi

echo "Starting local RoboCasa π0.5 policy server..."
setsid env \
  OPENPI_DATA_HOME="$PROJECT_DIR/cache/robocasa-openpi" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}" \
  "$PYTHON" "$PROJECT_DIR/showcase/serve_robocasa_policy.py" \
  --port "$PORT" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  >"$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 900); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server failed. See $OUTPUT_DIR/server.log" >&2
    tail -100 "$OUTPUT_DIR/server.log" >&2
    exit 1
  fi
  if port_ready; then
    break
  fi
  sleep 1
done
if ! port_ready; then
  echo "Timed out waiting for the policy server. See $OUTPUT_DIR/server.log" >&2
  exit 1
fi

echo "timestamp,name,memory.used [MiB],utilization.gpu [%],power.draw [W],temperature.gpu" > "$OUTPUT_DIR/gpu.csv"
(
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "$OUTPUT_DIR/gpu.csv"
    sleep 1
  done
) &
GPU_PID=$!

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"$PYTHON" "$PROJECT_DIR/showcase/robocasa_policy_smoke.py" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --task-set "${ROBOCASA_TASK_SET:-atomic_seen}" \
  --task-id "${ROBOCASA_TASK_ID:-0}" \
  --split "${ROBOCASA_SPLIT:-target}" \
  --seed "${SEED:-7}" \
  --execute-actions "${ROBOCASA_SMOKE_ACTIONS:-5}" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/client.log"

echo "Local π0.5 RoboCasa smoke test complete: $OUTPUT_DIR/result.json"
