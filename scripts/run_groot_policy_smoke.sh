#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
GROOT_DIR="$PROJECT_DIR/upstream-robocasa-groot"
PYTHON="$GROOT_DIR/.venv/bin/python"
CHECKPOINT="${GROOT_CHECKPOINT:-$PROJECT_DIR/cache/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000}"
PORT="${POLICY_PORT:-8000}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/groot-policy-smoke/$TIMESTAMP}"

if [[ ! -x "$PYTHON" ]]; then
  echo "GR00T is not set up. Run ./scripts/setup_groot.sh first." >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT/config.json" ]] || \
  [[ ! -f "$CHECKPOINT/model.safetensors.index.json" ]] || \
  [[ ! -f "$CHECKPOINT/experiment_cfg/metadata.json" ]]; then
  echo "The GR00T checkpoint is incomplete at $CHECKPOINT" >&2
  echo "Download it with: GROOT_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_groot.sh" >&2
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

listener_ready() {
  ss -ltn "sport = :$PORT" | awk 'NR > 1 {found=1} END {exit !found}'
}

if command -v ollama >/dev/null 2>&1; then
  ollama stop "${LOCAL_LLM_MODEL:-gemma3:1b}" >/dev/null 2>&1 || true
fi

GPU_COMPUTE_APPS="$(
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "$GPU_COMPUTE_APPS" && "${ALLOW_GPU_OVERSUBSCRIPTION:-0}" != "1" ]]; then
  echo "The GPU is already occupied by another compute process:" >&2
  echo "$GPU_COMPUTE_APPS" >&2
  echo "Stop it before the GR00T smoke test, or set ALLOW_GPU_OVERSUBSCRIPTION=1." >&2
  exit 1
fi

echo "Starting local RoboCasa GR00T N1.5 policy server..."
setsid env BACKEND=robocasa MODEL=groot-n1.5 POLICY_PORT="$PORT" \
  NO_ALBUMENTATIONS_UPDATE=1 "$SCRIPT_DIR/run_server.sh" \
  >"$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 900); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server failed. See $OUTPUT_DIR/server.log" >&2
    tail -100 "$OUTPUT_DIR/server.log" >&2
    exit 1
  fi
  if listener_ready; then
    break
  fi
  sleep 1
done
if ! listener_ready; then
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
export NO_ALBUMENTATIONS_UPDATE=1

"$PYTHON" "$PROJECT_DIR/showcase/groot_policy_smoke.py" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --task-set "${ROBOCASA_TASK_SET:-atomic_seen}" \
  --task-id "${ROBOCASA_TASK_ID:-0}" \
  --split "${ROBOCASA_SPLIT:-target}" \
  --seed "${SEED:-7}" \
  --execute-actions "${GROOT_SMOKE_ACTIONS:-1}" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/client.log"

echo "Local GR00T RoboCasa smoke test complete: $OUTPUT_DIR/result.json"
