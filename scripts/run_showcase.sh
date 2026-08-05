#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$PROJECT_DIR/upstream-openpi"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_DIR="${SESSION_DIR:-$PROJECT_DIR/showcase-runs/$TIMESTAMP}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_IDS="${TASK_IDS:-0}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-1}"
SEED="${SEED:-7}"
PI05_PORT="${PI05_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8085}"
REALTIME_DELAY_MS="${REALTIME_DELAY_MS:-35}"
NETWORK_AUDIT="${NETWORK_AUDIT:-1}"
AUTO_OPEN="${AUTO_OPEN:-1}"
HOLD_OPEN="${HOLD_OPEN:-0}"
INTERACTIVE="${INTERACTIVE:-0}"
INITIAL_PROMPT="${INITIAL_PROMPT:-}"
LOCAL_LLM_URL="${LOCAL_LLM_URL:-}"
LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-}"

mkdir -p "$SESSION_DIR/frames" "$SESSION_DIR/videos" "$PROJECT_DIR/showcase-runs"
ln -sfn "$SESSION_DIR" "$PROJECT_DIR/showcase-runs/latest"

cleanup() {
  for process_group in "${DASHBOARD_PID:-}" "${SERVER_PID:-}"; do
    if [[ -n "$process_group" ]]; then
      kill -- "-$process_group" 2>/dev/null || true
      wait "$process_group" 2>/dev/null || true
    fi
  done
  if [[ -n "${GPU_PID:-}" ]]; then
    kill "$GPU_PID" 2>/dev/null || true
    wait "$GPU_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

port_ready() {
  "$OPENPI_DIR/.venv/bin/python" - "$1" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.25)
    raise SystemExit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
}

wait_for_port() {
  local port="$1"
  local owner_pid="$2"
  for _ in $(seq 1 900); do
    if ! kill -0 "$owner_pid" 2>/dev/null; then
      return 1
    fi
    if port_ready "$port"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

echo "Starting local π0.5 server..."
if [[ "$NETWORK_AUDIT" == "1" ]]; then
  setsid strace -f -e trace=network -s 256 -o "$SESSION_DIR/network-server.log" \
    env PI05_PORT="$PI05_PORT" "$SCRIPT_DIR/run_server.sh" > "$SESSION_DIR/server.log" 2>&1 &
else
  setsid env PI05_PORT="$PI05_PORT" "$SCRIPT_DIR/run_server.sh" > "$SESSION_DIR/server.log" 2>&1 &
fi
SERVER_PID=$!

if ! wait_for_port "$PI05_PORT" "$SERVER_PID"; then
  echo "Policy server failed. See $SESSION_DIR/server.log" >&2
  tail -80 "$SESSION_DIR/server.log" >&2
  exit 1
fi

echo "timestamp,name,memory.used [MiB],utilization.gpu [%],power.draw [W],temperature.gpu" > "$SESSION_DIR/gpu.csv"
(
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "$SESSION_DIR/gpu.csv"
    sleep 1
  done
) &
GPU_PID=$!

Dashboard_COMMAND=(
  "$OPENPI_DIR/.venv/bin/python" "$PROJECT_DIR/showcase/dashboard_server.py"
  --session-dir "$SESSION_DIR" \
  --static-dir "$PROJECT_DIR/showcase/static" \
  --port "$DASHBOARD_PORT"
)
if [[ -n "$LOCAL_LLM_URL" || -n "$LOCAL_LLM_MODEL" ]]; then
  if [[ -z "$LOCAL_LLM_URL" || -z "$LOCAL_LLM_MODEL" ]]; then
    echo "LOCAL_LLM_URL and LOCAL_LLM_MODEL must be set together." >&2
    exit 2
  fi
  Dashboard_COMMAND+=(--local-llm-url "$LOCAL_LLM_URL" --local-llm-model "$LOCAL_LLM_MODEL")
fi
setsid "${Dashboard_COMMAND[@]}" > "$SESSION_DIR/dashboard.log" 2>&1 &
DASHBOARD_PID=$!

if ! wait_for_port "$DASHBOARD_PORT" "$DASHBOARD_PID"; then
  echo "Dashboard failed. See $SESSION_DIR/dashboard.log" >&2
  exit 1
fi

DASHBOARD_URL="http://127.0.0.1:$DASHBOARD_PORT"
echo "Dashboard: $DASHBOARD_URL"
echo "Session: $SESSION_DIR"
if [[ "$AUTO_OPEN" == "1" ]] && [[ -n "${DISPLAY:-}" ]]; then
  xdg-open "$DASHBOARD_URL" >/dev/null 2>&1 || true
fi

export LIBERO_CONFIG_PATH="$PROJECT_DIR/config/libero"
export PYTHONPATH="$OPENPI_DIR/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [[ "$INTERACTIVE" == "1" ]]; then
  if [[ ! "$TASK_IDS" =~ ^[0-9]+$ ]]; then
    echo "Interactive mode requires one numeric TASK_IDS value." >&2
    exit 2
  fi
  CLIENT_COMMAND=(
    "$OPENPI_DIR/examples/libero/.venv/bin/python"
    "$PROJECT_DIR/showcase/interactive_libero.py"
    --host 127.0.0.1
    --port "$PI05_PORT"
    --task-suite-name "$TASK_SUITE"
    --task-id "$TASK_IDS"
    --video-out-path "$SESSION_DIR/videos"
    --session-dir "$SESSION_DIR"
    --seed "$SEED"
    --realtime-delay-ms "$REALTIME_DELAY_MS"
    --initial-prompt "$INITIAL_PROMPT"
  )
else
  CLIENT_COMMAND=(
    "$OPENPI_DIR/examples/libero/.venv/bin/python"
    "$PROJECT_DIR/showcase/instrumented_libero.py"
    --host 127.0.0.1
    --port "$PI05_PORT"
    --task-suite-name "$TASK_SUITE"
    --task-ids "$TASK_IDS"
    --num-trials-per-task "$TRIALS_PER_TASK"
    --video-out-path "$SESSION_DIR/videos"
    --session-dir "$SESSION_DIR"
    --seed "$SEED"
    --realtime-delay-ms "$REALTIME_DELAY_MS"
  )
fi
if [[ "$NETWORK_AUDIT" == "1" ]]; then
  CLIENT_COMMAND+=(--network-audit)
else
  CLIENT_COMMAND+=(--no-network-audit)
fi

set +e
if [[ "$NETWORK_AUDIT" == "1" ]]; then
  strace -f -e trace=network -s 256 -o "$SESSION_DIR/network-client.log" \
    "${CLIENT_COMMAND[@]}" 2>&1 | tee "$SESSION_DIR/client.log"
  CLIENT_STATUS=${PIPESTATUS[0]}
else
  "${CLIENT_COMMAND[@]}" 2>&1 | tee "$SESSION_DIR/client.log"
  CLIENT_STATUS=${PIPESTATUS[0]}
fi
set -e

# Stop the traced inference runtime before reporting so network logs are final.
kill -- "-$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
kill "$GPU_PID" 2>/dev/null || true
wait "$GPU_PID" 2>/dev/null || true
GPU_PID=""

"$OPENPI_DIR/.venv/bin/python" "$PROJECT_DIR/showcase/generate_report.py" "$SESSION_DIR"
echo "Showcase exit status: $CLIENT_STATUS"
echo "Report: $SESSION_DIR/report.md"
echo "Dashboard snapshot data: $SESSION_DIR/state.json"

if [[ "$HOLD_OPEN" == "1" ]]; then
  echo "Dashboard remains available at $DASHBOARD_URL; press Ctrl+C to stop."
  wait "$DASHBOARD_PID"
fi

exit "$CLIENT_STATUS"
