#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PORT="${PI05_PORT:-8000}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SERVER_LOG="$PROJECT_DIR/logs/server-$TIMESTAMP.log"
CLIENT_LOG="$PROJECT_DIR/logs/client-$TIMESTAMP.log"
GPU_LOG="$PROJECT_DIR/logs/gpu-$TIMESTAMP.csv"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/run" "$PROJECT_DIR/videos"

cleanup() {
  if [[ -n "${GPU_PID:-}" ]]; then
    kill "$GPU_PID" 2>/dev/null || true
    wait "$GPU_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting π0.5 policy server; log: $SERVER_LOG"
"$SCRIPT_DIR/run_server.sh" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PROJECT_DIR/run/server.pid"

for _ in $(seq 1 900); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server exited before becoming ready. See $SERVER_LOG" >&2
    tail -80 "$SERVER_LOG" >&2
    exit 1
  fi
  if "$PROJECT_DIR/upstream-openpi/.venv/bin/python" - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.2)
    raise SystemExit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
  then
    break
  fi
  sleep 1
done

if ! "$PROJECT_DIR/upstream-openpi/.venv/bin/python" - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1)
    raise SystemExit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
then
  echo "Timed out waiting for the policy server. See $SERVER_LOG" >&2
  exit 1
fi

echo "timestamp,name,memory.used [MiB],utilization.gpu [%],power.draw [W]" > "$GPU_LOG"
(
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits >> "$GPU_LOG"
    sleep 1
  done
) &
GPU_PID=$!

echo "Server ready. Running ${TASK_SUITE:-libero_spatial}, ${TRIALS_PER_TASK:-1} trial(s) per task."
set +e
"$SCRIPT_DIR/run_client.sh" 2>&1 | tee "$CLIENT_LOG"
CLIENT_STATUS=${PIPESTATUS[0]}
set -e

echo "Client exit status: $CLIENT_STATUS"
echo "Server log: $SERVER_LOG"
echo "Client log: $CLIENT_LOG"
echo "GPU log: $GPU_LOG"
echo "Videos: $PROJECT_DIR/videos/${TASK_SUITE:-libero_spatial}"
exit "$CLIENT_STATUS"
