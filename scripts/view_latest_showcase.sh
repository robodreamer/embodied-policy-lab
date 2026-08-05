#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SESSION_DIR="${1:-$PROJECT_DIR/showcase-runs/latest}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8085}"

if [[ ! -f "$SESSION_DIR/state.json" ]]; then
  echo "No showcase session found at $SESSION_DIR" >&2
  exit 1
fi

echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT"
if [[ -n "${DISPLAY:-}" ]]; then
  xdg-open "http://127.0.0.1:$DASHBOARD_PORT" >/dev/null 2>&1 || true
fi
exec "$PROJECT_DIR/upstream-openpi/.venv/bin/python" "$PROJECT_DIR/showcase/dashboard_server.py" \
  --session-dir "$SESSION_DIR" \
  --static-dir "$PROJECT_DIR/showcase/static" \
  --port "$DASHBOARD_PORT"
