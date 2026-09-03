#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SESSION_DIR="${1:-$PROJECT_DIR/showcase-runs/latest}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8085}"
DASHBOARD_PYTHON="${DASHBOARD_PYTHON:-$PROJECT_DIR/upstream-openpi/.venv/bin/python}"

if [[ ! -x "$DASHBOARD_PYTHON" ]]; then
  for candidate in \
    "$PROJECT_DIR/../../embodied-policy-lab/upstream-openpi/.venv/bin/python" \
    "$PROJECT_DIR/../upstream-fastwam/.venv/bin/python" \
    "$PROJECT_DIR/../../upstream-fastwam/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
      DASHBOARD_PYTHON="$candidate"
      break
    fi
  done
fi
if [[ ! -x "$DASHBOARD_PYTHON" ]]; then
  DASHBOARD_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$DASHBOARD_PYTHON" ]] || [[ ! -x "$DASHBOARD_PYTHON" ]]; then
  echo "No Python runtime found for the dependency-free dashboard server." >&2
  exit 1
fi

if [[ ! -f "$SESSION_DIR/state.json" ]]; then
  echo "No showcase session found at $SESSION_DIR" >&2
  exit 1
fi

echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT"
if [[ -n "${DISPLAY:-}" ]]; then
  xdg-open "http://127.0.0.1:$DASHBOARD_PORT" >/dev/null 2>&1 || true
fi
exec "$DASHBOARD_PYTHON" "$PROJECT_DIR/showcase/dashboard_server.py" \
  --session-dir "$SESSION_DIR" \
  --static-dir "$PROJECT_DIR/showcase/static" \
  --port "$DASHBOARD_PORT"
