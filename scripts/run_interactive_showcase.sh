#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export INTERACTIVE=1
export HOLD_OPEN="${HOLD_OPEN:-1}"

if [[ -z "${LOCAL_LLM_URL:-}" && -z "${LOCAL_LLM_MODEL:-}" ]] && \
  command -v ollama >/dev/null 2>&1 && \
  curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && \
  ollama show gemma3:1b >/dev/null 2>&1; then
  export LOCAL_LLM_URL="http://127.0.0.1:11434/api/generate"
  export LOCAL_LLM_MODEL="gemma3:1b"
fi

exec "$SCRIPT_DIR/run_showcase.sh" "$@"
