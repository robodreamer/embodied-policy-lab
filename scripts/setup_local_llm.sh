#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODEL="${LOCAL_LLM_MODEL:-gemma3:1b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed." >&2
  echo "Follow the official Linux instructions: https://docs.ollama.com/linux" >&2
  exit 1
fi

if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/run"
  setsid ollama serve > "$PROJECT_DIR/logs/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  printf '%s\n' "$OLLAMA_PID" > "$PROJECT_DIR/run/ollama.pid"
  for _ in $(seq 1 60); do
    curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      echo "Ollama failed to start. See logs/ollama.log" >&2
      exit 1
    fi
    sleep 1
  done
fi

echo "Downloading/checking local prompt model: $MODEL"
ollama pull "$MODEL"
echo "Local prompt generation is ready."
echo "Run: $PROJECT_DIR/scripts/run_interactive_showcase.sh"
