#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

{
  date --iso-8601=seconds
  uname -a
  lsb_release -ds
  echo
  lscpu | sed -n '1,24p'
  echo
  free -h
  echo
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  echo
  git -C "$PROJECT_DIR/upstream-openpi" rev-parse HEAD
  uv --version
  "$PROJECT_DIR/upstream-openpi/.venv/bin/python" --version
  "$PROJECT_DIR/upstream-openpi/examples/libero/.venv/bin/python" --version
} > "$PROJECT_DIR/logs/system-info.txt"

echo "Wrote $PROJECT_DIR/logs/system-info.txt"
