#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

list_output="$($PROJECT_DIR/lab --list)"
grep -F "robocasa  pi05          ready / default" <<<"$list_output" >/dev/null
grep -F "robocasa  groot-n1.5    ready" <<<"$list_output" >/dev/null
grep -F "libero    fastwam       experimental" <<<"$list_output" >/dev/null
if grep -E 'dino-wm|jepa-wm' <<<"$list_output" >/dev/null; then
  echo "unsupported learned predictors must not appear in ./lab --list" >&2
  exit 1
fi

while read -r backend model; do
  grep -E "^${backend}[[:space:]]+${model}[[:space:]]" <<<"$list_output" >/dev/null || {
    echo "registry profile missing from ./lab --list: $backend/$model" >&2
    exit 1
  }
done < <(python3 - "$PROJECT_DIR" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from showcase import backend_registry

for backend, model in sorted(backend_registry.PROFILES):
    print(backend, model)
PY
)

default_output="$($PROJECT_DIR/lab --default --dry-run)"
grep -F "Simulator: robocasa" <<<"$default_output" >/dev/null
grep -F "Model: pi05" <<<"$default_output" >/dev/null
grep -F "Predictor: none" <<<"$default_output" >/dev/null
grep -F -- "--world-model none" <<<"$default_output" >/dev/null
grep -F -- "--task-set atomic_seen --task-id 0" <<<"$default_output" >/dev/null

groot_output="$($PROJECT_DIR/lab \
  --backend robocasa --model groot-n1.5 --mode batch \
  --task-id 2 --trials 3 --default --dry-run)"
grep -F "Model: groot-n1.5" <<<"$groot_output" >/dev/null
grep -F -- "--batch --trials 3" <<<"$groot_output" >/dev/null

if "$PROJECT_DIR/lab" --backend libero --model groot-n1.5 \
  --default --dry-run >/dev/null 2>&1; then
  echo "expected incompatible LIBERO + GR00T profile to fail" >&2
  exit 1
fi

fastwam_output="$("$PROJECT_DIR/lab" \
  --backend libero --model fastwam --mode batch \
  --task-id 2 --trials 1 --default --dry-run)"
grep -F "Model: fastwam" <<<"$fastwam_output" >/dev/null
grep -F -- "--backend libero --model fastwam" <<<"$fastwam_output" >/dev/null

echo "lab CLI checks passed"
