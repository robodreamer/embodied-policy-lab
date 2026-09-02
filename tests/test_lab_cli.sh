#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

list_output="$($PROJECT_DIR/lab --list)"
grep -F "VLA     robocasa  pi05          action-only   ready / default" <<<"$list_output" >/dev/null
grep -F "VLA     robocasa  groot-n1.5    action-only   ready" <<<"$list_output" >/dev/null
grep -F "WAM     libero    fastwam       action-only   experimental" <<<"$list_output" >/dev/null
grep -F "WAM     libero    flexpi        full-joint    experimental" <<<"$list_output" >/dev/null
grep -F "WAM     libero    flexpi        action-only   experimental" <<<"$list_output" >/dev/null
grep -F "WAM     robotwin  fastwam       action-only   experimental" <<<"$list_output" >/dev/null
grep -F "WAM     robotwin  flexpi        full-joint    experimental" <<<"$list_output" >/dev/null
grep -F "WAM     robotwin  flexpi        action-only   experimental" <<<"$list_output" >/dev/null
if grep -E 'dino-wm|jepa-wm' <<<"$list_output" >/dev/null; then
  echo "unsupported learned predictors must not appear in ./lab --list" >&2
  exit 1
fi

while read -r backend model; do
  grep -E "^[^[:space:]]+[[:space:]]+${backend}[[:space:]]+${model}[[:space:]]" <<<"$list_output" >/dev/null || {
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
grep -F "Experiment: VLA" <<<"$default_output" >/dev/null
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
grep -F "Experiment: WAM" <<<"$fastwam_output" >/dev/null

flexpi_output="$("$PROJECT_DIR/lab" \
  --backend libero --model flexpi --mode interactive \
  --task-id 0 --default --dry-run)"
grep -F "Model: flexpi" <<<"$flexpi_output" >/dev/null
grep -F -- "--backend libero --model flexpi" <<<"$flexpi_output" >/dev/null
grep -F -- "--flexpi-mode full-joint" <<<"$flexpi_output" >/dev/null

flexpi_action_output="$("$PROJECT_DIR/lab" \
  --backend libero --model flexpi --flexpi-mode action-only \
  --mode interactive --task-id 0 --default --dry-run)"
grep -F "Flex-π inference: action-only" <<<"$flexpi_action_output" >/dev/null
grep -F -- "--flexpi-mode action-only" <<<"$flexpi_action_output" >/dev/null

wam_default_output="$("$PROJECT_DIR/lab" \
  --policy-family wam --default --dry-run)"
grep -F "Experiment: WAM" <<<"$wam_default_output" >/dev/null
grep -F "Simulator: libero" <<<"$wam_default_output" >/dev/null
grep -F "Model: fastwam" <<<"$wam_default_output" >/dev/null

robotwin_fastwam_output="$("$PROJECT_DIR/lab" \
  --backend robotwin --model fastwam \
  --task-set demo_clean --task-id click_bell --trials 2 --default --dry-run)"
grep -F "Simulator: robotwin" <<<"$robotwin_fastwam_output" >/dev/null
grep -F "Phase: demo_clean · task click_bell" <<<"$robotwin_fastwam_output" >/dev/null
grep -F "run_robotwin_evaluation.sh --model fastwam --task click_bell --phase demo_clean --trials 2" \
  <<<"$robotwin_fastwam_output" >/dev/null

robotwin_flexpi_output="$("$PROJECT_DIR/lab" \
  --backend robotwin --model flexpi --flexpi-mode action-only --mode batch \
  --task-set demo_randomized --task-id turn_switch --default --dry-run)"
grep -F "Flex-π inference: action-only" <<<"$robotwin_flexpi_output" >/dev/null
grep -F "run_robotwin_evaluation.sh --model flexpi --task turn_switch --phase demo_randomized --trials 1 --flexpi-mode action-only" \
  <<<"$robotwin_flexpi_output" >/dev/null

if "$PROJECT_DIR/lab" --backend robotwin --model fastwam \
  --mode interactive --default --dry-run >/dev/null 2>&1; then
  echo "expected RoboTwin interactive mode to fail until the shared studio adapter exists" >&2
  exit 1
fi

if "$PROJECT_DIR/lab" --backend robotwin --model pi05 \
  --mode batch --default --dry-run >/dev/null 2>&1; then
  echo "expected unvalidated RoboTwin + π0.5 profile to fail" >&2
  exit 1
fi

if "$PROJECT_DIR/lab" --policy-family vla --backend libero --model flexpi \
  --default --dry-run >/dev/null 2>&1; then
  echo "expected VLA + Flex-π family mismatch to fail" >&2
  exit 1
fi

if "$PROJECT_DIR/lab" --backend libero --model fastwam \
  --flexpi-mode action-only --default --dry-run >/dev/null 2>&1; then
  echo "expected --flexpi-mode with non-Flex-π model to fail" >&2
  exit 1
fi

grep -F 'exec "$SCRIPT_DIR/../lab"' \
  "$PROJECT_DIR/scripts/run_interactive_showcase.sh" >/dev/null
grep -F 'LIBERO_OPENPI_DIR="${OPENPI_DIR:-}"' \
  "$PROJECT_DIR/scripts/run_showcase.sh" >/dev/null
grep -F 'OPENPI_DATA_HOME="${LIBERO_OPENPI_DATA_HOME:-}"' \
  "$PROJECT_DIR/scripts/run_showcase.sh" >/dev/null

fake_openpi="$(mktemp -d)"
trap 'rm -rf "$fake_openpi"' EXIT
mkdir -p "$fake_openpi/scripts" \
  "$fake_openpi/cache/openpi-assets/checkpoints/pi05_libero"
: > "$fake_openpi/scripts/serve_policy.py"
server_trace="$(
  BACKEND=libero MODEL=pi05 POLICY_PORT=8123 \
    LIBERO_OPENPI_DIR="$fake_openpi" \
    OPENPI_DATA_HOME="$fake_openpi/cache" \
    OPENPI_UV_COMMAND=/bin/echo \
    bash -x "$PROJECT_DIR/scripts/run_server.sh" 2>&1
)"
grep -F "+ cd $fake_openpi" <<<"$server_trace" >/dev/null
grep -F "+ OPENPI_DATA_HOME=$fake_openpi/cache" <<<"$server_trace" >/dev/null
grep -F "run scripts/serve_policy.py --env LIBERO --port 8123" \
  <<<"$server_trace" >/dev/null

echo "lab CLI checks passed"
