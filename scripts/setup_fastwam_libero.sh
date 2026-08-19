#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXPECTED_REVISION="45d8e1458921d83f8ad6cf9ce993d371208dabd0"
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

find_fastwam() {
  if [[ -n "${FASTWAM_DIR:-}" ]]; then
    printf '%s\n' "$FASTWAM_DIR"
    return
  fi
  local candidate
  for candidate in "$PROJECT_DIR/../upstream-fastwam" \
    "$PROJECT_DIR/../../upstream-fastwam"; do
    if git -C "$candidate" rev-parse --git-dir >/dev/null 2>&1; then
      (cd "$candidate" && pwd)
      return
    fi
  done
  return 1
}

FASTWAM_ROOT="$(find_fastwam || true)"
if [[ -z "$FASTWAM_ROOT" ]]; then
  echo "Fast-WAM source is missing. Clone the pinned upstream beside this repo:" >&2
  echo "  git clone https://github.com/yuantianyuan01/FastWAM.git ../upstream-fastwam" >&2
  echo "  git -C ../upstream-fastwam checkout $EXPECTED_REVISION" >&2
  exit 1
fi

status=0
check() {
  local description="$1" path="$2"
  if [[ -e "$path" ]]; then
    printf 'ok      %s: %s\n' "$description" "$path"
  else
    printf 'missing %s: %s\n' "$description" "$path" >&2
    status=1
  fi
}

actual_revision="$(git -C "$FASTWAM_ROOT" rev-parse HEAD)"
if [[ "$actual_revision" == "$EXPECTED_REVISION" ]]; then
  echo "ok      upstream revision: $actual_revision"
else
  echo "wrong   upstream revision: $actual_revision (expected $EXPECTED_REVISION)" >&2
  status=1
fi
check "Fast-WAM Python" "$FASTWAM_ROOT/.venv/bin/python"
check "release checkpoint" "$FASTWAM_ROOT/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
check "normalization stats" "$FASTWAM_ROOT/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"

OPENPI_ROOT="${LIBERO_OPENPI_DIR:-$PROJECT_DIR/upstream-openpi}"
if [[ ! -x "$OPENPI_ROOT/examples/libero/.venv/bin/python" ]]; then
  candidate="$PROJECT_DIR/../../embodied-policy-lab/upstream-openpi"
  if [[ -x "$candidate/examples/libero/.venv/bin/python" ]]; then
    OPENPI_ROOT="$(cd "$candidate" && pwd)"
  fi
fi
check "LIBERO simulator Python" "$OPENPI_ROOT/examples/libero/.venv/bin/python"
if [[ -f "$PROJECT_DIR/config/libero/config.yaml" ]]; then
  echo "ok      LIBERO path config: $PROJECT_DIR/config/libero/config.yaml"
elif [[ "$CHECK_ONLY" == "1" ]]; then
  echo "missing LIBERO path config: $PROJECT_DIR/config/libero/config.yaml" >&2
  status=1
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  if [[ "$status" != "0" ]]; then
    echo "Run $0 without --check to create local LIBERO configuration after prerequisites exist." >&2
  fi
  exit "$status"
fi
if [[ "$status" != "0" ]]; then
  echo >&2
  echo "Install Fast-WAM using its pinned upstream instructions and release assets." >&2
  echo "Set up the LIBERO client with ./scripts/setup.sh if it is the missing item." >&2
  exit "$status"
fi

mkdir -p "$PROJECT_DIR/config/libero"
if [[ ! -f "$PROJECT_DIR/config/libero/config.yaml" ]]; then
  printf 'n\n' | env \
    LIBERO_CONFIG_PATH="$PROJECT_DIR/config/libero" \
    PYTHONPATH="$OPENPI_ROOT/third_party/libero" \
    "$OPENPI_ROOT/examples/libero/.venv/bin/python" -c 'import libero.libero'
fi

env \
  LIBERO_CONFIG_PATH="$PROJECT_DIR/config/libero" \
  PYTHONPATH="$OPENPI_ROOT/third_party/libero" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  "$OPENPI_ROOT/examples/libero/.venv/bin/python" - <<'PY'
from libero.libero import benchmark
import mujoco

print("MuJoCo:", mujoco.__version__)
if mujoco.__version__ != "3.3.2":
    print("WARNING: Fast-WAM requests MuJoCo 3.3.2; record this version mismatch in results.")
print("LIBERO suites:", sorted(benchmark.get_benchmark_dict()))
PY

"$FASTWAM_ROOT/.venv/bin/python" - <<'PY'
import hydra
import omegaconf
import PIL
import torch
import fastwam

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Fast-WAM import:", fastwam.__file__)
print("Hydra:", hydra.__version__)
print("OmegaConf:", omegaconf.__version__)
print("Pillow:", PIL.__version__)
if not torch.cuda.is_available():
    raise SystemExit("Fast-WAM CUDA runtime is not ready")
PY

echo "Fast-WAM + LIBERO prerequisites are ready."
echo "Run: ./scripts/run_interactive_showcase.sh --backend libero --model fastwam --task-id 2"
