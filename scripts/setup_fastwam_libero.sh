#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXPECTED_REVISION="45d8e1458921d83f8ad6cf9ce993d371208dabd0"
RELEASE_REVISION="05680dbe51815d18bbe70d3faa61b2cffaf03cd7"
CHECKPOINT_SHA256="1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
STATS_SHA256="30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
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
  if [[ "$CHECK_ONLY" == "1" ]]; then
    echo "missing Fast-WAM source beside this repo (expected revision $EXPECTED_REVISION)" >&2
    exit 1
  fi
  FASTWAM_ROOT="$(cd "$PROJECT_DIR/.." && pwd)/upstream-fastwam"
  git clone https://github.com/yuantianyuan01/FastWAM.git "$FASTWAM_ROOT"
  git -C "$FASTWAM_ROOT" checkout "$EXPECTED_REVISION"
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

verify_sha256() {
  local description="$1" path="$2" expected="$3" actual
  if [[ ! -f "$path" ]]; then
    return
  fi
  actual="$(sha256sum "$path" | cut -d ' ' -f 1)"
  if [[ "$actual" == "$expected" ]]; then
    printf 'ok      %s sha256: %s\n' "$description" "$actual"
  else
    printf 'wrong   %s sha256: %s (expected %s)\n' \
      "$description" "$actual" "$expected" >&2
    status=1
  fi
}

actual_revision="$(git -C "$FASTWAM_ROOT" rev-parse HEAD)"
if [[ "$actual_revision" == "$EXPECTED_REVISION" ]]; then
  echo "ok      upstream revision: $actual_revision"
else
  echo "wrong   upstream revision: $actual_revision (expected $EXPECTED_REVISION)" >&2
  echo "Refusing to change an existing checkout; pin it explicitly or set FASTWAM_DIR." >&2
  exit 1
fi

RELEASE_DIR="$FASTWAM_ROOT/checkpoints/fastwam_release"
CHECKPOINT="$RELEASE_DIR/libero_uncond_2cam224.pt"
STATS="$RELEASE_DIR/libero_uncond_2cam224_dataset_stats.json"
if [[ "$CHECK_ONLY" != "1" ]]; then
  if [[ ! -x "$FASTWAM_ROOT/.venv/bin/python" ]]; then
    uv venv --python 3.10 "$FASTWAM_ROOT/.venv"
  fi
  PYTHON="$FASTWAM_ROOT/.venv/bin/python"
  UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
    uv pip install --python "$PYTHON" --index-strategy unsafe-best-match \
    -e "$FASTWAM_ROOT"
  mkdir -p "$RELEASE_DIR"
  uvx hf@latest download yuanty/fastwam \
    libero_uncond_2cam224.pt \
    libero_uncond_2cam224_dataset_stats.json \
    --revision "$RELEASE_REVISION" \
    --local-dir "$RELEASE_DIR"
fi

check "Fast-WAM Python" "$FASTWAM_ROOT/.venv/bin/python"
check "release checkpoint" "$CHECKPOINT"
check "normalization stats" "$STATS"
verify_sha256 "release checkpoint" "$CHECKPOINT" "$CHECKPOINT_SHA256"
verify_sha256 "normalization stats" "$STATS" "$STATS_SHA256"

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
  echo "Rerun $0 to install the pinned Fast-WAM runtime and release assets." >&2
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
