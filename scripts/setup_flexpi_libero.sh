#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXPECTED_REVISION="20c1b2b71ea35a415d5d47c39b04443cfadad7a1"
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

find_flexpi() {
  if [[ -n "${FLEXPI_DIR:-}" ]]; then
    printf '%s\n' "$FLEXPI_DIR"
    return
  fi
  local candidate
  for candidate in "$PROJECT_DIR/../upstream-flexpi" \
    "$PROJECT_DIR/../../upstream-flexpi"; do
    if git -C "$candidate" rev-parse --git-dir >/dev/null 2>&1; then
      (cd "$candidate" && pwd)
      return
    fi
  done
  return 1
}

FLEXPI_ROOT="$(find_flexpi || true)"
if [[ -z "$FLEXPI_ROOT" ]]; then
  if [[ "$CHECK_ONLY" == "1" ]]; then
    echo "missing Flex-π source beside this repo (expected revision $EXPECTED_REVISION)" >&2
    exit 1
  fi
  FLEXPI_ROOT="$(cd "$PROJECT_DIR/.." && pwd)/upstream-flexpi"
  git clone https://github.com/geyan21/flex-pi.git "$FLEXPI_ROOT"
  git -C "$FLEXPI_ROOT" checkout "$EXPECTED_REVISION"
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

actual_revision="$(git -C "$FLEXPI_ROOT" rev-parse HEAD)"
if [[ "$actual_revision" == "$EXPECTED_REVISION" ]]; then
  echo "ok      upstream revision: $actual_revision"
else
  echo "wrong   upstream revision: $actual_revision (expected $EXPECTED_REVISION)" >&2
  echo "Refusing to change an existing checkout; pin it explicitly or set FLEXPI_DIR." >&2
  exit 1
fi

RELEASE_DIR="$FLEXPI_ROOT/runs/flexpi-libero"
CHECKPOINT="$RELEASE_DIR/checkpoints/weights/step_010860.pt"
INTRINSICS_ROOT="$FLEXPI_ROOT/data/libero-intrinsics"
check "Flex-π Python" "$FLEXPI_ROOT/.venv/bin/python"
check "LIBERO submodule" "$FLEXPI_ROOT/third_party/LIBERO/libero/libero/__init__.py"
check "release config" "$RELEASE_DIR/config.yaml"
check "normalization stats" "$RELEASE_DIR/dataset_stats.json"
check "release checkpoint" "$CHECKPOINT"
for suite in libero_spatial libero_object libero_goal libero_10; do
  check "published camera intrinsics ($suite)" \
    "$INTRINSICS_ROOT/${suite}_no_noops_lerobot/meta/camera_intrinsics.json"
done
check "Wan 2.2 VAE" \
  "$FLEXPI_ROOT/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
check "UMT5 XXL encoder" \
  "$FLEXPI_ROOT/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
check "UMT5 tokenizer" \
  "$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"
HF_CACHE_BASE="${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}/hub}"
DINO_WEIGHTS="$(
  find "$HF_CACHE_BASE/models--timm--vit_base_patch16_dinov3.lvd1689m" \
    -path '*/snapshots/*/model.safetensors' -print -quit 2>/dev/null || true
)"
check "DINOv3 ViT-B/16 weights" \
  "${DINO_WEIGHTS:-$HF_CACHE_BASE/models--timm--vit_base_patch16_dinov3.lvd1689m/.missing-model.safetensors}"

if [[ "$CHECK_ONLY" == "1" ]]; then
  exit "$status"
fi

git -C "$FLEXPI_ROOT" submodule update --init third_party/LIBERO
if [[ ! -x "$FLEXPI_ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.10 "$FLEXPI_ROOT/.venv"
fi
PYTHON="$FLEXPI_ROOT/.venv/bin/python"
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  uv pip install --python "$PYTHON" --index-strategy unsafe-best-match \
  -e "$FLEXPI_ROOT[libero]"
uv pip install --python "$PYTHON" --reinstall --no-deps \
  --index-url https://pypi.org/simple torchcodec==0.5
uv pip install --python "$PYTHON" --no-deps robosuite==1.4.0
uv pip install --python "$PYTHON" pytest tyro

mkdir -p "$RELEASE_DIR" "$INTRINSICS_ROOT"
uvx hf@latest download flex-pi/flexpi-libero \
  --local-dir "$RELEASE_DIR"
uvx hf@latest download flex-pi/libero_mujoco3.3.2_depth \
  --repo-type dataset \
  --include '*/meta/camera_intrinsics.json' \
  --local-dir "$INTRINSICS_ROOT"

export DIFFSYNTH_MODEL_BASE_PATH="$FLEXPI_ROOT/checkpoints"
# ModelScope otherwise defaults to one range worker even for 11 GB files.
export MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-16}"
mkdir -p "$DIFFSYNTH_MODEL_BASE_PATH"
(
  cd "$FLEXPI_ROOT"
  "$PYTHON" - <<'PY'
from flexpi.models.helpers.io import ModelConfig

for repository, pattern in (
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "google/umt5-xxl/"),
):
    print(f"Warming required inference asset: {repository} / {pattern}", flush=True)
    ModelConfig(model_id=repository, origin_file_pattern=pattern).download_if_necessary()

import timm
timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True)
print("Required VAE, T5/tokenizer, and DINOv3 assets are ready.")
PY
)

LIBERO_CONFIG="$FLEXPI_ROOT/.libero-config"
mkdir -p "$LIBERO_CONFIG"
if [[ ! -f "$LIBERO_CONFIG/config.yaml" ]]; then
  printf 'n\n' | env \
    PYTHONNOUSERSITE=1 \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG" \
    PYTHONPATH="$FLEXPI_ROOT/third_party/LIBERO:$FLEXPI_ROOT/src" \
    "$PYTHON" -c 'import libero.libero'
fi
env \
  PYTHONNOUSERSITE=1 \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG" \
  PYTHONPATH="$FLEXPI_ROOT/third_party/LIBERO:$FLEXPI_ROOT/src" \
  "$PYTHON" - <<'PY'
from flexpi.utils.libero_setup import assert_mujoco_pin
from libero.libero import benchmark
import torch

assert_mujoco_pin()
if not torch.cuda.is_available():
    raise SystemExit("Flex-π CUDA runtime is not ready")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.get_device_name())
print("LIBERO suites:", sorted(benchmark.get_benchmark_dict()))
PY

echo "Flex-π + LIBERO prerequisites are ready."
echo "World-action co-generation (default): ./scripts/run_interactive_showcase.sh --backend libero --model flexpi --task-id 0"
echo "Action-only: ./scripts/run_interactive_showcase.sh --backend libero --model flexpi --flexpi-mode action-only --task-id 0"
