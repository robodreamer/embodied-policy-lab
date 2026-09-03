#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXPECTED_REVISION="20c1b2b71ea35a415d5d47c39b04443cfadad7a1"
RELEASE_REVISION="f853cb49331aa0ab8124cbd1e1fb3a56e07a2523"
INTRINSICS_REVISION="0dedf7fcce85b4de2b82687c4f99c148b97b3a90"
DINO_REVISION="c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
CHECKPOINT_SHA256="1aca314666ffdd62ca1cb5b0e0b0e5f836b68b81b1ef455ae1641ddd12386211"
CONFIG_SHA256="46b00bf570f63bbe3465b9e81c476c8b9874c25fa278d74499fb4d1dc34e8650"
STATS_SHA256="8a7a12f54844e0ea1cb009d1e7db460be38dc39c6376e425c5cd2f428ef59880"
INTRINSICS_SHA256="f3acb280d90b37eaafc45ed435039721a8069ddbfc70bfb9dcaa3d29bbd184f2"
VAE_SHA256="0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"
T5_SHA256="d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23"
DINO_SHA256="1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"
TOKENIZER_JSON_SHA256="6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
TOKENIZER_MODEL_SHA256="e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"
TOKENIZER_CONFIG_SHA256="ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"
TOKENIZER_SPECIAL_SHA256="7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"
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

actual_revision="$(git -C "$FLEXPI_ROOT" rev-parse HEAD)"
if [[ "$actual_revision" == "$EXPECTED_REVISION" ]]; then
  echo "ok      upstream revision: $actual_revision"
else
  echo "wrong   upstream revision: $actual_revision (expected $EXPECTED_REVISION)" >&2
  echo "Refusing to change an existing checkout; pin it explicitly or set FLEXPI_DIR." >&2
  exit 1
fi

tracked_changes="$(
  git -C "$FLEXPI_ROOT" status --porcelain \
    --untracked-files=no --ignore-submodules=none
)"
if [[ -z "$tracked_changes" ]]; then
  echo "ok      tracked source and submodule worktrees are clean"
else
  echo "wrong   tracked source or submodule worktree has local changes:" >&2
  printf '%s\n' "$tracked_changes" >&2
  status=1
fi

EXPECTED_LIBERO_REVISION="$(
  git -C "$FLEXPI_ROOT" ls-tree HEAD third_party/LIBERO | awk '{print $3}'
)"
ACTUAL_LIBERO_REVISION="$(
  git -C "$FLEXPI_ROOT/third_party/LIBERO" rev-parse HEAD 2>/dev/null || true
)"
if [[ -n "$EXPECTED_LIBERO_REVISION" && \
      "$ACTUAL_LIBERO_REVISION" == "$EXPECTED_LIBERO_REVISION" ]]; then
  echo "ok      LIBERO submodule revision: $ACTUAL_LIBERO_REVISION"
else
  echo "wrong   LIBERO submodule revision: ${ACTUAL_LIBERO_REVISION:-missing} " \
    "(expected ${EXPECTED_LIBERO_REVISION:-gitlink missing})" >&2
  status=1
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
check "UMT5 tokenizer model" \
  "$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model"
check "UMT5 tokenizer config" \
  "$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json"
check "UMT5 special tokens" \
  "$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json"
HF_CACHE_BASE="${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}/hub}"
DINO_WEIGHTS="$HF_CACHE_BASE/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/$DINO_REVISION/model.safetensors"
check "DINOv3 ViT-B/16 weights" \
  "$DINO_WEIGHTS"

verify_sha256 "release checkpoint" "$CHECKPOINT" "$CHECKPOINT_SHA256"
verify_sha256 "release config" "$RELEASE_DIR/config.yaml" "$CONFIG_SHA256"
verify_sha256 "normalization stats" "$RELEASE_DIR/dataset_stats.json" "$STATS_SHA256"
for suite in libero_spatial libero_object libero_goal libero_10; do
  verify_sha256 "camera intrinsics ($suite)" \
    "$INTRINSICS_ROOT/${suite}_no_noops_lerobot/meta/camera_intrinsics.json" \
    "$INTRINSICS_SHA256"
done
verify_sha256 "Wan 2.2 VAE" \
  "$FLEXPI_ROOT/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
  "$VAE_SHA256"
verify_sha256 "UMT5 XXL encoder" \
  "$FLEXPI_ROOT/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
  "$T5_SHA256"
verify_sha256 "UMT5 tokenizer" \
  "$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json" \
  "$TOKENIZER_JSON_SHA256"
verify_sha256 "UMT5 tokenizer model" \
  "$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model" \
  "$TOKENIZER_MODEL_SHA256"
verify_sha256 "UMT5 tokenizer config" \
  "$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json" \
  "$TOKENIZER_CONFIG_SHA256"
verify_sha256 "UMT5 special tokens" \
  "$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json" \
  "$TOKENIZER_SPECIAL_SHA256"
verify_sha256 "DINOv3 ViT-B/16 weights" "$DINO_WEIGHTS" "$DINO_SHA256"

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
  --revision "$RELEASE_REVISION" \
  --local-dir "$RELEASE_DIR"
uvx hf@latest download flex-pi/libero_mujoco3.3.2_depth \
  --repo-type dataset \
  --revision "$INTRINSICS_REVISION" \
  --include '*/meta/camera_intrinsics.json' \
  --local-dir "$INTRINSICS_ROOT"

export DIFFSYNTH_MODEL_BASE_PATH="$FLEXPI_ROOT/checkpoints"
# ModelScope otherwise defaults to one range worker even for 11 GB files.
export MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-16}"
mkdir -p "$DIFFSYNTH_MODEL_BASE_PATH"
(
  cd "$FLEXPI_ROOT"
  "$PYTHON" "$PROJECT_DIR/scripts/download_flexpi_assets.py"
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

# Re-enter the read-only verifier now that installation/download/configuration
# are complete. This catches truncated or mutable remote artifacts before CUDA
# model construction can deserialize or consume them.
"$0" --check

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
