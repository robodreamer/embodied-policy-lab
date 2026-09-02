#!/usr/bin/env bash
# Prepare revision-pinned, isolated RoboTwin runtimes without implicit large downloads.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

ROBOTWIN_REVISION="bf44be51cf5717a5595ce59447f2cf5263d2aa95"
FASTWAM_REVISION="45d8e1458921d83f8ad6cf9ce993d371208dabd0"
FLEXPI_REVISION="20c1b2b71ea35a415d5d47c39b04443cfadad7a1"
CUROBO_REVISION="8e734f3ced1df898990bcd92de40abce475907db"
PYTORCH3D_REVISION="75ebeeaea0908c5527e7b1e305fbc7681382db47"
FASTWAM_HF_REVISION="8eaceeb24c3cc92ff2a9c9a9d266a4941b836705"
FLEXPI_HF_REVISION="87d3833ea3bd89c4922945631db81b346e780785"

FASTWAM_CHECKPOINT_SHA256="776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63"
ROBOTWIN_STATS_SHA256="7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095"
FLEXPI_CHECKPOINT_SHA256="98a13399bef2fcb17a8b2815852af08bbc45a14773e86a0cfa69af44a3d5da15"
FLEXPI_CONFIG_SHA256="2b6f6ad60793cb97cd51f32e83e2ab63ffa7318be269476dfaa59d6089433200"
VAE_SHA256="0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"
T5_SHA256="d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23"
DINO_SHA256="1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"
TOKENIZER_JSON_SHA256="6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
TOKENIZER_MODEL_SHA256="e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"

die() { echo "setup_robotwin: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: scripts/setup_robotwin.sh [OPTIONS]

Options:
  --model NAME             fastwam, flexpi, or both (default: both)
  --check                  read-only readiness and integrity check
  --download-assets        fetch the official ~16 GB RoboTwin asset bundle
  --download-checkpoints   fetch selected release weights (~12 GB per model)
  --skip-runtime-install   prepare source/assets only
  -h, --help               show this help

Without download flags the script clones exact source revisions and creates a
separate Python 3.10 RoboTwin runtime for each selected model. It never installs
RoboTwin into either model's LIBERO environment. Asset and checkpoint downloads
are explicit because a complete first setup is tens of gigabytes.

Environment overrides: ROBOTWIN_DIR, FASTWAM_DIR, FLEXPI_DIR.
EOF
}

model="both"
check_only=0
download_assets=0
download_checkpoints=0
install_runtime=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="${2:?--model requires a value}"; shift 2 ;;
    --check) check_only=1; shift ;;
    --download-assets) download_assets=1; shift ;;
    --download-checkpoints) download_checkpoints=1; shift ;;
    --skip-runtime-install) install_runtime=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
[[ "$model" == "fastwam" || "$model" == "flexpi" || "$model" == "both" ]] \
  || die "--model must be fastwam, flexpi, or both"
if [[ "$check_only" == "1" && \
  ( "$download_assets" == "1" || "$download_checkpoints" == "1" ) ]]; then
  die "--check cannot be combined with download flags"
fi

find_checkout() {
  local override_name="$1" directory_name="$2" override_value="" candidate
  override_value="${!override_name:-}"
  if [[ -n "$override_value" ]]; then
    printf '%s\n' "$override_value"
    return
  fi
  for candidate in "$PROJECT_DIR/../$directory_name" "$PROJECT_DIR/../../$directory_name"; do
    if git -C "$candidate" rev-parse --git-dir >/dev/null 2>&1; then
      (cd "$candidate" && pwd)
      return
    fi
  done
  return 1
}

default_sibling_path() {
  local directory_name="$1"
  if [[ "$(basename "$(dirname "$PROJECT_DIR")")" == "git-worktrees" ]]; then
    printf '%s\n' "$(cd "$PROJECT_DIR/../.." && pwd)/$directory_name"
  else
    printf '%s\n' "$(cd "$PROJECT_DIR/.." && pwd)/$directory_name"
  fi
}

ensure_checkout() {
  local path="$1" url="$2" revision="$3" label="$4"
  if [[ ! -d "$path/.git" ]]; then
    [[ "$check_only" == "0" ]] || return 1
    git clone --filter=blob:none "$url" "$path"
    git -C "$path" checkout --detach "$revision"
  fi
  local actual
  actual="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
  [[ "$actual" == "$revision" ]] \
    || die "$label revision is ${actual:-missing}; expected $revision (existing checkouts are never rewritten)"
  if [[ -n "$(git -C "$path" status --porcelain --untracked-files=no)" ]]; then
    die "$label has tracked local changes; refusing to mutate it"
  fi
  printf 'ok      %s source: %s @ %s\n' "$label" "$path" "$revision"
}

status=0
check_path() {
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
    return 0
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

ROBOTWIN_ROOT="$(find_checkout ROBOTWIN_DIR upstream-robotwin || true)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(default_sibling_path upstream-robotwin)}"
if ! ensure_checkout "$ROBOTWIN_ROOT" \
  https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_REVISION" "RoboTwin"; then
  echo "missing RoboTwin source: $ROBOTWIN_ROOT @ $ROBOTWIN_REVISION" >&2
  status=1
fi

FASTWAM_ROOT=""
FLEXPI_ROOT=""
if [[ "$model" == "fastwam" || "$model" == "both" ]]; then
  FASTWAM_ROOT="$(find_checkout FASTWAM_DIR upstream-fastwam || true)"
  FASTWAM_ROOT="${FASTWAM_ROOT:-$(default_sibling_path upstream-fastwam)}"
  if ! ensure_checkout "$FASTWAM_ROOT" https://github.com/yuantianyuan01/FastWAM.git \
    "$FASTWAM_REVISION" "Fast-WAM"; then
    echo "missing Fast-WAM source: $FASTWAM_ROOT @ $FASTWAM_REVISION" >&2
    status=1
  fi
fi
if [[ "$model" == "flexpi" || "$model" == "both" ]]; then
  FLEXPI_ROOT="$(find_checkout FLEXPI_DIR upstream-flexpi || true)"
  FLEXPI_ROOT="${FLEXPI_ROOT:-$(default_sibling_path upstream-flexpi)}"
  if ! ensure_checkout "$FLEXPI_ROOT" https://github.com/geyan21/flex-pi.git \
    "$FLEXPI_REVISION" "Flex-π"; then
    echo "missing Flex-π source: $FLEXPI_ROOT @ $FLEXPI_REVISION" >&2
    status=1
  fi
fi

if [[ "$check_only" == "1" ]]; then
  if [[ -n "$FASTWAM_ROOT" ]]; then
    check_path "Fast-WAM RoboTwin runtime" "$FASTWAM_ROOT/.venv-robotwin/bin/python"
    check_path "Fast-WAM RoboTwin task config" "$FASTWAM_ROOT/third_party/RoboTwin/task_config/demo_clean.yml"
    check_path "Fast-WAM RoboTwin assets" "$FASTWAM_ROOT/third_party/RoboTwin/assets/embodiments"
    fast_ckpt="$FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
    fast_stats="$FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
    check_path "Fast-WAM RoboTwin checkpoint" "$fast_ckpt"
    check_path "Fast-WAM RoboTwin statistics" "$fast_stats"
    verify_sha256 "Fast-WAM RoboTwin checkpoint" "$fast_ckpt" "$FASTWAM_CHECKPOINT_SHA256"
    verify_sha256 "Fast-WAM RoboTwin statistics" "$fast_stats" "$ROBOTWIN_STATS_SHA256"
  fi
  if [[ -n "$FLEXPI_ROOT" ]]; then
    check_path "Flex-π RoboTwin runtime" "$FLEXPI_ROOT/.venv-robotwin/bin/python"
    check_path "Flex-π RoboTwin task config" "$FLEXPI_ROOT/third_party/RoboTwin/task_config/demo_clean.yml"
    check_path "Flex-π RoboTwin assets" "$FLEXPI_ROOT/third_party/RoboTwin/assets/embodiments"
    flex_release="$FLEXPI_ROOT/runs/flexpi-robotwin"
    flex_ckpt="$flex_release/checkpoints/weights/step_048060.pt"
    check_path "Flex-π RoboTwin checkpoint" "$flex_ckpt"
    check_path "Flex-π RoboTwin config" "$flex_release/config.yaml"
    check_path "Flex-π RoboTwin statistics" "$flex_release/dataset_stats.json"
    verify_sha256 "Flex-π RoboTwin checkpoint" "$flex_ckpt" "$FLEXPI_CHECKPOINT_SHA256"
    verify_sha256 "Flex-π RoboTwin config" "$flex_release/config.yaml" "$FLEXPI_CONFIG_SHA256"
    verify_sha256 "Flex-π RoboTwin statistics" "$flex_release/dataset_stats.json" "$ROBOTWIN_STATS_SHA256"
    flex_vae="$FLEXPI_ROOT/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
    flex_t5="$FLEXPI_ROOT/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
    flex_tokenizer="$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"
    flex_tokenizer_model="$FLEXPI_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model"
    hf_cache_base="${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}/hub}"
    flex_dino="$hf_cache_base/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/model.safetensors"
    check_path "Flex-π Wan 2.2 VAE" "$flex_vae"
    check_path "Flex-π UMT5 encoder" "$flex_t5"
    check_path "Flex-π tokenizer" "$flex_tokenizer"
    check_path "Flex-π tokenizer model" "$flex_tokenizer_model"
    check_path "Flex-π DINOv3 encoder" "$flex_dino"
    verify_sha256 "Flex-π Wan 2.2 VAE" "$flex_vae" "$VAE_SHA256"
    verify_sha256 "Flex-π UMT5 encoder" "$flex_t5" "$T5_SHA256"
    verify_sha256 "Flex-π tokenizer" "$flex_tokenizer" "$TOKENIZER_JSON_SHA256"
    verify_sha256 "Flex-π tokenizer model" "$flex_tokenizer_model" "$TOKENIZER_MODEL_SHA256"
    verify_sha256 "Flex-π DINOv3 encoder" "$flex_dino" "$DINO_SHA256"
  fi
  exit "$status"
fi

[[ "$status" == "0" ]] || die "source prerequisites are not ready"

if [[ "$download_assets" == "1" && ! -d "$ROBOTWIN_ROOT/assets/embodiments" ]]; then
  (
    cd "$ROBOTWIN_ROOT/assets"
    uv run --python 3.10 --with huggingface-hub python _download.py
    unzip -q background_texture.zip
    unzip -q embodiments.zip
    unzip -q objects.zip
    rm background_texture.zip embodiments.zip objects.zip
  )
  (
    cd "$ROBOTWIN_ROOT"
    uv run --python 3.10 python script/update_embodiment_config_path.py
  )
fi

if [[ "$install_runtime" == "1" ]]; then
  if [[ ! -d "$ROBOTWIN_ROOT/envs/curobo/.git" ]]; then
    git clone https://github.com/NVlabs/curobo.git "$ROBOTWIN_ROOT/envs/curobo"
    git -C "$ROBOTWIN_ROOT/envs/curobo" checkout --detach "$CUROBO_REVISION"
  fi
  actual_curobo="$(git -C "$ROBOTWIN_ROOT/envs/curobo" rev-parse HEAD)"
  [[ "$actual_curobo" == "$CUROBO_REVISION" ]] \
    || die "Curobo revision is $actual_curobo; expected $CUROBO_REVISION"
fi

prepare_vendor() {
  local model_root="$1"
  local vendor="$model_root/third_party/RoboTwin"
  [[ -f "$vendor/README.vendor.md" ]] || die "RoboTwin vendor manifest missing: $vendor"
  grep -F "$ROBOTWIN_REVISION" "$vendor/README.vendor.md" >/dev/null \
    || die "RoboTwin vendor in $model_root does not declare $ROBOTWIN_REVISION"
  if [[ ! -d "$vendor/task_config" ]]; then
    cp -a "$ROBOTWIN_ROOT/task_config" "$vendor/task_config"
  fi
  if [[ -d "$ROBOTWIN_ROOT/assets/embodiments" && \
    ! -e "$vendor/assets" && ! -L "$vendor/assets" ]]; then
    ln -s "$ROBOTWIN_ROOT/assets" "$vendor/assets"
  fi
  if [[ -d "$ROBOTWIN_ROOT/envs/curobo" && \
    ! -e "$vendor/envs/curobo" && ! -L "$vendor/envs/curobo" ]]; then
    ln -s "$ROBOTWIN_ROOT/envs/curobo" "$vendor/envs/curobo"
  fi
}

install_model_runtime() {
  local model_root="$1"
  local python="$model_root/.venv-robotwin/bin/python"
  if [[ ! -x "$python" ]]; then
    uv venv --python 3.10 "$model_root/.venv-robotwin"
  fi
  UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
    uv pip install --python "$python" --index-strategy unsafe-best-match -e "$model_root"
  uv pip install --python "$python" \
    transforms3d==0.4.2 sapien==3.0.0b1 scipy==1.10.1 mplib==0.2.1 \
    gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 \
    pydantic zarr h5py 'pyglet<2' moviepy termcolor av matplotlib ffmpeg
  uv pip install --python "$python" --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@$PYTORCH3D_REVISION"
  uv pip install --python "$python" --no-build-isolation -e "$ROBOTWIN_ROOT/envs/curobo"

  "$python" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

site = Path(distribution("sapien").locate_file(""))
urdf_loader = site / "sapien" / "wrapper" / "urdf_loader.py"
text = urdf_loader.read_text(encoding="utf-8")
text = text.replace('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")')
text = text.replace('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"')
text = text.replace('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")')
urdf_loader.write_text(text, encoding="utf-8")

planner = site / "mplib" / "planner.py"
text = planner.read_text(encoding="utf-8")
text = text.replace(
    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
)
planner.write_text(text, encoding="utf-8")
PY
}

if [[ -n "$FASTWAM_ROOT" ]]; then
  prepare_vendor "$FASTWAM_ROOT"
  [[ "$install_runtime" == "0" ]] || install_model_runtime "$FASTWAM_ROOT"
  if [[ "$download_checkpoints" == "1" ]]; then
    release="$FASTWAM_ROOT/checkpoints/fastwam_release"
    mkdir -p "$release"
    uvx hf@latest download yuanty/fastwam \
      robotwin_uncond_3cam_384.pt robotwin_uncond_3cam_384_dataset_stats.json \
      --revision "$FASTWAM_HF_REVISION" --local-dir "$release"
  fi
fi

if [[ -n "$FLEXPI_ROOT" ]]; then
  prepare_vendor "$FLEXPI_ROOT"
  [[ "$install_runtime" == "0" ]] || install_model_runtime "$FLEXPI_ROOT"
  if [[ "$download_checkpoints" == "1" ]]; then
    release="$FLEXPI_ROOT/runs/flexpi-robotwin"
    mkdir -p "$release"
    uvx hf@latest download flex-pi/flexpi-robotwin \
      --revision "$FLEXPI_HF_REVISION" --local-dir "$release"
    flex_python="$FLEXPI_ROOT/.venv-robotwin/bin/python"
    [[ -x "$flex_python" ]] \
      || die "Flex-π ancillary model assets require the isolated runtime; omit --skip-runtime-install"
    export DIFFSYNTH_MODEL_BASE_PATH="$FLEXPI_ROOT/checkpoints"
    export MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-16}"
    mkdir -p "$DIFFSYNTH_MODEL_BASE_PATH"
    "$flex_python" "$PROJECT_DIR/scripts/download_flexpi_assets.py"
  fi
fi

echo
echo "RoboTwin source/runtime preparation complete. Large downloads remain opt-in."
echo "Readiness: ./scripts/setup_robotwin.sh --model $model --check"
echo "Simulator-only smoke: ./scripts/run_robotwin_smoke.sh --task click_bell"
example_model="$model"
[[ "$example_model" != "both" ]] || example_model="flexpi"
echo "Browser studio: ./lab --backend robotwin --model $example_model --mode interactive --task-id click_bell --default"
echo "Native batch: ./lab --backend robotwin --model $example_model --mode batch --task-id click_bell --default"
