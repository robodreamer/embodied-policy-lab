#!/usr/bin/env python3
"""Download Flex-pi's shared inference assets with file-level integrity pins."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


MODELSCOPE_ASSETS = {
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors": {
        "Wan2.2_VAE.safetensors": (
            "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"
        ),
        "models_t5_umt5-xxl-enc-bf16.safetensors": (
            "d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23"
        ),
    },
    "Wan-AI/Wan2.1-T2V-1.3B": {
        "google/umt5-xxl/special_tokens_map.json": (
            "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"
        ),
        "google/umt5-xxl/spiece.model": (
            "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"
        ),
        "google/umt5-xxl/tokenizer.json": (
            "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
        ),
        "google/umt5-xxl/tokenizer_config.json": (
            "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"
        ),
    },
}

DINO_REPOSITORY = "timm/vit_base_patch16_dinov3.lvd1689m"
DINO_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_remote_manifest(api, repository: str, expected: dict[str, str]) -> None:
    """Fail before a large download if mutable ModelScope master has changed."""
    files = api.get_model_files(repository, revision="master", recursive=True)
    remote = {
        item.get("Path"): item.get("Sha256")
        for item in files
        if item.get("Path") in expected
    }
    missing = sorted(set(expected) - set(remote))
    if missing:
        raise RuntimeError(
            f"ModelScope repository {repository} is missing pinned files: {missing}"
        )
    changed = {
        path: (expected[path], remote[path])
        for path in expected
        if remote[path] != expected[path]
    }
    if changed:
        raise RuntimeError(
            f"ModelScope repository {repository} changed pinned file hashes: {changed}"
        )


def validate_local_files(root: Path, expected: dict[str, str]) -> None:
    for relative_path, expected_sha256 in expected.items():
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"ModelScope download did not create {path}")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: {actual_sha256} != {expected_sha256}"
            )
        print(f"Verified {relative_path}: {actual_sha256}", flush=True)


def main() -> None:
    from huggingface_hub import snapshot_download as hf_snapshot_download
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    base_value = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
    if not base_value:
        raise RuntimeError("DIFFSYNTH_MODEL_BASE_PATH must name the checkpoint directory")
    base = Path(base_value).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    api = HubApi()
    for repository, expected in MODELSCOPE_ASSETS.items():
        # The publisher exposes only the ModelScope `master` branch. The
        # per-file Revision fields are commit metadata, not valid repo refs.
        validate_remote_manifest(api, repository, expected)
        print(f"Warming verified inference assets: {repository}", flush=True)
        destination = base / repository
        snapshot_download(
            repository,
            revision="master",
            local_dir=str(destination),
            allow_file_pattern=list(expected),
        )
        validate_local_files(destination, expected)

    hf_snapshot_download(DINO_REPOSITORY, revision=DINO_REVISION)
    print("Required VAE, T5/tokenizer, and DINOv3 assets are ready.", flush=True)


if __name__ == "__main__":
    main()
