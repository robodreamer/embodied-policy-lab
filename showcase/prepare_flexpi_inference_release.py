#!/usr/bin/env python3
"""Prepare a no-copy Flex-pi release view with inference-only config overrides."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import torch
import yaml


def inspect_complete_checkpoint(path: pathlib.Path) -> dict[str, int]:
    """Confirm the release checkpoint carries both experts before skipping bases."""
    with torch.device("meta"):
        payload = torch.load(
            path,
            map_location="meta",
            mmap=True,
            weights_only=True,
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
        raise TypeError(f"Flex-pi checkpoint has no complete MoT payload: {path}")
    keys = payload["mot"]
    counts = {
        "video": sum(key.startswith("mixtures.video.") for key in keys),
        "action": sum(key.startswith("mixtures.action.") for key in keys),
    }
    if counts["video"] < 800 or counts["action"] < 800:
        raise ValueError(
            "Refusing inference-only initialization because the checkpoint does "
            f"not contain complete video/action experts: {counts}"
        )
    return counts


def inference_config(source: pathlib.Path) -> dict[str, Any]:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise TypeError(f"Flex-pi release config has no model mapping: {source}")
    # The saved training config requests Wan/ActionDiT initialization before
    # loading the release checkpoint. The inference preset intentionally skips
    # both because the release checkpoint contains the complete experts.
    config["model"]["skip_dit_load_from_pretrain"] = True
    config["model"]["action_dit_pretrained_path"] = None
    return config


def atomic_text(path: pathlib.Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def prepare(
    checkpoint: pathlib.Path,
    config_path: pathlib.Path,
    destination: pathlib.Path,
) -> pathlib.Path:
    checkpoint = checkpoint.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    counts = inspect_complete_checkpoint(checkpoint)
    config = inference_config(config_path)
    weights = destination / "checkpoints" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint = weights / checkpoint.name
    if runtime_checkpoint.exists():
        if not os.path.samefile(checkpoint, runtime_checkpoint):
            raise FileExistsError(
                f"runtime checkpoint path already contains another file: {runtime_checkpoint}"
            )
    else:
        os.link(checkpoint, runtime_checkpoint)

    atomic_text(
        destination / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    atomic_text(
        destination / "inference-release.json",
        json.dumps(
            {
                "source_checkpoint": str(checkpoint),
                "source_config": str(config_path),
                "runtime_checkpoint": str(runtime_checkpoint),
                "checkpoint_expert_keys": counts,
                "skip_dit_load_from_pretrain": True,
                "action_dit_pretrained_path": None,
                "storage": "hard-link; no checkpoint bytes copied",
            },
            indent=2,
        ),
    )
    return runtime_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--destination", required=True, type=pathlib.Path)
    args = parser.parse_args()
    runtime_checkpoint = prepare(args.checkpoint, args.config, args.destination)
    print(
        "Prepared inference-only Flex-pi release view with complete checkpoint "
        "experts; no Wan/ActionDiT base preload is required.",
        file=sys.stderr,
    )
    print(runtime_checkpoint)


if __name__ == "__main__":
    main()
