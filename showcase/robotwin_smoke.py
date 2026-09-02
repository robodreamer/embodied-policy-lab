#!/usr/bin/env python3
"""Model-free RoboTwin reset, observation, and 14D no-op action smoke test."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")
ACTION_DIMENSION = 14


def validate_observation(observation: dict[str, Any]) -> dict[str, list[int]]:
    """Validate the released Fast-WAM/Flex-π RoboTwin interface."""
    try:
        state = observation["joint_action"]["vector"]
        cameras = observation["observation"]
    except KeyError as error:
        raise ValueError(f"missing RoboTwin observation key: {error}") from error

    state_shape = list(state.shape)
    if state_shape != [ACTION_DIMENSION]:
        raise ValueError(f"expected 14D joint state, got shape {state_shape}")

    shapes: dict[str, list[int]] = {}
    for camera in CAMERA_KEYS:
        try:
            rgb = cameras[camera]["rgb"]
        except KeyError as error:
            raise ValueError(f"missing RoboTwin camera RGB: {camera} ({error})") from error
        shape = list(rgb.shape)
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"{camera} RGB must be HWC with three channels, got {shape}")
        shapes[camera] = shape
    return shapes


def load_task_arguments(robotwin_root: Path, task: str, phase: str) -> dict[str, Any]:
    import yaml

    with (robotwin_root / "task_config" / f"{phase}.yml").open(encoding="utf-8") as file:
        arguments = yaml.safe_load(file)
    with (robotwin_root / "task_config" / "_embodiment_config.yml").open(
        encoding="utf-8"
    ) as file:
        embodiment_types = yaml.safe_load(file)
    with (robotwin_root / "task_config" / "_camera_config.yml").open(
        encoding="utf-8"
    ) as file:
        camera_types = yaml.safe_load(file)

    arguments.update(
        task_name=task,
        task_config=phase,
        ckpt_setting="model-free-smoke",
        policy_name="model-free-smoke",
        eval_mode=True,
        collect_data=False,
        render_freq=0,
    )
    embodiment = arguments["embodiment"]
    if len(embodiment) != 1:
        raise ValueError(f"smoke path expects one shared bimanual embodiment, got {embodiment}")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    arguments["left_robot_file"] = robot_file
    arguments["right_robot_file"] = robot_file
    arguments["dual_arm_embodied"] = True
    head_type = arguments["camera"]["head_camera_type"]
    arguments["head_camera_h"] = camera_types[head_type]["h"]
    arguments["head_camera_w"] = camera_types[head_type]["w"]

    with (robotwin_root / robot_file / "config.yml").open(encoding="utf-8") as file:
        robot_config = yaml.safe_load(file)
    arguments["left_embodiment_config"] = robot_config
    arguments["right_embodiment_config"] = robot_config
    return arguments


def run_smoke(robotwin_root: Path, task: str, phase: str, seed: int) -> dict[str, Any]:
    import numpy as np

    original_cwd = Path.cwd()
    sys.path.insert(0, str(robotwin_root))
    os.chdir(robotwin_root)
    environment = None
    try:
        task_module = importlib.import_module(f"envs.{task}")
        task_class = getattr(task_module, task)
        arguments = load_task_arguments(robotwin_root, task, phase)
        environment = task_class()
        environment.setup_demo(now_ep_num=0, seed=seed, is_test=True, **arguments)

        before = environment.get_obs()
        before_shapes = validate_observation(before)
        state = np.asarray(before["joint_action"]["vector"], dtype=np.float32)
        environment.take_action(state.copy(), action_type="qpos")
        after = environment.get_obs()
        after_shapes = validate_observation(after)

        vendor_manifest = robotwin_root / "README.vendor.md"
        revision = "unknown"
        if vendor_manifest.is_file():
            for line in vendor_manifest.read_text(encoding="utf-8").splitlines():
                if line.startswith("- Upstream commit:"):
                    revision = line.split("`", maxsplit=2)[1]
                    break
        return {
            "schema_version": 1,
            "kind": "robotwin-model-free-smoke",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "robotwin_root": str(robotwin_root),
            "robotwin_revision": revision,
            "task": task,
            "phase": phase,
            "seed": seed,
            "embodiment": "aloha-agilex",
            "state_dimension": int(state.shape[0]),
            "action_dimension": int(state.shape[0]),
            "action_type": "qpos",
            "steps_executed": 1,
            "cameras_before": before_shapes,
            "cameras_after": after_shapes,
            "task_success_after_noop": bool(environment.check_success()),
        }
    finally:
        if environment is not None:
            environment.close_env(clear_cache=True)
        os.chdir(original_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", default="click_bell")
    parser.add_argument(
        "--phase", choices=("demo_clean", "demo_randomized"), default="demo_clean"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.robotwin_root.expanduser().resolve()
    result = run_smoke(root, args.task, args.phase, args.seed)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Model-free smoke evidence: {output}")


if __name__ == "__main__":
    main()
