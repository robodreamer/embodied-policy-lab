"""Shared RoboCasa showcase primitives.

This module follows the observation and action contract used by the official
RoboCasa OpenPI evaluator while keeping dashboard/session bookkeeping in this
repository.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import re

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401 - importing registers robocasa/* Gym environments
from openpi_client import image_tools
from PIL import Image
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action


CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_eye_in_hand",
    "video.robot0_agentview_right",
)
VIEWER_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_eye_in_hand",
)


def timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


def humanize_task_name(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned[:120] or "task"


def task_names(task_set: str) -> list[str]:
    if task_set not in TASK_SET_REGISTRY or not TASK_SET_REGISTRY[task_set]:
        available = ", ".join(
            sorted(name for name, tasks in TASK_SET_REGISTRY.items() if tasks)
        )
        raise ValueError(
            f"Unknown or empty RoboCasa task set {task_set!r}; choose one of: {available}"
        )
    return list(TASK_SET_REGISTRY[task_set])


def task_catalog(task_set: str) -> list[dict]:
    return [
        {
            "id": task_id,
            "name": humanize_task_name(name),
            "task_name": name,
            "prompt": humanize_task_name(name),
            "prompt_ready": False,
        }
        for task_id, name in enumerate(task_names(task_set))
    ]


def parse_task_ids(raw: str, count: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(count))
    result = []
    for component in raw.split(","):
        task_id = int(component.strip())
        if task_id < 0 or task_id >= count:
            raise ValueError(f"Task ID {task_id} is outside [0, {count})")
        result.append(task_id)
    if not result:
        raise ValueError("At least one task ID is required")
    return result


def create_environment(task_name: str, split: str, seed: int):
    if split not in ("pretrain", "target"):
        raise ValueError("RoboCasa split must be 'pretrain' or 'target'")
    return gym.make(
        f"robocasa/{task_name}",
        split=split,
        seed=seed,
        disable_env_checker=True,
    )


def prepare_observation(observation: dict, resize_size: int, prompt: str):
    images = [
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.ascontiguousarray(observation[key]), resize_size, resize_size
            )
        )
        for key in CAMERA_KEYS
    ]
    robot_state = np.concatenate(
        (
            observation["state.end_effector_position_relative"],
            observation["state.end_effector_rotation_relative"],
            observation["state.base_position"],
            observation["state.base_rotation"],
            observation["state.gripper_qpos"],
        ),
        axis=0,
    )
    if robot_state.shape != (16,):
        raise ValueError(f"Expected a 16D RoboCasa state; got {robot_state.shape}")
    model_input = {
        "observation/image": images[0],
        "observation/wrist_image": images[1],
        "observation/right_image": images[2],
        "observation/state": robot_state,
        "prompt": str(prompt),
    }
    return images, robot_state, model_input


def render_viewer_frames(env, width: int, height: int) -> tuple[np.ndarray, ...]:
    """Render presentation-quality views without changing policy observations.

    RoboCasa's Gym wrapper intentionally supplies square 256px observations to
    the policy. Rendering the dashboard views directly from MuJoCo lets the UI
    use a sharper widescreen image while leaving that policy contract intact.
    """
    if width < 1 or height < 1:
        raise ValueError("Viewer width and height must be positive")
    robocasa_env = env.unwrapped.env
    return tuple(
        np.ascontiguousarray(
            robocasa_env.sim.render(
                camera_name=camera_name,
                width=width,
                height=height,
            )[::-1]
        )
        for camera_name in VIEWER_CAMERA_NAMES
    )


def validate_action_chunk(value) -> np.ndarray:
    actions = np.asarray(value)
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError(f"Expected a [horizon, 12] action chunk; got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned non-finite RoboCasa actions")
    return actions


def step_environment(env, action: np.ndarray):
    return env.step(convert_action(np.asarray(action)))


class LiveState:
    def __init__(self, session_dir: str, initial: dict):
        self.session_dir = pathlib.Path(session_dir)
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "state.json"
        self.data = dict(initial)
        self.update()

    def update(self, **values) -> None:
        self.data.update(values)
        self.data["updated_at"] = timestamp()
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def frames(self, external: np.ndarray, wrist: np.ndarray) -> None:
        for name, pixels in (("external.jpg", external), ("wrist.jpg", wrist)):
            destination = self.frames_dir / name
            temporary = self.frames_dir / f"{name}.tmp"
            Image.fromarray(pixels).save(
                temporary,
                format="JPEG",
                quality=92,
                subsampling=1,
            )
            os.replace(temporary, destination)


class ControlInbox:
    def __init__(self, session_dir: str):
        self.directory = pathlib.Path(session_dir) / "controls"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seen: set[str] = set()

    def read(self):
        for path in sorted(self.directory.glob("*.json")):
            if path.name in self.seen:
                continue
            try:
                command = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            self.seen.add(path.name)
            return command
        return None


def prompt_stats(history: list[dict]) -> dict:
    stats = {}
    for attempt in history:
        if not attempt.get("rate_eligible"):
            continue
        key = "task {} | {} | {} steps".format(
            attempt["task_id"], attempt["prompt"], attempt["max_steps"]
        )
        item = stats.setdefault(
            key,
            {"episodes": 0, "successes": 0, "max_steps": attempt["max_steps"]},
        )
        item["episodes"] += 1
        item["successes"] += int(attempt["status"] == "success")
        item["success_rate"] = round(item["successes"] / item["episodes"], 4)
    return stats


def base_horizon(task_name: str) -> int:
    return int(get_task_horizon(task_name))
