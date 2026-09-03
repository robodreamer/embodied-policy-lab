"""Shared RoboCasa showcase primitives.

This module follows the observation and action contract used by the official
RoboCasa OpenPI evaluator while keeping dashboard/session bookkeeping in this
repository.
"""

from __future__ import annotations

import datetime
import gc
import json
import os
import pathlib
import random
import re

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401 - importing registers robocasa/* Gym environments
from PIL import Image
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from robosuite.utils.binding_utils import MjRenderContextOffscreen

try:
    from .robocasa_contracts import (
        CAMERA_KEYS,
        robot_state_from_observation,
        validate_action_chunk,
    )
except ImportError:  # Direct script execution adds showcase/ to sys.path.
    from robocasa_contracts import (
        CAMERA_KEYS,
        robot_state_from_observation,
        validate_action_chunk,
    )

__all__ = [
    "CAMERA_KEYS",
    "robot_state_from_observation",
    "validate_action_chunk",
]


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
    # RoboCasa samples fixtures, layouts, and textures during construction using
    # process-global NumPy and Python RNGs before Gym's reset seed takes effect.
    np.random.seed(seed)
    random.seed(seed)
    return gym.make(
        f"robocasa/{task_name}",
        split=split,
        seed=seed,
        disable_env_checker=True,
    )


def create_environment_pair(task_name: str, split: str, seed: int):
    """Construct matching live and counterfactual environments.

    RoboCasa chooses fixtures and objects while constructing an environment.
    Replaying the same NumPy RNG state is therefore required before a MuJoCo
    state can be copied safely between the two models.
    """
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    live_env = None
    try:
        np.random.seed(seed)
        random.seed(seed)
        live_env = create_environment(task_name, split, seed)
        live_numpy_state = np.random.get_state()
        live_python_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        branch_env = create_environment(task_name, split, seed)
        np.random.set_state(live_numpy_state)
        random.setstate(live_python_state)
        return live_env, branch_env
    except Exception:
        if live_env is not None:
            live_env.close()
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        raise


def reset_environment_pair(live_env, branch_env, seed: int):
    """Reset a paired environment with identical randomized scene choices."""
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        live_observation, live_info = live_env.reset(seed=seed)
        live_numpy_state = np.random.get_state()
        live_python_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        branch_env.reset(seed=seed)
        np.random.set_state(live_numpy_state)
        random.setstate(live_python_state)
    except Exception:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        raise

    live_state = live_env.unwrapped.env.sim.get_state()
    branch_state = branch_env.unwrapped.env.sim.get_state()
    if (
        np.asarray(live_state.qpos).shape != np.asarray(branch_state.qpos).shape
        or np.asarray(live_state.qvel).shape != np.asarray(branch_state.qvel).shape
    ):
        raise ValueError(
            "Paired RoboCasa resets produced different MuJoCo models; "
            "counterfactual comparison cannot run safely"
        )
    return live_observation, live_info


def suspend_environment_renderer(env) -> None:
    """Release a branch EGL context so it cannot contaminate live rendering."""
    sim = env.unwrapped.env.sim
    context = sim._render_context_offscreen
    if context is None:
        return
    sim._render_context_offscreen = None
    del context
    gc.collect()


def resume_environment_renderer(env) -> None:
    """Recreate an offscreen context immediately before branch prediction."""
    inner = env.unwrapped.env
    if inner.sim._render_context_offscreen is None:
        MjRenderContextOffscreen(
            inner.sim,
            device_id=inner.render_gpu_device_id,
        )


def restore_environment_renderer(env) -> None:
    """Rebuild the live context after another MuJoCo model used EGL.

    RoboSuite's per-simulation render contexts share one process-level EGL
    display. Creating or destroying the branch context can leave the existing
    live context with stale framebuffer resources, producing duplicate cameras
    or corrupted colors even though the correct context is made current.
    """
    suspend_environment_renderer(env)
    resume_environment_renderer(env)


def render_viewer_frames(
    env, width: int, height: int, *, observation: dict | None = None
) -> tuple[np.ndarray, ...]:
    """Render full-resolution, correctly oriented dashboard cameras.

    Results are copied immediately because MuJoCo reuses the offscreen buffer.
    Counterfactual render contexts are suspended outside their prediction
    window, so these live renders retain the correct camera and color state.
    """
    del observation  # Kept in the signature for branch/live call-site symmetry.
    if width < 1 or height < 1:
        raise ValueError("Viewer width and height must be positive")
    sim = env.unwrapped.env.sim
    return tuple(
        np.ascontiguousarray(
            np.array(
                sim.render(camera_name=camera_name, width=width, height=height),
                dtype=np.uint8,
                copy=True,
            )[::-1]
        )
        for camera_name in (
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        )
    )


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
