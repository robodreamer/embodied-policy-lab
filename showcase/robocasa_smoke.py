"""Headless RoboCasa reset/render/step smoke test without a policy model."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import robocasa  # noqa: F401 - importing registers the Gym environments
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-set", default="atomic_seen")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--split", choices=("pretrain", "target"), default="target")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_set not in TASK_SET_REGISTRY:
        available = ", ".join(sorted(TASK_SET_REGISTRY))
        raise ValueError(
            f"Unknown task set {args.task_set!r}; choose one of: {available}"
        )

    tasks = list(TASK_SET_REGISTRY[args.task_set])
    if not 0 <= args.task_id < len(tasks):
        raise ValueError(f"Task ID must be in [0, {len(tasks)})")

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_name = tasks[args.task_id]
    started = time.perf_counter()
    env = gym.make(f"robocasa/{env_name}", split=args.split, seed=args.seed)
    frames: list[np.ndarray] = []
    try:
        observation, reset_info = env.reset()
        prompt = str(observation["annotation.human.task_description"])
        frames.append(np.ascontiguousarray(env.render()))
        info = dict(reset_info)
        for _ in range(args.steps):
            zero_action = convert_action(np.zeros(12, dtype=np.float32))
            observation, _, _, _, info = env.step(zero_action)
            frames.append(np.ascontiguousarray(env.render()))
    finally:
        env.close()

    video_path = output_dir / "robocasa-smoke.mp4"
    imageio.mimwrite(video_path, frames, fps=10)
    payload = {
        "backend": "robocasa",
        "simulator": "RoboCasa365 / robosuite / MuJoCo",
        "policy_used": False,
        "task_set": args.task_set,
        "task_id": args.task_id,
        "task_name": env_name,
        "split": args.split,
        "prompt": prompt,
        "task_horizon": get_task_horizon(env_name),
        "steps": args.steps,
        "success": bool(info.get("success", False)),
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.datetime.now(datetime.timezone.utc)
        .astimezone()
        .isoformat(),
        "video": str(video_path),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
