"""Run one audited local π0.5 request inside a RoboCasa environment."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import robocasa  # noqa: F401 - importing registers the Gym environments
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-set", default="atomic_seen")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--split", choices=("pretrain", "target"), default="target")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--execute-actions", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def prepare_observation(observation: dict, resize_size: int, prompt: str) -> tuple:
    camera_keys = (
        "video.robot0_agentview_left",
        "video.robot0_eye_in_hand",
        "video.robot0_agentview_right",
    )
    images = [
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.ascontiguousarray(observation[key]), resize_size, resize_size
            )
        )
        for key in camera_keys
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
    model_input = {
        "observation/image": images[0],
        "observation/wrist_image": images[1],
        "observation/right_image": images[2],
        "observation/state": robot_state,
        "prompt": str(prompt),
    }
    return images, robot_state, model_input


def main() -> None:
    args = parse_args()
    tasks = list(TASK_SET_REGISTRY[args.task_set])
    if not 0 <= args.task_id < len(tasks):
        raise ValueError(f"Task ID must be in [0, {len(tasks)})")
    if args.execute_actions < 1:
        raise ValueError("--execute-actions must be positive")

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_name = tasks[args.task_id]
    started_at = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
    started = time.perf_counter()
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = gym.make(f"robocasa/{env_name}", split=args.split, seed=args.seed)
    frames: list[np.ndarray] = []
    try:
        observation, _ = env.reset()
        prompt = str(observation["annotation.human.task_description"])
        frames.append(np.ascontiguousarray(env.render()))
        images, robot_state, model_input = prepare_observation(
            observation, args.resize_size, prompt
        )
        inference_started = time.perf_counter()
        response = client.infer(model_input)
        inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
        actions = np.asarray(response["actions"])
        if actions.ndim != 2 or actions.shape[1] != 12:
            raise ValueError(
                f"Expected a [horizon, 12] action chunk; got {actions.shape}"
            )

        executed = min(args.execute_actions, len(actions))
        success = False
        for action in actions[:executed]:
            observation, _, _, _, info = env.step(convert_action(action))
            frames.append(np.ascontiguousarray(env.render()))
            success = success or bool(info.get("success", False))
    finally:
        env.close()

    action_hash = hashlib.sha256(actions.tobytes()).hexdigest()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    video_path = output_dir / "robocasa-pi05-policy-smoke.mp4"
    imageio.mimwrite(video_path, frames, fps=10)
    audit = {
        "created_at": datetime.datetime.now(datetime.timezone.utc)
        .astimezone()
        .isoformat(),
        "backend": "robocasa",
        "policy_endpoint": f"ws://{args.host}:{args.port}",
        "prompt": prompt,
        "prompt_sha256": prompt_hash,
        "action_chunk_shape": list(actions.shape),
        "action_chunk_sha256": action_hash,
        "inference_latency_ms": round(inference_latency_ms, 2),
    }
    (output_dir / "inference-audit.jsonl").write_text(
        json.dumps(audit) + "\n", encoding="utf-8"
    )
    result = {
        "backend": "robocasa",
        "simulator": "RoboCasa365 / robosuite / MuJoCo",
        "policy_used": True,
        "model": "pi05_pretrain_human300",
        "runtime": "local JAX/CUDA",
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "policy_endpoint": f"ws://{args.host}:{args.port}",
        "task_set": args.task_set,
        "task_id": args.task_id,
        "task_name": env_name,
        "split": args.split,
        "prompt": prompt,
        "task_horizon": get_task_horizon(env_name),
        "state_dimension": int(len(robot_state)),
        "camera_count": len(images),
        "action_dimension": int(actions.shape[1]),
        "predicted_action_horizon": int(len(actions)),
        "executed_actions": executed,
        "success_after_smoke_actions": success,
        "inference_latency_ms": round(inference_latency_ms, 2),
        "prompt_sha256": prompt_hash,
        "action_chunk_sha256": action_hash,
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "started_at": started_at,
        "finished_at": datetime.datetime.now(datetime.timezone.utc)
        .astimezone()
        .isoformat(),
        "video": str(video_path),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
