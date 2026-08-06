"""Run one audited local GR00T request inside a RoboCasa environment."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import time

import imageio.v2 as imageio
import numpy as np

try:
    from . import robocasa_policy_plugins
    from . import robocasa_runtime as core
except ImportError:
    import robocasa_policy_plugins
    import robocasa_runtime as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-set", default="atomic_seen")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--split", choices=("pretrain", "target"), default="target")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--execute-actions", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = core.task_names(args.task_set)
    if not 0 <= args.task_id < len(tasks):
        raise ValueError(f"Task ID must be in [0, {len(tasks)})")
    if args.execute_actions < 1:
        raise ValueError("--execute-actions must be positive")

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_name = tasks[args.task_id]
    started_at = core.timestamp()
    started = time.perf_counter()
    policy = robocasa_policy_plugins.create_policy_plugin(
        "groot-n1.5", args.host, args.port
    )
    env = core.create_environment(env_name, args.split, args.seed)
    frames: list[np.ndarray] = []
    try:
        observation, _ = env.reset()
        prompt = str(observation[robocasa_policy_plugins.GR00T_LANGUAGE_KEY])
        frames.append(np.ascontiguousarray(env.render()))
        request = policy.prepare(observation, prompt)
        inference_started = time.perf_counter()
        actions = policy.infer(request)
        inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0

        executed = min(args.execute_actions, len(actions))
        success = False
        for action in actions[:executed]:
            observation, _, _, _, info = core.step_environment(env, action)
            frames.append(np.ascontiguousarray(env.render()))
            success = success or bool(info.get("success", False))
    finally:
        env.close()

    action_hash = hashlib.sha256(actions.tobytes()).hexdigest()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    video_path = output_dir / "robocasa-groot-n1.5-policy-smoke.mp4"
    imageio.mimwrite(video_path, frames, fps=10)
    audit = {
        "created_at": core.timestamp(),
        "backend": "robocasa",
        "model_plugin": "groot-n1.5",
        "policy_endpoint": policy.endpoint,
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
        "model_plugin": "groot-n1.5",
        "model": policy.profile.model_name,
        "runtime": policy.spec.runtime,
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "policy_endpoint": policy.endpoint,
        "task_set": args.task_set,
        "task_id": args.task_id,
        "task_name": env_name,
        "split": args.split,
        "prompt": prompt,
        "task_horizon": core.base_horizon(env_name),
        "state_dimension": int(len(request.robot_state)),
        "camera_count": len(robocasa_policy_plugins.GR00T_VIDEO_KEYS),
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
        "finished_at": core.timestamp(),
        "video": str(video_path),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
