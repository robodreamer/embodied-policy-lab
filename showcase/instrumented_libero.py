"""Instrumented π0.5 LIBERO evaluator for the local showcase dashboard.

This intentionally mirrors the official openpi LIBERO evaluator while adding
live state, camera frames, action chunks, timing, and selectable task IDs. It
does not modify the upstream checkout.
"""

import collections
import dataclasses
import datetime
import json
import logging
import math
import os
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tqdm
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_ids: str = "0"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    video_out_path: str = "showcase-runs/videos"
    session_dir: str = "showcase-runs/current"
    seed: int = 7
    realtime_delay_ms: int = 35
    network_audit: bool = True


def _timestamp():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


class LiveState:
    def __init__(self, session_dir, initial):
        self.session_dir = pathlib.Path(session_dir)
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "state.json"
        self.data = dict(initial)
        self.update()

    def update(self, **values):
        self.data.update(values)
        self.data["updated_at"] = _timestamp()
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(str(temporary), str(self.path))

    def frames(self, external, wrist):
        for name, pixels in (("external.jpg", external), ("wrist.jpg", wrist)):
            destination = self.frames_dir / name
            temporary = self.frames_dir / (name + ".tmp")
            Image.fromarray(pixels).save(str(temporary), format="JPEG", quality=88)
            os.replace(str(temporary), str(destination))


def _parse_task_ids(raw, task_count):
    if raw.strip().lower() == "all":
        return list(range(task_count))
    result = []
    for component in raw.split(","):
        task_id = int(component.strip())
        if task_id < 0 or task_id >= task_count:
            raise ValueError(
                "Task ID {} is outside [0, {})".format(task_id, task_count)
            )
        result.append(task_id)
    return result


def _max_steps(task_suite_name):
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite_name]


def _get_libero_env(task, resolution, seed):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


def _prepare_observation(obs, resize_size, prompt):
    external = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    external = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(external, resize_size, resize_size)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )
    robot_state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    model_input = {
        "observation/image": external,
        "observation/wrist_image": wrist,
        "observation/state": robot_state,
        "prompt": str(prompt),
    }
    return external, wrist, robot_state, model_input


def evaluate(args):
    np.random.seed(args.seed)
    started_at = _timestamp()
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = _parse_task_ids(args.task_ids, suite.n_tasks)
    video_path = pathlib.Path(args.video_out_path)
    video_path.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    latencies = []
    total_episodes = 0
    total_successes = 0
    max_steps = _max_steps(args.task_suite_name)

    state = LiveState(
        args.session_dir,
        {
            "phase": "initializing",
            "model": "pi05_libero",
            "runtime": "local JAX/CUDA",
            "policy_endpoint": "ws://{}:{}".format(args.host, args.port),
            "network_audit": args.network_audit,
            "suite": args.task_suite_name,
            "task_ids": task_ids,
            "total_tasks": len(task_ids),
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "action_horizon": 10,
            "started_at": started_at,
            "successes": 0,
            "episodes": 0,
            "inference_latencies_ms": [],
        },
    )

    try:
        for task_position, task_id in enumerate(tqdm.tqdm(task_ids), start=1):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(
                task, LIBERO_ENV_RESOLUTION, args.seed
            )

            for episode_index in range(args.num_trials_per_task):
                env.reset()
                obs = env.set_init_state(initial_states[episode_index])
                action_plan = collections.deque()
                replay_images = []
                done = False
                step = 0
                state.update(
                    phase="running",
                    task_id=task_id,
                    task_position=task_position,
                    prompt=str(task_description),
                    episode=episode_index + 1,
                    step=0,
                    max_steps=max_steps,
                    last_action_chunk=[],
                    current_action=[],
                    inference_latency_ms=None,
                )

                logging.info(
                    "Task %d/%d: %s", task_position, len(task_ids), task_description
                )
                while step < max_steps + args.num_steps_wait:
                    if step < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        step += 1
                        continue

                    external, wrist, robot_state, model_input = _prepare_observation(
                        obs, args.resize_size, task_description
                    )
                    state.frames(external, wrist)
                    replay_images.append(external)

                    inference_latency = None
                    if not action_plan:
                        inference_started = time.perf_counter()
                        action_chunk = np.asarray(client.infer(model_input)["actions"])
                        inference_latency = (
                            time.perf_counter() - inference_started
                        ) * 1000.0
                        latencies.append(inference_latency)
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                "Policy returned fewer actions than replan_steps"
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])
                        state.update(
                            last_action_chunk=np.round(action_chunk, 5).tolist(),
                            inference_latency_ms=round(inference_latency, 2),
                            inference_latencies_ms=[
                                round(value, 2) for value in latencies[-120:]
                            ],
                        )

                    action = np.asarray(action_plan.popleft())
                    state.update(
                        step=step - args.num_steps_wait,
                        progress=min(
                            1.0, float(step - args.num_steps_wait) / max_steps
                        ),
                        current_action=np.round(action, 5).tolist(),
                        robot_state=np.round(robot_state, 5).tolist(),
                    )
                    obs, _, done, _ = env.step(action.tolist())
                    if args.realtime_delay_ms > 0:
                        time.sleep(args.realtime_delay_ms / 1000.0)
                    step += 1
                    if done:
                        final_external, final_wrist, _, _ = _prepare_observation(
                            obs, args.resize_size, task_description
                        )
                        state.frames(final_external, final_wrist)
                        replay_images.append(final_external)
                        total_successes += 1
                        break

                total_episodes += 1
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                output_video = video_path / "rollout_{}_{}.mp4".format(
                    task_segment, suffix
                )
                imageio.mimwrite(
                    output_video, [np.asarray(frame) for frame in replay_images], fps=10
                )
                state.update(
                    phase="task_complete",
                    task_success=bool(done),
                    successes=total_successes,
                    episodes=total_episodes,
                    success_rate=round(total_successes / total_episodes, 4),
                    last_video=str(output_video),
                    progress=1.0,
                )
                logging.info(
                    "Success: %s (%d/%d)", done, total_successes, total_episodes
                )
            env.close()

        finished_at = _timestamp()
        warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
        state.update(
            phase="complete",
            finished_at=finished_at,
            successes=total_successes,
            episodes=total_episodes,
            success_rate=round(total_successes / total_episodes, 4),
            cold_inference_latency_ms=round(latencies[0], 2) if latencies else None,
            mean_inference_latency_ms=round(float(np.mean(latencies)), 2)
            if latencies
            else None,
            warm_mean_inference_latency_ms=round(float(np.mean(warm_latencies)), 2)
            if warm_latencies
            else None,
            median_inference_latency_ms=round(float(np.median(warm_latencies)), 2)
            if warm_latencies
            else None,
            p95_inference_latency_ms=round(float(np.percentile(warm_latencies, 95)), 2)
            if warm_latencies
            else None,
        )
        logging.info("Total success rate: %.3f", total_successes / total_episodes)
    except Exception as error:
        state.update(
            phase="error",
            error="{}: {}".format(type(error).__name__, error),
            finished_at=_timestamp(),
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    evaluate(tyro.cli(Args))
