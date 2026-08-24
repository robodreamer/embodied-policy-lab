"""Instrumented LIBERO policy evaluator for the local showcase dashboard.

This intentionally mirrors the official openpi LIBERO evaluator while adding
live state, camera frames, action chunks, timing, and selectable task IDs. It
does not modify the upstream checkout.
"""

import collections
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import time

import imageio

# Flex-π's Python 3.10 / torch>=2.6 runtime needs its official LIBERO setup
# hook before importing the benchmark package. It pins the active checkout's
# data paths and permits the trusted NumPy-backed benchmark init-state files.
# Older OpenPI client environments do not install flexpi and keep their
# existing import behavior.
try:
    from flexpi.utils.libero_setup import prepare_libero
except ImportError:
    pass
else:
    prepare_libero()

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image
import tqdm
import tyro

from libero_policy_plugins import create_libero_policy_client


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    model: str = "pi05"
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_ids: str = "0"
    num_steps_wait: int = 10
    max_policy_steps: int = 0
    num_trials_per_task: int = 1
    video_out_path: str = "showcase-runs/videos"
    session_dir: str = "showcase-runs/current"
    seed: int = 7
    realtime_delay_ms: int = 35
    network_audit: bool = True
    flexpi_mode: str = "full-joint"
    benchmark_mode: bool = False
    save_videos: bool = True
    latency_probe_warmups: int = 0
    latency_probe_calls: int = 0


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


def _max_steps(task_suite_name, model="pi05"):
    if model in ("fastwam", "fast-wam", "fast_wam"):
        return {
            "libero_spatial": 400,
            "libero_object": 400,
            "libero_goal": 400,
            "libero_10": 700,
            "libero_90": 700,
        }[task_suite_name]
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite_name]


def _get_libero_env(task, resolution, seed, *, camera_depths=False):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": camera_depths,
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


def _depth_mm(obs, env):
    """Convert robosuite's normalized depth buffers to lossless millimetres."""

    from robosuite.utils.camera_utils import get_real_depth_map

    result = {}
    for output_name, observation_name in (
        ("external", "agentview_depth"),
        ("wrist", "robot0_eye_in_hand_depth"),
    ):
        if observation_name not in obs:
            raise ValueError(
                f"Depth-enabled LIBERO observation is missing {observation_name!r}"
            )
        normalized = np.clip(np.asarray(obs[observation_name]), 0.0, 1.0)
        metres = np.asarray(get_real_depth_map(env.sim, normalized))[..., 0]
        millimetres = np.clip(metres * 1000.0, 0, 65535).astype(np.uint16)
        result[output_name] = np.ascontiguousarray(millimetres[::-1, ::-1])
    return result


def _prepare_observation(obs, resize_size, prompt, client, env=None):
    del resize_size  # Each policy client owns its exact input preprocessing contract.
    external, wrist = _display_images(obs)
    robot_state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    depth = None
    if client.requires_depth:
        if env is None:
            raise ValueError(
                "A live LIBERO environment is required for depth conversion"
            )
        depth = _depth_mm(obs, env)
    model_input = client.prepare_observation(
        external, wrist, robot_state, str(prompt), depth=depth
    )
    return external, wrist, robot_state, model_input


def _display_images(obs):
    """Return display/train-aligned raw RGB without policy-specific resizing."""

    external = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return np.asarray(external, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)


def _latency_probe(client, model_input, warmups, calls):
    """Measure repeated warm calls on one observation without stepping the robot."""

    wall_ms = []
    server_total_ms = []
    denoise_core_ms = []
    peak_allocated = []
    peak_reserved = []
    for call_index in range(warmups + calls):
        started = time.perf_counter()
        response = client.infer(model_input)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if call_index < warmups:
            continue
        timing = response.get("timing") or {}
        wall_ms.append(elapsed_ms)
        if timing.get("total_seconds") is not None:
            server_total_ms.append(float(timing["total_seconds"]) * 1000.0)
        core_seconds = timing.get(
            "inference_seconds", timing.get("action_inference_seconds")
        )
        if core_seconds is not None:
            denoise_core_ms.append(float(core_seconds) * 1000.0)
        peak_allocated.append(int(timing.get("cuda_peak_allocated_bytes") or 0))
        peak_reserved.append(int(timing.get("cuda_peak_reserved_bytes") or 0))

    def stats(values):
        if not values:
            return None
        return {
            "mean_ms": round(float(np.mean(values)), 3),
            "median_ms": round(float(np.median(values)), 3),
            "p95_ms": round(float(np.percentile(values, 95)), 3),
        }

    return {
        "warmup_calls": warmups,
        "timed_calls": calls,
        "observation_reused": True,
        "batch_size": 1,
        "wall_round_trip": stats(wall_ms),
        "server_total": stats(server_total_ms),
        "denoise_core": stats(denoise_core_ms),
        "max_cuda_allocated_bytes": max(peak_allocated, default=0),
        "max_cuda_reserved_bytes": max(peak_reserved, default=0),
    }


def evaluate(args):
    if args.latency_probe_warmups < 0 or args.latency_probe_calls < 0:
        raise ValueError("Latency probe counts must be non-negative")
    np.random.seed(args.seed)
    try:
        import mujoco

        mujoco_version = mujoco.__version__
    except (ImportError, AttributeError):
        mujoco_version = "unknown"
    started_at = _timestamp()
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = _parse_task_ids(args.task_ids, suite.n_tasks)
    video_path = pathlib.Path(args.video_out_path)
    video_path.mkdir(parents=True, exist_ok=True)
    client = create_libero_policy_client(
        args.model, args.host, args.port, flexpi_mode=args.flexpi_mode
    )
    inference_audit_path = pathlib.Path(args.session_dir) / "inference-audit.jsonl"
    latencies = []
    model_request_count = 0
    total_episodes = 0
    total_successes = 0
    latency_probe_result = None
    max_steps = (
        args.max_policy_steps
        if args.max_policy_steps > 0
        else _max_steps(args.task_suite_name, args.model)
    )

    state = LiveState(
        args.session_dir,
        {
            "phase": "initializing",
            "backend": "libero",
            "simulator": "LIBERO / robosuite / MuJoCo",
            "mujoco_version": mujoco_version,
            "libero_source": str(pathlib.Path(benchmark.__file__).resolve()),
            "model_plugin": client.spec.key,
            "model": client.profile.model_name,
            "model_display_name": client.spec.display_name,
            "runtime": client.spec.runtime,
            "policy_transport": client.spec.transport,
            "policy_endpoint": client.endpoint,
            "network_audit": args.network_audit,
            "benchmark_mode": args.benchmark_mode,
            "save_videos": args.save_videos,
            "suite": args.task_suite_name,
            "task_ids": task_ids,
            "total_tasks": len(task_ids),
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "action_horizon": client.profile.action_horizon,
            "action_dimension": 7,
            "policy_mode": client.mode,
            "available_policy_modes": list(client.available_modes),
            "model_image_width": client.model_image_width,
            "model_image_height": client.model_image_height,
            "camera_observation_width": LIBERO_ENV_RESOLUTION,
            "camera_observation_height": LIBERO_ENV_RESOLUTION,
            "action_labels": [
                "EEF ΔX",
                "EEF ΔY",
                "EEF ΔZ",
                "EEF ROT X",
                "EEF ROT Y",
                "EEF ROT Z",
                "GRIPPER",
            ],
            "state_dimension": 8,
            "started_at": started_at,
            "successes": 0,
            "episodes": 0,
            "inference_latencies_ms": [],
            "model_request_count": 0,
        },
    )

    try:
        for task_position, task_id in enumerate(tqdm.tqdm(task_ids), start=1):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(
                task,
                LIBERO_ENV_RESOLUTION,
                args.seed,
                camera_depths=client.requires_depth,
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
                        obs, args.resize_size, task_description, client, env
                    )
                    if not args.benchmark_mode:
                        state.frames(external, wrist)
                    if args.save_videos:
                        replay_images.append(external)

                    if latency_probe_result is None and args.latency_probe_calls:
                        latency_probe_result = _latency_probe(
                            client,
                            model_input,
                            args.latency_probe_warmups,
                            args.latency_probe_calls,
                        )
                        latency_probe_path = (
                            pathlib.Path(args.session_dir) / "latency-probe.json"
                        )
                        latency_probe_path.write_text(
                            json.dumps(latency_probe_result, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        state.update(latency_probe=latency_probe_result)

                    inference_latency = None
                    if not action_plan:
                        inference_started = time.perf_counter()
                        policy_response = client.infer(model_input)
                        action_chunk = np.asarray(policy_response["actions"])
                        inference_latency = (
                            time.perf_counter() - inference_started
                        ) * 1000.0
                        latencies.append(inference_latency)
                        model_request_count += 1
                        prompt = str(model_input["prompt"])
                        audit_event = {
                            "created_at": _timestamp(),
                            "request_id": model_request_count,
                            "task_id": task_id,
                            "episode": episode_index + 1,
                            "step": max(0, step - args.num_steps_wait),
                            "prompt": prompt,
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode("utf-8")
                            ).hexdigest(),
                            "action_chunk_sha256": hashlib.sha256(
                                action_chunk.tobytes()
                            ).hexdigest(),
                            "inference_latency_ms": round(inference_latency, 2),
                            "policy_timing": policy_response.get("timing"),
                            "policy_artifact": policy_response.get("artifact"),
                            "policy_mode": client.mode,
                            "policy_prediction": policy_response.get("prediction"),
                            "max_steps": max_steps,
                        }
                        with inference_audit_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(audit_event) + "\n")
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
                            model_request_count=model_request_count,
                            last_inference_audit=audit_event,
                        )

                    action = np.asarray(action_plan.popleft())
                    if not args.benchmark_mode:
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
                        if not args.benchmark_mode or args.save_videos:
                            final_external, final_wrist, _, _ = _prepare_observation(
                                obs, args.resize_size, task_description, client, env
                            )
                            if not args.benchmark_mode:
                                state.frames(final_external, final_wrist)
                            if args.save_videos:
                                replay_images.append(final_external)
                        total_successes += 1
                        break

                total_episodes += 1
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                output_video = None
                if args.save_videos and replay_images:
                    output_video = video_path / "rollout_{}_{}.mp4".format(
                        task_segment, suffix
                    )
                    imageio.mimwrite(
                        output_video,
                        [np.asarray(frame) for frame in replay_images],
                        fps=10,
                    )
                state.update(
                    phase="task_complete",
                    task_success=bool(done),
                    successes=total_successes,
                    episodes=total_episodes,
                    success_rate=round(total_successes / total_episodes, 4),
                    last_video=str(output_video) if output_video else None,
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
            latency_probe=latency_probe_result,
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
