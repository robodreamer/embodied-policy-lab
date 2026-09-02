#!/usr/bin/env python3
"""Interactive RoboTwin 2.0 session for the browser studio.

This adapter deliberately runs the released policy wrapper and SAPIEN task in
one process. RoboTwin observations include three RGB/depth cameras and the
policy executes absolute 14D qpos actions; keeping that boundary in-process
avoids translating it into the unrelated LIBERO HTTP schema.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml
from PIL import Image


CAMERA_FILES = {
    "head_camera": "external.jpg",
    "left_camera": "wrist.jpg",
    "right_camera": "right_wrist.jpg",
}
ACTION_LABELS = [
    "LEFT J1",
    "LEFT J2",
    "LEFT J3",
    "LEFT J4",
    "LEFT J5",
    "LEFT J6",
    "LEFT GRIPPER",
    "RIGHT J1",
    "RIGHT J2",
    "RIGHT J3",
    "RIGHT J4",
    "RIGHT J5",
    "RIGHT J6",
    "RIGHT GRIPPER",
]


def timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


def humanize_task(task: str) -> str:
    return task.replace("_", " ")


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class LiveState:
    def __init__(self, session_dir: pathlib.Path, initial: dict[str, Any]):
        self.path = session_dir / "state.json"
        self.data = dict(initial)
        self.update()

    def update(self, **values: Any) -> None:
        self.data.update(values)
        self.data["updated_at"] = timestamp()
        atomic_json(self.path, self.data)


class ControlInbox:
    def __init__(self, session_dir: pathlib.Path):
        self.directory = session_dir / "controls"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seen: set[str] = set()

    def read(self) -> dict[str, Any] | None:
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


class VideoWriter:
    """Stream one camera to ffmpeg without retaining a rollout in memory."""

    def __init__(self, path: pathlib.Path, frame: np.ndarray, fps: int = 10):
        height, width = frame.shape[:2]
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-an",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed for {self.path} ({return_code})")


def load_module(path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def task_catalog(robotwin_root: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    limits_path = robotwin_root / "task_config" / "_eval_step_limit.yml"
    limits = yaml.safe_load(limits_path.read_text(encoding="utf-8"))
    tasks = [
        {
            "id": task,
            "name": humanize_task(task),
            "prompt": humanize_task(task),
            "position": index + 1,
            "max_steps": int(limit),
        }
        for index, (task, limit) in enumerate(limits.items())
    ]
    return tasks, {str(task): int(limit) for task, limit in limits.items()}


def write_frames(session_dir: pathlib.Path, observation: dict[str, Any]) -> None:
    frame_dir = session_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cameras = observation["observation"]
    for camera, filename in CAMERA_FILES.items():
        frame = np.asarray(cameras[camera]["rgb"], dtype=np.uint8)
        destination = frame_dir / filename
        temporary = destination.with_suffix(".tmp.jpg")
        Image.fromarray(frame, mode="RGB").save(temporary, quality=88)
        os.replace(temporary, destination)


def camera_frames(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    cameras = observation["observation"]
    return {
        camera: np.asarray(cameras[camera]["rgb"], dtype=np.uint8)
        for camera in CAMERA_FILES
    }


def load_policy(args: argparse.Namespace, state: LiveState):
    policy_name = "fastwam_policy" if args.model == "fastwam" else "flexpi_policy"
    deploy = args.model_root / "experiments" / "robotwin" / policy_name / "deploy_policy.py"
    state.update(
        command_message=(
            f"Loading {args.model_display_name} checkpoint in the isolated RoboTwin runtime"
        )
    )
    module = load_module(deploy, f"embodied_lab_{policy_name}")
    joint = args.flexpi_mode == "full-joint"
    policy_args: dict[str, Any] = {
        "sim_cfg_path": str(args.model_root / "configs" / "sim_robotwin.yaml"),
        "sim_task": args.task_profile,
        "ckpt_setting": str(args.checkpoint),
        "dataset_stats_path": str(args.dataset_stats),
        "device": "cuda",
        "mixed_precision": "bf16",
        "action_horizon": 32,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "timing_enabled": True,
    }
    if args.model == "flexpi":
        policy_args.update(
            offload_text_encoder=True,
            infer_joint_video=joint,
            infer_joint_dino=joint,
            infer_joint_pointmap=joint,
        )
    started = time.perf_counter()
    policy = module.get_model(policy_args)
    state.update(cold_inference_latency_ms=round((time.perf_counter() - started) * 1000, 2))
    return module, policy


def set_flexpi_mode(policy: Any, mode: str) -> None:
    if mode not in {"full-joint", "action-only"}:
        raise ValueError("Flex-π mode must be full-joint or action-only")
    joint = mode == "full-joint"
    for attribute in ("infer_joint_video", "infer_joint_dino", "infer_joint_pointmap"):
        if hasattr(policy, attribute):
            setattr(policy, attribute, joint)


def prepare_task(
    robotwin_root: pathlib.Path,
    task: str,
    phase: str,
    seed: int,
    state: LiveState,
):
    """Reproduce the upstream expert-seed check and unseen instruction path."""
    from robotwin_smoke import load_task_arguments

    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "description" / "utils"))
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    task_module = importlib.import_module(f"envs.{task}")
    task_class = getattr(task_module, task)
    native_seed = 100000 * (1 + seed)
    arguments = load_task_arguments(robotwin_root, task, phase)
    arguments.update(
        policy_name="embodied-policy-lab",
        ckpt_setting="browser-studio",
        eval_video_log=False,
        eval_mode=True,
        render_freq=0,
    )

    for offset in range(25):
        candidate_seed = native_seed + offset
        environment = task_class()
        state.update(
            command_message=(
                f"Validating RoboTwin seed {candidate_seed} with the upstream expert planner"
            )
        )
        try:
            environment.setup_demo(
                now_ep_num=0, seed=candidate_seed, is_test=True, **arguments
            )
            episode_info = environment.play_once()
            valid = bool(environment.plan_success and environment.check_success())
            environment.close_env()
            if not valid:
                continue
            environment.setup_demo(
                now_ep_num=0, seed=candidate_seed, is_test=True, **arguments
            )
            descriptions = generate_episode_descriptions(
                task, [episode_info["info"]], 1
            )
            unseen = descriptions[0].get("unseen") or descriptions[0].get("seen")
            if not unseen:
                raise ValueError(f"RoboTwin generated no instruction for {task}")
            instruction = str(np.random.default_rng(candidate_seed).choice(unseen))
            environment.set_instruction(instruction=instruction)
            return environment, instruction, candidate_seed
        except UnStableError:
            environment.close_env()
            continue
        except Exception:
            environment.close_env()
            raise
    raise RuntimeError(f"No expert-valid seed found for {task} in 25 attempts")


def prompt_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for item in history:
        if not item.get("rate_eligible"):
            continue
        key = f"{item['task_id']} | {item['prompt']} | {item['max_steps']} steps"
        value = stats.setdefault(key, {"episodes": 0, "successes": 0})
        value["episodes"] += 1
        value["successes"] += int(item["status"] == "success")
        value["success_rate"] = round(value["successes"] / value["episodes"], 4)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fastwam", "flexpi"), required=True)
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    parser.add_argument("--robotwin-root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-stats", type=pathlib.Path, required=True)
    parser.add_argument("--task-profile", required=True)
    parser.add_argument("--task", default="click_bell")
    parser.add_argument("--phase", choices=("demo_clean", "demo_randomized"), default="demo_clean")
    parser.add_argument("--session-dir", type=pathlib.Path, required=True)
    parser.add_argument("--replan-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime-delay-ms", type=int, default=35)
    parser.add_argument("--flexpi-mode", choices=("full-joint", "action-only"), default="full-joint")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--network-audit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    args.model_root = args.model_root.resolve()
    args.robotwin_root = args.robotwin_root.resolve()
    args.session_dir = args.session_dir.resolve()
    args.model_display_name = "Fast-WAM" if args.model == "fastwam" else "Flex-π"
    os.chdir(args.robotwin_root)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    tasks, step_limits = task_catalog(args.robotwin_root)
    task_names = list(step_limits)
    if args.task not in step_limits:
        raise ValueError(f"Unknown RoboTwin task: {args.task}")

    state = LiveState(
        args.session_dir,
        {
            "phase": "initializing",
            "interactive": True,
            "backend": "robotwin",
            "simulator": "RoboTwin 2.0 / SAPIEN / Vulkan",
            "model_plugin": args.model,
            "model": (
                "robotwin_uncond_3cam_384"
                if args.model == "fastwam"
                else "flexpi_robotwin_3cam_384"
            ),
            "model_display_name": args.model_display_name,
            "runtime": "local PyTorch/CUDA · native RoboTwin adapter",
            "policy_transport": "in-process native",
            "policy_endpoint": f"in-process://robotwin/{args.model}",
            "network_audit": args.network_audit,
            "network_verdict": "pending" if args.network_audit else "not_audited",
            "suite": args.phase,
            "task_ids": [args.task],
            "task_id": args.task,
            "task_position": task_names.index(args.task) + 1,
            "available_tasks": tasks,
            "total_tasks": len(tasks),
            "dynamic_task_prompts": False,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "action_horizon": 32,
            "action_dimension": 14,
            "action_labels": ACTION_LABELS,
            "state_dimension": 14,
            "camera_count": 3,
            "camera_observation_width": 320,
            "camera_observation_height": 240,
            "model_image_width": 320,
            "model_image_height": 384,
            "viewer_width": 320,
            "viewer_height": 240,
            "external_camera_label": "HEAD CAMERA",
            "wrist_camera_label": "LEFT WRIST CAMERA",
            "third_camera_label": "RIGHT WRIST CAMERA",
            "policy_mode": args.flexpi_mode if args.model == "flexpi" else "action-only",
            "available_policy_modes": (
                [
                    {
                        "key": "full-joint",
                        "display_name": "World-action co-generation",
                        "description": (
                            "Enable released joint denoising. This adapter charts the "
                            "14D actions but does not yet decode or retain future media."
                        ),
                    },
                    {
                        "key": "action-only",
                        "display_name": "Action only",
                        "description": "Disable Flex-π joint future streams for lower compute.",
                    },
                ]
                if args.model == "flexpi"
                else []
            ),
            "available_world_models": [],
            "world_model": "none",
            "world_model_display_name": "No predictor",
            "compare_world_model": False,
            "canonical_prompt": humanize_task(args.task),
            "prompt": humanize_task(args.task),
            "prompt_source": "canonical",
            "base_max_steps": step_limits[args.task],
            "max_steps": step_limits[args.task],
            "rollout_budget_multiplier": 1,
            "evaluation_mode": "scored",
            "episodes": 0,
            "successes": 0,
            "completed_attempts": 0,
            "unscored_attempts": 0,
            "aborted_attempts": 0,
            "attempt_history": [],
            "prompt_stats": {},
            "inference_latencies_ms": [],
            "model_request_count": 0,
            "started_at": timestamp(),
            "command_message": "Starting the native RoboTwin policy adapter",
        },
    )

    module, policy = load_policy(args, state)
    inbox = ControlInbox(args.session_dir)
    history: list[dict[str, Any]] = []
    latencies: list[float] = []
    successes = 0
    scored_episodes = 0
    attempt_number = 0
    current_task = args.task
    prompt_override: str | None = None
    prompt_source = "canonical"
    evaluation_mode = "scored"
    budget_multiplier = 1
    should_stop = False
    model_request_count = 0

    original_infer = policy._infer_action_chunk

    def observed_infer(observation, instruction):
        nonlocal model_request_count
        request_id = model_request_count + 1
        state.update(
            model_request_id=request_id,
            model_request_status="waiting_for_response",
            command_message=f"Running {args.model_display_name} inference request #{request_id}",
        )
        started = time.perf_counter()
        chunk = original_infer(observation, instruction)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        model_request_count = request_id
        latencies.append(elapsed)
        prompt_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        action_array = np.asarray(chunk, dtype=np.float32)
        inference_event = {
            "created_at": timestamp(),
            "request_id": request_id,
            "backend": "robotwin",
            "model": args.model,
            "policy_mode": args.flexpi_mode if args.model == "flexpi" else "action-only",
            "prompt": instruction,
            "prompt_source": prompt_source,
            "prompt_sha256": prompt_hash,
            "action_shape": list(action_array.shape),
            "action_sha256": hashlib.sha256(action_array.tobytes()).hexdigest(),
            "latency_ms": elapsed,
        }
        with (args.session_dir / "inference-audit.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(inference_event) + "\n")
        warm = latencies[1:] if len(latencies) > 1 else latencies
        state.update(
            model_request_count=model_request_count,
            model_request_status="complete",
            model_ack_prompt=instruction,
            model_ack_prompt_source=prompt_source,
            model_ack_prompt_sha256=prompt_hash,
            inference_latency_ms=elapsed,
            inference_latencies_ms=latencies,
            median_inference_latency_ms=round(float(np.median(warm)), 2),
            p95_inference_latency_ms=round(float(np.percentile(warm, 95)), 2),
            last_action_chunk=action_array.tolist(),
        )
        return chunk

    policy._infer_action_chunk = observed_infer
    state.update(
        phase="awaiting_command",
        command_message="Model ready. Choose one of 50 RoboTwin tasks, then start a rollout.",
    )

    pending: dict[str, Any] | None = (
        {"id": "auto-start", "action": "start_rollout"} if args.auto_start else None
    )
    while not should_stop:
        command = pending or inbox.read()
        pending = None
        if not command:
            time.sleep(0.1)
            continue
        action = str(command.get("action", ""))
        if action == "stop":
            state.update(command_ack=command.get("id"), command_message="Finishing session")
            break
        if action == "set_task":
            requested = str(command.get("task_id", current_task))
            if requested not in step_limits:
                state.update(control_error=f"Unknown RoboTwin task: {requested}")
                continue
            current_task = requested
            prompt_override = None
            prompt_source = "canonical"
            state.update(
                task_id=current_task,
                task_position=task_names.index(current_task) + 1,
                canonical_prompt=humanize_task(current_task),
                prompt=humanize_task(current_task),
                prompt_source=prompt_source,
                base_max_steps=step_limits[current_task],
                max_steps=step_limits[current_task] * budget_multiplier,
                command_ack=command.get("id"),
                command_message="RoboTwin task staged for the next rollout",
            )
            continue
        if action == "set_policy_mode" and args.model == "flexpi":
            try:
                args.flexpi_mode = str(command.get("policy_mode", args.flexpi_mode))
                set_flexpi_mode(policy, args.flexpi_mode)
            except ValueError as error:
                state.update(control_error=str(error))
                continue
            state.update(
                policy_mode=args.flexpi_mode,
                command_ack=command.get("id"),
                command_message="Flex-π inference mode staged for the next rollout",
            )
            continue
        if action not in {"start", "reset", "start_rollout"}:
            state.update(command_ack=command.get("id"), control_error=f"Unsupported action: {action}")
            continue

        requested_task = str(command.get("task_id", current_task))
        if requested_task not in step_limits:
            state.update(control_error=f"Unknown RoboTwin task: {requested_task}")
            continue
        current_task = requested_task
        budget_multiplier = int(command.get("rollout_budget_multiplier", budget_multiplier))
        if budget_multiplier not in {1, 2, 3}:
            state.update(control_error="Rollout budget multiplier must be 1, 2, or 3")
            continue
        evaluation_mode = str(command.get("evaluation_mode", evaluation_mode))
        requested_source = str(command.get("source", "canonical"))
        requested_prompt = str(command.get("prompt", "")).strip()
        prompt_override = requested_prompt if requested_source != "canonical" else None
        prompt_source = requested_source if prompt_override else "canonical"
        if args.model == "flexpi" and command.get("policy_mode"):
            args.flexpi_mode = str(command["policy_mode"])
            set_flexpi_mode(policy, args.flexpi_mode)

        attempt_number += 1
        state.update(
            phase="preparing_task",
            task_id=current_task,
            task_position=task_names.index(current_task) + 1,
            command_ack=command.get("id"),
            control_error=None,
            attempt=attempt_number,
            base_max_steps=step_limits[current_task],
            max_steps=step_limits[current_task] * budget_multiplier,
            rollout_budget_multiplier=budget_multiplier,
            evaluation_mode=evaluation_mode,
            policy_mode=args.flexpi_mode if args.model == "flexpi" else "action-only",
        )
        environment = None
        writers: dict[str, VideoWriter] = {}
        attempt_started = time.perf_counter()
        aborted = False
        try:
            environment, canonical_prompt, actual_seed = prepare_task(
                args.robotwin_root,
                current_task,
                args.phase,
                args.seed + attempt_number - 1,
                state,
            )
            instruction = prompt_override or canonical_prompt
            environment.set_instruction(instruction=instruction)
            module.reset_model(policy)
            observation = environment.get_obs()
            write_frames(args.session_dir, observation)
            initial_frames = camera_frames(observation)
            video_paths = {
                camera: args.session_dir
                / "videos"
                / f"attempt-{attempt_number:03d}-{camera.replace('_camera', '')}.mp4"
                for camera in CAMERA_FILES
            }
            writers = {
                camera: VideoWriter(video_paths[camera], frame)
                for camera, frame in initial_frames.items()
            }
            for camera, frame in initial_frames.items():
                writers[camera].write(frame)
            max_steps = step_limits[current_task] * budget_multiplier
            state.update(
                phase="running",
                canonical_prompt=canonical_prompt,
                prompt=instruction,
                prompt_source=prompt_source,
                model_ack_prompt=None,
                model_ack_prompt_sha256=None,
                seed=actual_seed,
                step=0,
                progress=0.0,
                command_message="Executing native 14D qpos actions in RoboTwin",
            )
            for step in range(max_steps):
                module.eval(environment, policy, observation)
                observation = environment.get_obs()
                write_frames(args.session_dir, observation)
                for camera, frame in camera_frames(observation).items():
                    writers[camera].write(frame)
                succeeded = bool(environment.eval_success or environment.check_success())
                state.update(
                    step=step + 1,
                    progress=(step + 1) / max_steps,
                    task_success=succeeded,
                )
                live_command = inbox.read()
                if live_command:
                    live_action = str(live_command.get("action", ""))
                    if live_action == "stop":
                        should_stop = True
                        aborted = True
                        state.update(command_ack=live_command.get("id"))
                        break
                    if live_action == "set_prompt":
                        replacement = str(live_command.get("prompt", "")).strip()
                        if replacement:
                            instruction = replacement
                            prompt_override = replacement
                            prompt_source = str(live_command.get("source", "typed"))
                            evaluation_mode = "exploratory"
                            environment.set_instruction(instruction=replacement)
                            policy.pending_actions.clear()
                            state.update(
                                prompt=instruction,
                                prompt_source=prompt_source,
                                evaluation_mode=evaluation_mode,
                                command_ack=live_command.get("id"),
                                command_message="Instruction changed; replanning from the next observation",
                            )
                if succeeded or aborted:
                    break
                if args.realtime_delay_ms:
                    time.sleep(args.realtime_delay_ms / 1000.0)

            success = bool(environment.eval_success or environment.check_success()) and not aborted
            rate_eligible = evaluation_mode == "scored" and not aborted
            status = "aborted" if aborted else ("success" if success else ("failure" if rate_eligible else "unscored"))
            if rate_eligible:
                scored_episodes += 1
                successes += int(success)
            history.append(
                {
                    "attempt": attempt_number,
                    "task_id": current_task,
                    "task_position": task_names.index(current_task) + 1,
                    "prompt": instruction,
                    "prompt_source": prompt_source,
                    "evaluation_mode": evaluation_mode,
                    "max_steps": max_steps,
                    "status": status,
                    "rate_eligible": rate_eligible,
                    "duration_seconds": round(time.perf_counter() - attempt_started, 2),
                    "videos": {camera: str(path) for camera, path in video_paths.items()},
                }
            )
        except Exception as error:
            state.update(
                phase="error",
                control_error=f"{type(error).__name__}: {error}",
                command_message="RoboTwin rollout failed; see client.log for the traceback",
            )
            raise
        finally:
            for writer in writers.values():
                writer.close()
            if environment is not None:
                environment.close_env(clear_cache=True)

        unscored = sum(not item.get("rate_eligible") and item["status"] != "aborted" for item in history)
        aborted_count = sum(item["status"] == "aborted" for item in history)
        state.update(
            phase="stopped" if should_stop else "awaiting_command",
            successes=successes,
            episodes=scored_episodes,
            completed_attempts=len(history),
            unscored_attempts=unscored,
            aborted_attempts=aborted_count,
            attempt_history=history,
            prompt_stats=prompt_stats(history),
            task_ids=sorted({item["task_id"] for item in history}),
            last_attempt_status=history[-1]["status"],
            progress=1.0,
            command_message="Session finished" if should_stop else "Ready for another RoboTwin rollout",
        )

    state.update(phase="stopped", finished_at=timestamp(), command_message="Review saved RoboTwin results")
    atomic_json(args.session_dir / "report.json", state.data)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        import traceback

        traceback.print_exc()
        state_path = args.session_dir.expanduser().resolve() / "state.json"
        if state_path.is_file():
            try:
                failed_state = json.loads(state_path.read_text(encoding="utf-8"))
                failed_state.update(
                    phase="error",
                    control_error=f"{type(error).__name__}: {error}",
                    command_message="RoboTwin adapter failed; see client.log",
                    finished_at=timestamp(),
                    updated_at=timestamp(),
                )
                atomic_json(state_path, failed_state)
            except (OSError, json.JSONDecodeError):
                pass
        raise


if __name__ == "__main__":
    main()
