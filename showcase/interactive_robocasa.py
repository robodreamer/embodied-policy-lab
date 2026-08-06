"""Persistent π0.5 + RoboCasa session for the shared local dashboard."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import pathlib
import time

import imageio.v2 as imageio
import numpy as np
from openpi_client import websocket_client_policy
import tyro

try:
    from . import robocasa_runtime as core
except ImportError:  # Direct script execution adds showcase/ to sys.path.
    import robocasa_runtime as core


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    viewer_width: int = 960
    viewer_height: int = 540
    viewer_fps: float = 6.0
    replan_steps: int = 5
    task_set_name: str = "atomic_seen"
    task_id: int = 0
    task_ids: str = ""
    num_trials_per_task: int = 1
    split: str = "target"
    video_out_path: str = "showcase-runs/videos"
    session_dir: str = "showcase-runs/current"
    seed: int = 7
    realtime_delay_ms: int = 35
    network_audit: bool = True
    initial_prompt: str = ""
    initial_evaluation_mode: str = ""
    initial_rollout_budget_multiplier: int = 0
    interactive: bool = True
    auto_start: bool = False


def _latency_fields(latencies: list[float]) -> dict:
    warm = latencies[1:] if len(latencies) > 1 else latencies
    return {
        "cold_inference_latency_ms": round(latencies[0], 2) if latencies else None,
        "warm_mean_inference_latency_ms": round(float(np.mean(warm)), 2)
        if warm
        else None,
        "median_inference_latency_ms": round(float(np.median(warm)), 2)
        if warm
        else None,
        "p95_inference_latency_ms": round(float(np.percentile(warm, 95)), 2)
        if warm
        else None,
    }


def run(args: Args) -> None:
    np.random.seed(args.seed)
    names = core.task_names(args.task_set_name)
    if not 0 <= args.task_id < len(names):
        raise ValueError(f"Initial task ID must be in [0, {len(names)})")
    if args.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")
    if args.replan_steps < 1:
        raise ValueError("replan_steps must be positive")
    if args.viewer_width < 1 or args.viewer_height < 1:
        raise ValueError("viewer_width and viewer_height must be positive")
    if args.viewer_fps <= 0:
        raise ValueError("viewer_fps must be positive")
    if args.initial_evaluation_mode not in ("", "scored", "exploratory"):
        raise ValueError("initial_evaluation_mode must be scored or exploratory")
    if args.initial_rollout_budget_multiplier not in (0, 1, 2, 3):
        raise ValueError("initial_rollout_budget_multiplier must be 1, 2, or 3")
    if not args.interactive and not args.auto_start:
        raise ValueError("Non-interactive RoboCasa runs require auto_start")

    selected_task_ids = (
        [args.task_id]
        if args.interactive
        else core.parse_task_ids(args.task_ids or str(args.task_id), len(names))
    )
    batch_schedule = [
        task_id
        for task_id in selected_task_ids
        for _ in range(args.num_trials_per_task)
    ]
    catalog = core.task_catalog(args.task_set_name)
    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    session_dir = pathlib.Path(args.session_dir)
    audit_path = session_dir / "inference-audit.jsonl"
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    inbox = core.ControlInbox(args.session_dir)

    current_task_id = selected_task_ids[0]
    prompt_override = args.initial_prompt.strip() or None
    prompt_source = "typed" if prompt_override else "canonical"
    evaluation_mode = args.initial_evaluation_mode or (
        "exploratory" if prompt_override else "scored"
    )
    rollout_budget_multiplier = args.initial_rollout_budget_multiplier or (
        3 if prompt_override else (2 if args.interactive else 1)
    )
    latencies: list[float] = []
    history: list[dict] = []
    successes = 0
    completed_attempts = 0
    completed_episodes = 0
    attempt_number = 0
    model_request_count = 0

    initial_task_name = names[current_task_id]
    initial_horizon = core.base_horizon(initial_task_name)
    state = core.LiveState(
        args.session_dir,
        {
            "phase": "initializing",
            "interactive": args.interactive,
            "backend": "robocasa",
            "simulator": "RoboCasa365 / robosuite / MuJoCo",
            "model": "pi05_pretrain_human300",
            "runtime": "local JAX/CUDA",
            "policy_endpoint": f"ws://{args.host}:{args.port}",
            "network_audit": args.network_audit,
            "suite": args.task_set_name,
            "split": args.split,
            "task_ids": selected_task_ids,
            "available_tasks": catalog,
            "total_tasks": len(names),
            "dynamic_task_prompts": True,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "action_horizon": 50,
            "action_dimension": 12,
            "action_labels": [
                "EEF ΔX",
                "EEF ΔY",
                "EEF ΔZ",
                "EEF ROT X",
                "EEF ROT Y",
                "EEF ROT Z",
                "GRIPPER",
                "BASE X",
                "BASE Y",
                "BASE YAW",
                "BASE AUX",
                "CONTROL MODE",
            ],
            "state_dimension": 16,
            "camera_count": 3,
            "model_image_width": args.resize_size,
            "model_image_height": args.resize_size,
            "viewer_width": args.viewer_width,
            "viewer_height": args.viewer_height,
            "viewer_fps": args.viewer_fps,
            "base_max_steps": initial_horizon,
            "max_steps": initial_horizon * rollout_budget_multiplier,
            "rollout_budget_multiplier": rollout_budget_multiplier,
            "evaluation_mode": evaluation_mode,
            "started_at": core.timestamp(),
            "successes": 0,
            "episodes": 0,
            "completed_attempts": 0,
            "unscored_attempts": 0,
            "aborted_attempts": 0,
            "attempt_history": [],
            "prompt_stats": {},
            "inference_latencies_ms": [],
            "model_request_count": 0,
        },
    )

    def set_catalog_prompt(task_id: int, prompt: str) -> None:
        catalog[task_id] = {
            **catalog[task_id],
            "prompt": prompt,
            "prompt_ready": True,
        }

    last_viewer_frame_at = 0.0

    def publish_viewer_frames(env, *, force: bool = False) -> None:
        nonlocal last_viewer_frame_at
        now = time.perf_counter()
        if not force and now - last_viewer_frame_at < 1.0 / args.viewer_fps:
            return
        external, wrist = core.render_viewer_frames(
            env,
            width=args.viewer_width,
            height=args.viewer_height,
        )
        state.frames(external, wrist)
        last_viewer_frame_at = now

    def preview_task(task_id: int) -> str:
        task_name = names[task_id]
        env = core.create_environment(task_name, args.split, args.seed)
        try:
            observation, _ = env.reset()
            canonical = str(observation["annotation.human.task_description"])
            publish_viewer_frames(env, force=True)
        finally:
            env.close()
        set_catalog_prompt(task_id, canonical)
        return canonical

    def apply_rollout_settings(command: dict) -> None:
        nonlocal current_task_id, prompt_override, prompt_source
        nonlocal evaluation_mode, rollout_budget_multiplier
        requested_task = int(command.get("task_id", current_task_id))
        if not 0 <= requested_task < len(names):
            raise ValueError(f"Invalid task ID: {requested_task}")
        task_changed = requested_task != current_task_id
        current_task_id = requested_task
        requested_source = str(command.get("source", "typed"))
        candidate = str(command.get("prompt", "")).strip()
        if requested_source == "canonical":
            prompt_override = None
            prompt_source = "canonical"
        else:
            prompt_override = candidate or None
            prompt_source = (
                requested_source
                if prompt_override
                and requested_source in ("typed", "local_llm", "local_llm_exploratory")
                else "canonical"
            )
        if task_changed and requested_source == "canonical":
            evaluation_mode = "scored"
        requested_multiplier = int(
            command.get("rollout_budget_multiplier", rollout_budget_multiplier)
        )
        if requested_multiplier not in (1, 2, 3):
            raise ValueError("Rollout budget multiplier must be 1, 2, or 3")
        rollout_budget_multiplier = requested_multiplier
        requested_evaluation = str(
            command.get(
                "evaluation_mode",
                "exploratory"
                if prompt_source == "local_llm_exploratory"
                else evaluation_mode,
            )
        )
        if requested_evaluation not in ("scored", "exploratory"):
            raise ValueError("Evaluation mode must be scored or exploratory")
        evaluation_mode = requested_evaluation

    try:
        if args.interactive:
            initial_canonical = preview_task(current_task_id)
        else:
            initial_canonical = catalog[current_task_id]["prompt"]
        state.update(
            phase="awaiting_command" if not args.auto_start else "initializing",
            task_id=current_task_id,
            task_name=names[current_task_id],
            task_position=current_task_id + 1,
            canonical_prompt=initial_canonical,
            prompt=prompt_override or initial_canonical,
            prompt_source=prompt_source,
            available_tasks=catalog,
            command_message=(
                "Choose a task and instruction, then start a rollout"
                if not args.auto_start
                else "Starting command-line rollout"
            ),
        )

        should_stop = False
        if not args.auto_start:
            while not should_stop and attempt_number == 0:
                command = inbox.read()
                if not command:
                    time.sleep(0.1)
                    continue
                action = command.get("action")
                if action == "stop":
                    should_stop = True
                    state.update(command_ack=command.get("id"))
                elif action == "set_task":
                    requested_task = int(command.get("task_id", current_task_id))
                    if not 0 <= requested_task < len(names):
                        state.update(control_error=f"Invalid task ID: {requested_task}")
                        continue
                    current_task_id = requested_task
                    prompt_override = None
                    prompt_source = "canonical"
                    evaluation_mode = "scored"
                    state.update(
                        phase="preparing_task",
                        task_id=current_task_id,
                        task_name=names[current_task_id],
                        task_position=current_task_id + 1,
                        command_message="Preparing selected RoboCasa scene",
                    )
                    canonical = preview_task(current_task_id)
                    horizon = core.base_horizon(names[current_task_id])
                    state.update(
                        phase="awaiting_command",
                        task_id=current_task_id,
                        task_name=names[current_task_id],
                        task_position=current_task_id + 1,
                        canonical_prompt=canonical,
                        prompt=canonical,
                        prompt_source="canonical",
                        evaluation_mode="scored",
                        base_max_steps=horizon,
                        max_steps=horizon * rollout_budget_multiplier,
                        available_tasks=catalog,
                        command_ack=command.get("id"),
                        command_message="Selected scene is ready",
                    )
                elif action == "set_prompt":
                    candidate = str(command.get("prompt", "")).strip()
                    prompt_override = candidate or None
                    prompt_source = (
                        str(command.get("source", "typed"))
                        if prompt_override
                        else "canonical"
                    )
                    evaluation_mode = str(
                        command.get("evaluation_mode", evaluation_mode)
                    )
                    state.update(
                        prompt=prompt_override or initial_canonical,
                        prompt_source=prompt_source,
                        evaluation_mode=evaluation_mode,
                        command_ack=command.get("id"),
                    )
                elif action in ("start", "reset", "start_rollout"):
                    if action == "start_rollout":
                        try:
                            apply_rollout_settings(command)
                        except ValueError as error:
                            state.update(control_error=str(error))
                            continue
                    state.update(
                        command_ack=command.get("id"),
                        command_message="Starting RoboCasa rollout",
                    )
                    break

        while not should_stop:
            attempt_task_id = current_task_id
            task_name = names[attempt_task_id]
            env = core.create_environment(task_name, args.split, args.seed)
            try:
                observation, _ = env.reset()
                canonical_prompt = str(observation["annotation.human.task_description"])
                set_catalog_prompt(attempt_task_id, canonical_prompt)
                active_prompt = prompt_override or canonical_prompt
                attempt_budget_multiplier = rollout_budget_multiplier
                attempt_evaluation_mode = evaluation_mode
                base_max_steps = core.base_horizon(task_name)
                attempt_max_steps = base_max_steps * attempt_budget_multiplier
                action_plan = collections.deque()
                replay_images = [np.ascontiguousarray(env.render())]
                prompt_timeline: list[dict] = []
                attempt_request_count = 0
                first_action_chunk_sha256 = None
                selected_goal_reached = False
                aborted = False
                auto_start_next = False
                abort_reason = None
                step = 0
                attempt_number += 1
                attempt_started = time.perf_counter()
                publish_viewer_frames(env, force=True)

                state.update(
                    phase="running",
                    task_id=attempt_task_id,
                    task_name=task_name,
                    task_position=attempt_task_id + 1,
                    canonical_prompt=canonical_prompt,
                    prompt=active_prompt,
                    prompt_source=prompt_source,
                    available_tasks=catalog,
                    attempt=attempt_number,
                    step=0,
                    base_max_steps=base_max_steps,
                    max_steps=attempt_max_steps,
                    rollout_budget_multiplier=attempt_budget_multiplier,
                    evaluation_mode=attempt_evaluation_mode,
                    progress=0.0,
                    last_action_chunk=[],
                    current_action=[],
                    inference_latency_ms=None,
                    command_message=f"Running attempt {attempt_number}",
                    control_error=None,
                )
                logging.info(
                    "RoboCasa attempt %d, %s: %s",
                    attempt_number,
                    task_name,
                    active_prompt,
                )

                while step < attempt_max_steps:
                    command = inbox.read() if args.interactive else None
                    if command:
                        action = command.get("action")
                        if action == "stop":
                            should_stop = True
                            aborted = True
                            abort_reason = "stopped"
                            state.update(command_ack=command.get("id"))
                            break
                        if action == "set_prompt":
                            candidate = str(command.get("prompt", "")).strip()
                            requested_source = str(command.get("source", "typed"))
                            if requested_source == "canonical" or not candidate:
                                prompt_override = None
                                prompt_source = "canonical"
                                active_prompt = canonical_prompt
                            else:
                                prompt_override = candidate
                                prompt_source = requested_source
                                active_prompt = candidate
                            requested_evaluation = str(
                                command.get("evaluation_mode", evaluation_mode)
                            )
                            if requested_evaluation not in ("scored", "exploratory"):
                                state.update(control_error="Invalid evaluation mode")
                                continue
                            evaluation_mode = requested_evaluation
                            action_plan.clear()
                            state.update(
                                prompt=active_prompt,
                                prompt_source=prompt_source,
                                evaluation_mode=evaluation_mode,
                                command_ack=command.get("id"),
                                command_message="Instruction applied; replanning",
                            )
                        elif action in ("reset", "set_task", "start_rollout"):
                            if action == "start_rollout":
                                try:
                                    apply_rollout_settings(command)
                                except ValueError as error:
                                    state.update(control_error=str(error))
                                    continue
                                auto_start_next = True
                            elif action == "set_task":
                                requested_task = int(
                                    command.get("task_id", current_task_id)
                                )
                                if not 0 <= requested_task < len(names):
                                    state.update(
                                        control_error=f"Invalid task ID: {requested_task}"
                                    )
                                    continue
                                current_task_id = requested_task
                                prompt_override = None
                                prompt_source = "canonical"
                                evaluation_mode = "scored"
                            else:
                                auto_start_next = True
                            aborted = True
                            abort_reason = action
                            state.update(
                                command_ack=command.get("id"),
                                command_message=(
                                    "Starting a fresh rollout"
                                    if action in ("start_rollout", "reset")
                                    else "Switching task"
                                ),
                            )
                            break

                    images, robot_state, model_input = core.prepare_observation(
                        observation, args.resize_size, active_prompt
                    )

                    if not action_plan:
                        if (
                            not prompt_timeline
                            or prompt_timeline[-1]["prompt"] != active_prompt
                        ):
                            prompt_timeline.append(
                                {
                                    "step": step,
                                    "prompt": active_prompt,
                                    "source": prompt_source,
                                }
                            )
                        model_request_count += 1
                        attempt_request_count += 1
                        request_id = model_request_count
                        request_prompt = str(model_input["prompt"])
                        prompt_sha256 = hashlib.sha256(
                            request_prompt.encode("utf-8")
                        ).hexdigest()
                        state.update(
                            model_request_count=model_request_count,
                            model_request_status="waiting_for_response",
                            model_request_prompt=request_prompt,
                            model_request_prompt_sha256=prompt_sha256,
                            model_request_id=request_id,
                        )
                        inference_started = time.perf_counter()
                        response = client.infer(model_input)
                        inference_latency = (
                            time.perf_counter() - inference_started
                        ) * 1000.0
                        action_chunk = core.validate_action_chunk(response["actions"])
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                "Policy returned fewer actions than replan_steps"
                            )
                        action_hash = hashlib.sha256(action_chunk.tobytes()).hexdigest()
                        if first_action_chunk_sha256 is None:
                            first_action_chunk_sha256 = action_hash
                        audit_event = {
                            "created_at": core.timestamp(),
                            "backend": "robocasa",
                            "request_id": request_id,
                            "attempt": attempt_number,
                            "task_id": attempt_task_id,
                            "task_name": task_name,
                            "step": step,
                            "prompt": request_prompt,
                            "prompt_source": prompt_source,
                            "evaluation_mode": attempt_evaluation_mode,
                            "prompt_sha256": prompt_sha256,
                            "action_chunk_shape": list(action_chunk.shape),
                            "action_chunk_sha256": action_hash,
                            "inference_latency_ms": round(inference_latency, 2),
                            "max_steps": attempt_max_steps,
                        }
                        with audit_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(audit_event) + "\n")
                        latencies.append(inference_latency)
                        action_plan.extend(action_chunk[: args.replan_steps])
                        state.update(
                            last_action_chunk=np.round(action_chunk, 5).tolist(),
                            inference_latency_ms=round(inference_latency, 2),
                            inference_latencies_ms=[
                                round(value, 2) for value in latencies[-120:]
                            ],
                            prompt_timeline=prompt_timeline,
                            model_request_status="response_received",
                            model_ack_prompt=request_prompt,
                            model_ack_prompt_source=prompt_source,
                            model_ack_prompt_sha256=prompt_sha256,
                            last_inference_audit=audit_event,
                        )

                    current_action = np.asarray(action_plan.popleft())
                    state.update(
                        step=step,
                        progress=min(1.0, float(step) / attempt_max_steps),
                        current_action=np.round(current_action, 5).tolist(),
                        robot_state=np.round(robot_state, 5).tolist(),
                    )
                    observation, _, _, _, info = core.step_environment(
                        env, current_action
                    )
                    publish_viewer_frames(env)
                    selected_goal_reached = selected_goal_reached or bool(
                        info.get("success", False)
                    )
                    step += 1
                    if step % 2 == 0 or selected_goal_reached:
                        replay_images.append(np.ascontiguousarray(env.render()))
                    if args.realtime_delay_ms > 0:
                        time.sleep(args.realtime_delay_ms / 1000.0)
                    if selected_goal_reached and attempt_evaluation_mode == "scored":
                        publish_viewer_frames(env, force=True)
                        break
            finally:
                env.close()

            duration = time.perf_counter() - attempt_started
            if aborted:
                status = "aborted"
            elif attempt_evaluation_mode == "exploratory":
                status = "unscored"
            else:
                status = "success" if selected_goal_reached else "failure"

            output_video = video_dir / "attempt_{:03d}_{}_{}.mp4".format(
                attempt_number, core.safe_filename(task_name), status
            )
            if replay_images:
                imageio.mimwrite(output_video, replay_images, fps=20)
            if not prompt_timeline:
                prompt_timeline.append(
                    {"step": 0, "prompt": active_prompt, "source": prompt_source}
                )
            mixed_prompt = len({item["prompt"] for item in prompt_timeline}) > 1
            recorded_source = "mixed" if mixed_prompt else prompt_timeline[0]["source"]
            rate_eligible = (
                status in ("success", "failure")
                and not mixed_prompt
                and attempt_evaluation_mode == "scored"
            )
            if status in ("success", "failure", "unscored"):
                completed_attempts += 1
            if rate_eligible:
                completed_episodes += 1
                successes += int(selected_goal_reached)
            history.append(
                {
                    "attempt": attempt_number,
                    "task_id": attempt_task_id,
                    "task_name": task_name,
                    "canonical_prompt": canonical_prompt,
                    "prompt": active_prompt,
                    "prompt_source": recorded_source,
                    "prompt_timeline": prompt_timeline,
                    "mixed_prompt": mixed_prompt,
                    "rate_eligible": rate_eligible,
                    "max_steps": attempt_max_steps,
                    "base_max_steps": base_max_steps,
                    "rollout_budget_multiplier": attempt_budget_multiplier,
                    "evaluation_mode": attempt_evaluation_mode,
                    "selected_goal_reached": selected_goal_reached,
                    "model_request_count": attempt_request_count,
                    "first_action_chunk_sha256": first_action_chunk_sha256,
                    "status": status,
                    "abort_reason": abort_reason,
                    "duration_seconds": round(duration, 2),
                    "video": str(output_video) if replay_images else None,
                }
            )
            state.update(
                phase="stopped" if should_stop else "awaiting_command",
                task_success=selected_goal_reached
                if status in ("success", "failure")
                else None,
                last_attempt_status=status,
                successes=successes,
                episodes=completed_episodes,
                completed_attempts=completed_attempts,
                unscored_attempts=completed_attempts - completed_episodes,
                aborted_attempts=sum(item["status"] == "aborted" for item in history),
                success_rate=round(successes / completed_episodes, 4)
                if completed_episodes
                else None,
                attempt_history=history,
                prompt_stats=core.prompt_stats(history),
                task_ids=sorted({item["task_id"] for item in history}),
                last_video=str(output_video) if replay_images else None,
                progress=1.0
                if status in ("success", "failure", "unscored")
                else state.data.get("progress", 0.0),
                command_message=(
                    "Session finished"
                    if should_stop
                    else "Ready to start another rollout"
                ),
                **_latency_fields(latencies),
            )

            if should_stop:
                break
            if not args.interactive:
                if attempt_number >= len(batch_schedule):
                    should_stop = True
                    break
                current_task_id = batch_schedule[attempt_number]
                continue
            if auto_start_next:
                continue

            while not should_stop and state.data["phase"] == "awaiting_command":
                command = inbox.read()
                if not command:
                    time.sleep(0.1)
                    continue
                action = command.get("action")
                if action == "stop":
                    should_stop = True
                    state.update(
                        command_ack=command.get("id"),
                        command_message="Finishing session",
                    )
                elif action == "set_task":
                    requested_task = int(command.get("task_id", current_task_id))
                    if not 0 <= requested_task < len(names):
                        state.update(control_error=f"Invalid task ID: {requested_task}")
                        continue
                    current_task_id = requested_task
                    prompt_override = None
                    prompt_source = "canonical"
                    evaluation_mode = "scored"
                    state.update(
                        phase="preparing_task",
                        task_id=current_task_id,
                        task_name=names[current_task_id],
                        task_position=current_task_id + 1,
                        command_message="Preparing selected RoboCasa scene",
                    )
                    canonical = preview_task(current_task_id)
                    horizon = core.base_horizon(names[current_task_id])
                    state.update(
                        task_id=current_task_id,
                        task_name=names[current_task_id],
                        task_position=current_task_id + 1,
                        canonical_prompt=canonical,
                        prompt=canonical,
                        prompt_source="canonical",
                        evaluation_mode="scored",
                        base_max_steps=horizon,
                        max_steps=horizon * rollout_budget_multiplier,
                        available_tasks=catalog,
                        command_ack=command.get("id"),
                        command_message="Selected scene is ready",
                    )
                elif action == "set_prompt":
                    candidate = str(command.get("prompt", "")).strip()
                    prompt_override = candidate or None
                    prompt_source = (
                        str(command.get("source", "typed"))
                        if prompt_override
                        else "canonical"
                    )
                    evaluation_mode = str(
                        command.get("evaluation_mode", evaluation_mode)
                    )
                    state.update(
                        prompt=prompt_override or canonical_prompt,
                        prompt_source=prompt_source,
                        evaluation_mode=evaluation_mode,
                        command_ack=command.get("id"),
                        command_message="Instruction staged for the next rollout",
                    )
                elif action == "start_rollout":
                    try:
                        apply_rollout_settings(command)
                    except ValueError as error:
                        state.update(control_error=str(error))
                        continue
                    state.update(
                        command_ack=command.get("id"),
                        command_message="Starting rollout",
                    )
                    break
                elif action in ("reset", "start"):
                    state.update(
                        command_ack=command.get("id"),
                        command_message="Starting rollout",
                    )
                    break

        state.update(
            phase="stopped",
            finished_at=core.timestamp(),
            command_message="Review saved results",
            **_latency_fields(latencies),
        )
    except Exception as error:
        state.update(
            phase="error",
            error=f"{type(error).__name__}: {error}",
            finished_at=core.timestamp(),
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))
