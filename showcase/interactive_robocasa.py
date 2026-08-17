"""Persistent model-plugin + RoboCasa session for the local dashboard."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import pathlib
import shutil
import time

import imageio.v2 as imageio
import numpy as np
import tyro

try:
    from . import backend_registry
    from . import robocasa_runtime as core
    from . import robocasa_policy_plugins
    from . import world_model_plugins
    from . import world_model_registry
except ImportError:  # Direct script execution adds showcase/ to sys.path.
    import backend_registry
    import robocasa_runtime as core
    import robocasa_policy_plugins
    import world_model_plugins
    import world_model_registry


@dataclasses.dataclass
class Args:
    model: str = "pi05"
    world_model: str = "robocasa-sim"
    preview_steps: int = 5
    preview_approval: str = "auto"  # Legacy CLI field; comparisons never gate execution.
    compare_world_model: bool = False
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
    _, policy_spec = backend_registry.require_compatible("robocasa", args.model)
    policy_profile = backend_registry.get_profile("robocasa", policy_spec.key)
    initial_world_model = world_model_registry.require_world_model(
        "robocasa", args.world_model
    )
    names = core.task_names(args.task_set_name)
    if not 0 <= args.task_id < len(names):
        raise ValueError(f"Initial task ID must be in [0, {len(names)})")
    if args.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")
    if args.replan_steps < 1:
        raise ValueError("replan_steps must be positive")
    if args.replan_steps > policy_profile.action_horizon:
        raise ValueError(
            f"replan_steps cannot exceed {policy_spec.display_name}'s "
            f"{policy_profile.action_horizon}-step action horizon"
        )
    if args.preview_steps < 1:
        raise ValueError("preview_steps must be positive")
    if args.preview_approval not in ("manual", "auto"):
        raise ValueError("preview_approval must be manual or auto")
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
    preview_audit_path = session_dir / "preview-audit.jsonl"
    preview_dir = session_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    policy = robocasa_policy_plugins.create_policy_plugin(
        policy_spec.key, args.host, args.port, args.resize_size
    )
    inbox = core.ControlInbox(args.session_dir)
    active_world_model = initial_world_model
    world_model = world_model_plugins.create_world_model_plugin(
        active_world_model.key,
        create_environment=core.create_environment,
        render_frames=core.render_viewer_frames,
        step_environment=core.step_environment,
    )
    compare_world_model = bool(args.compare_world_model and world_model is not None)

    def execution_prefix_steps() -> int:
        return args.replan_steps

    def select_world_model(key: str) -> None:
        nonlocal active_world_model, world_model
        selected = world_model_registry.require_world_model("robocasa", key)
        if selected.key == active_world_model.key:
            return
        if world_model is not None and hasattr(world_model, "close"):
            world_model.close()
        active_world_model = selected
        world_model = world_model_plugins.create_world_model_plugin(
            selected.key,
            create_environment=core.create_environment,
            render_frames=core.render_viewer_frames,
            step_environment=core.step_environment,
        )

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
            "model_plugin": policy_spec.key,
            "model": policy_profile.model_name,
            "model_display_name": policy_spec.display_name,
            "runtime": policy_spec.runtime,
            "policy_transport": policy_spec.transport,
            "policy_endpoint": policy.endpoint,
            "world_model": active_world_model.key,
            "world_model_display_name": active_world_model.display_name,
            "world_model_runtime": active_world_model.runtime,
            "world_model_prediction_kind": active_world_model.prediction_kind,
            "world_model_description": active_world_model.description,
            "available_world_models": world_model_registry.catalog("robocasa"),
            "preview_steps": args.preview_steps,
            "preview_approval": "post_execution_comparison",
            "compare_world_model": compare_world_model,
            "comparison_status": "waiting_for_action_chunk"
            if compare_world_model
            else "disabled",
            "preview_status": "disabled",
            "preview_count": 0,
            "network_audit": args.network_audit,
            "suite": args.task_set_name,
            "split": args.split,
            "task_ids": selected_task_ids,
            "available_tasks": catalog,
            "total_tasks": len(names),
            "dynamic_task_prompts": True,
            "seed": args.seed,
            "configured_replan_steps": args.replan_steps,
            "replan_steps": execution_prefix_steps(),
            "action_horizon": policy_profile.action_horizon,
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

    def publish_viewer_frames(
        env, observation: dict, *, force: bool = False
    ) -> None:
        nonlocal last_viewer_frame_at
        now = time.perf_counter()
        if not force and now - last_viewer_frame_at < 1.0 / args.viewer_fps:
            return
        external, wrist = core.render_viewer_frames(
            env,
            width=args.viewer_width,
            height=args.viewer_height,
            observation=observation,
        )
        state.frames(external, wrist)
        last_viewer_frame_at = now

    def preview_task(task_id: int) -> str:
        task_name = names[task_id]
        env = core.create_environment(task_name, args.split, args.seed)
        try:
            observation, _ = env.reset()
            canonical = str(observation["annotation.human.task_description"])
            camera_height, camera_width = observation[core.CAMERA_KEYS[0]].shape[:2]
            state.update(
                camera_observation_width=camera_width,
                camera_observation_height=camera_height,
            )
            publish_viewer_frames(env, observation, force=True)
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

    def apply_world_model(command: dict) -> None:
        nonlocal compare_world_model
        select_world_model(str(command.get("world_model", "")))
        if world_model is None:
            compare_world_model = False
        state.update(
            world_model=active_world_model.key,
            world_model_display_name=active_world_model.display_name,
            world_model_runtime=active_world_model.runtime,
            world_model_prediction_kind=active_world_model.prediction_kind,
            world_model_description=active_world_model.description,
            compare_world_model=compare_world_model,
            comparison_status="waiting_for_action_chunk"
            if compare_world_model
            else "disabled",
            preview_status="disabled",
            preview_result=None,
            preview_video_url=None,
            actual_video_url=None,
            replan_steps=execution_prefix_steps(),
            command_ack=command.get("id"),
            command_message=f"World model switched to {active_world_model.display_name}",
            control_error=None,
        )

    def apply_world_model_comparison(command: dict) -> None:
        nonlocal compare_world_model
        enabled = bool(command.get("enabled", False))
        if enabled and world_model is None:
            raise ValueError("Select a world model before enabling comparison")
        compare_world_model = enabled
        state.update(
            compare_world_model=compare_world_model,
            comparison_status="waiting_for_action_chunk"
            if compare_world_model
            else "disabled",
            preview_result=None,
            preview_video_url=None,
            actual_video_url=None,
            command_ack=command.get("id"),
            command_message=(
                "World-model comparison enabled for the next rollout"
                if compare_world_model
                else "World-model comparison disabled; policy will execute normally"
            ),
            control_error=None,
        )

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
                elif action == "set_world_model":
                    try:
                        apply_world_model(command)
                    except ValueError as error:
                        state.update(control_error=str(error))
                elif action == "set_world_model_comparison":
                    try:
                        apply_world_model_comparison(command)
                    except ValueError as error:
                        state.update(control_error=str(error))
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
            comparison_active = compare_world_model and world_model is not None
            preview_env = None
            if not comparison_active:
                env = core.create_environment(task_name, args.split, args.seed)
            else:
                env, preview_env = core.create_environment_pair(
                    task_name, args.split, args.seed
                )
                world_model.attach_branch(
                    preview_env,
                    task_name=task_name,
                    split=args.split,
                    seed=args.seed,
                )
            try:
                if preview_env is None:
                    observation, _ = env.reset(seed=args.seed)
                else:
                    observation, _ = core.reset_environment_pair(
                        env, preview_env, args.seed
                    )
                canonical_prompt = str(observation["annotation.human.task_description"])
                camera_height, camera_width = observation[
                    core.CAMERA_KEYS[0]
                ].shape[:2]
                set_catalog_prompt(attempt_task_id, canonical_prompt)
                active_prompt = prompt_override or canonical_prompt
                attempt_budget_multiplier = rollout_budget_multiplier
                attempt_evaluation_mode = evaluation_mode
                base_max_steps = core.base_horizon(task_name)
                attempt_max_steps = base_max_steps * attempt_budget_multiplier
                action_plan = collections.deque()
                replay_images = [np.ascontiguousarray(env.render())]
                prompt_timeline: list[dict] = []
                preview_history: list[dict] = []
                pending_comparison = None
                attempt_preview_count = 0
                attempt_request_count = 0
                first_action_chunk_sha256 = None
                selected_goal_reached = False
                aborted = False
                auto_start_next = False
                abort_reason = None
                step = 0
                attempt_number += 1
                attempt_started = time.perf_counter()
                publish_viewer_frames(env, observation, force=True)

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
                    camera_observation_width=camera_width,
                    camera_observation_height=camera_height,
                    compare_world_model=comparison_active,
                    comparison_status="waiting_for_action_chunk"
                    if comparison_active
                    else "disabled",
                    preview_status="disabled",
                    preview_result=None,
                    preview_video_url=None,
                    actual_video_url=None,
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
                            if pending_comparison is not None:
                                pending_comparison = None
                                state.update(
                                    comparison_status="waiting_for_action_chunk",
                                    preview_status="discarded_after_replan",
                                    preview_result=None,
                                    preview_video_url=None,
                                    actual_video_url=None,
                                )
                            action_plan.clear()
                            state.update(
                                prompt=active_prompt,
                                prompt_source=prompt_source,
                                evaluation_mode=evaluation_mode,
                                command_ack=command.get("id"),
                                command_message="Instruction applied; replanning",
                            )
                        elif action == "set_world_model":
                            state.update(
                                command_ack=command.get("id"),
                                control_error=(
                                    "Pause between rollouts to switch the world model; "
                                    "the current live/preview environment pair is fixed."
                                ),
                            )
                        elif action == "set_world_model_comparison":
                            state.update(
                                command_ack=command.get("id"),
                                control_error=(
                                    "Change world-model comparison between rollouts; "
                                    "the current environment configuration is fixed."
                                ),
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

                    model_input = policy.prepare(observation, active_prompt)
                    robot_state = model_input.robot_state

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
                        request_prompt = model_input.prompt
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
                        action_chunk = policy.infer(model_input)
                        inference_latency = (
                            time.perf_counter() - inference_started
                        ) * 1000.0
                        approved_prefix_steps = execution_prefix_steps()
                        if len(action_chunk) < approved_prefix_steps:
                            raise ValueError(
                                "Policy returned fewer actions than the execution prefix"
                            )
                        action_hash = hashlib.sha256(action_chunk.tobytes()).hexdigest()
                        if first_action_chunk_sha256 is None:
                            first_action_chunk_sha256 = action_hash
                        audit_event = {
                            "created_at": core.timestamp(),
                            "backend": "robocasa",
                            "model_plugin": policy_spec.key,
                            "model": policy_profile.model_name,
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
                            "execution_prefix_steps": approved_prefix_steps,
                        }
                        with audit_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(audit_event) + "\n")
                        latencies.append(inference_latency)
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

                        if comparison_active:
                            attempt_preview_count += 1
                            preview_number = len(preview_history) + 1
                            preview_path = preview_dir / (
                                f"attempt_{attempt_number:03d}_step_{step:05d}_"
                                f"prediction_{preview_number:03d}.mp4"
                            )
                            state.update(
                                comparison_status="predicting",
                                command_message=(
                                    f"Computing a hidden {approved_prefix_steps}-step "
                                    f"prediction with {active_world_model.display_name}; "
                                    "the policy will execute immediately afterward"
                                ),
                            )
                            preview_result = world_model.preview(
                                world_model_plugins.PreviewRequest(
                                    source_env=env,
                                    task_name=task_name,
                                    split=args.split,
                                    seed=args.seed,
                                    action_chunk=action_chunk,
                                    preview_steps=approved_prefix_steps,
                                    width=args.viewer_width,
                                    height=args.viewer_height,
                                    fps=args.viewer_fps,
                                    artifact_path=preview_path,
                                )
                            )
                            preview_event = {
                                "created_at": core.timestamp(),
                                "attempt": attempt_number,
                                "step": step,
                                "request_id": request_id,
                                "action_chunk_sha256": action_hash,
                                "execution_prefix_steps": approved_prefix_steps,
                                **preview_result.as_dict(),
                            }
                            actual_start = core.render_viewer_frames(
                                env,
                                width=args.viewer_width,
                                height=args.viewer_height,
                                observation=observation,
                            )[0]
                            pending_comparison = {
                                "number": preview_number,
                                "prediction_path": preview_path,
                                "actual_path": preview_dir
                                / (
                                    f"attempt_{attempt_number:03d}_step_{step:05d}_"
                                    f"actual_{preview_number:03d}.mp4"
                                ),
                                "event": preview_event,
                                "predicted_state": world_model.predicted_state(),
                                "actual_frames": [actual_start],
                                "actual_started": time.perf_counter(),
                            }
                            state.update(
                                comparison_status="executing_actual",
                                preview_status="hidden_until_actual_complete",
                                preview_result=None,
                                preview_video_url=None,
                                actual_video_url=None,
                                command_message=(
                                    f"Prediction captured; executing the real "
                                    f"{approved_prefix_steps}-step policy prefix"
                                ),
                            )

                        action_plan.extend(action_chunk[:approved_prefix_steps])

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
                    publish_viewer_frames(env, observation)
                    if pending_comparison is not None:
                        actual_frame = core.render_viewer_frames(
                            env,
                            width=args.viewer_width,
                            height=args.viewer_height,
                            observation=observation,
                        )[0]
                        pending_comparison["actual_frames"].append(actual_frame)
                        if not action_plan:
                            actual_path = pending_comparison["actual_path"]
                            imageio.mimwrite(
                                actual_path,
                                pending_comparison["actual_frames"],
                                fps=args.viewer_fps,
                            )
                            latest_prediction = preview_dir / "latest_prediction.mp4"
                            latest_actual = preview_dir / "latest_actual.mp4"
                            shutil.copyfile(
                                pending_comparison["prediction_path"], latest_prediction
                            )
                            shutil.copyfile(actual_path, latest_actual)
                            comparison_event = dict(pending_comparison["event"])
                            actual_state = env.unwrapped.env.sim.get_state()
                            actual_hash = world_model_plugins.state_sha256(actual_state)
                            state_comparison = world_model_plugins.compare_states(
                                pending_comparison["predicted_state"], actual_state
                            )
                            comparison_event.update(
                                actual_artifact_path=str(actual_path),
                                actual_state_sha256=actual_hash,
                                predicted_matches_actual=state_comparison[
                                    "within_tolerance"
                                ],
                                state_comparison=state_comparison,
                                actual_execution_duration_ms=round(
                                    (
                                        time.perf_counter()
                                        - pending_comparison["actual_started"]
                                    )
                                    * 1000.0,
                                    2,
                                ),
                                revealed_at=core.timestamp(),
                            )
                            preview_history.append(comparison_event)
                            with preview_audit_path.open(
                                "a", encoding="utf-8"
                            ) as stream:
                                stream.write(json.dumps(comparison_event) + "\n")
                            comparison_number = pending_comparison["number"]
                            state.update(
                                comparison_status="ready",
                                preview_status="revealed_after_actual_execution",
                                preview_result=comparison_event,
                                preview_history=preview_history,
                                preview_count=sum(
                                    len(item.get("preview_history", []))
                                    for item in history
                                )
                                + len(preview_history),
                                preview_video_url=(
                                    "/previews/latest_prediction.mp4"
                                    f"?v={attempt_number}-{comparison_number}"
                                ),
                                actual_video_url=(
                                    "/previews/latest_actual.mp4"
                                    f"?v={attempt_number}-{comparison_number}"
                                ),
                                command_message=(
                                    "Actual prefix complete; comparison is now visible"
                                ),
                            )
                            pending_comparison = None
                    selected_goal_reached = selected_goal_reached or bool(
                        info.get("success", False)
                    )
                    step += 1
                    if step % 2 == 0 or selected_goal_reached:
                        replay_images.append(np.ascontiguousarray(env.render()))
                    if args.realtime_delay_ms > 0:
                        time.sleep(args.realtime_delay_ms / 1000.0)
                    if selected_goal_reached and attempt_evaluation_mode == "scored":
                        publish_viewer_frames(env, observation, force=True)
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
                    "world_model": active_world_model.key,
                    "preview_count": attempt_preview_count,
                    "preview_history": preview_history,
                    "first_action_chunk_sha256": first_action_chunk_sha256,
                    "status": status,
                    "abort_reason": abort_reason,
                    "duration_seconds": round(duration, 2),
                    "video": str(output_video) if replay_images else None,
                }
            )
            final_state_values = {}
            if pending_comparison is not None:
                final_state_values = {
                    "comparison_status": "waiting_for_action_chunk"
                    if comparison_active
                    else "disabled",
                    "preview_status": "incomplete_actual_prefix_discarded",
                    "preview_result": None,
                    "preview_video_url": None,
                    "actual_video_url": None,
                }
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
                **final_state_values,
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
                elif action == "set_world_model":
                    try:
                        apply_world_model(command)
                    except ValueError as error:
                        state.update(control_error=str(error))
                elif action == "set_world_model_comparison":
                    try:
                        apply_world_model_comparison(command)
                    except ValueError as error:
                        state.update(control_error=str(error))
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

        if world_model is not None and hasattr(world_model, "close"):
            world_model.close()
        state.update(
            phase="stopped",
            finished_at=core.timestamp(),
            command_message="Review saved results",
            **_latency_fields(latencies),
        )
    except Exception as error:
        if world_model is not None and hasattr(world_model, "close"):
            world_model.close()
        state.update(
            phase="error",
            error=f"{type(error).__name__}: {error}",
            finished_at=core.timestamp(),
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))
