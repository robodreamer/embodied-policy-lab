"""Persistent interactive LIBERO policy session controlled by the dashboard."""

import collections
import dataclasses
import datetime
import hashlib
import json
import logging
import pathlib
import time

import imageio
from libero.libero import benchmark
import numpy as np
import tyro

import instrumented_libero as core
from libero_policy_plugins import create_libero_policy_client


@dataclasses.dataclass
class Args:
    model: str = "pi05"
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_steps_wait: int = 10
    max_policy_steps: int = 0
    video_out_path: str = "showcase-runs/videos"
    session_dir: str = "showcase-runs/current"
    seed: int = 7
    realtime_delay_ms: int = 35
    network_audit: bool = True
    initial_prompt: str = ""
    flexpi_mode: str = "full-joint"
    auto_start: bool = False


class ControlInbox:
    def __init__(self, session_dir):
        self.directory = pathlib.Path(session_dir) / "controls"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seen = set()

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


def _timestamp():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


def _prompt_stats(history):
    stats = {}
    for attempt in history:
        if not attempt.get("rate_eligible"):
            continue
        key = "task {} | {} | {} steps".format(
            attempt["task_id"], attempt["prompt"], attempt["max_steps"]
        )
        item = stats.setdefault(
            key,
            {
                "episodes": 0,
                "successes": 0,
                "max_steps": attempt["max_steps"],
            },
        )
        item["episodes"] += 1
        item["successes"] += int(attempt["status"] == "success")
        item["success_rate"] = round(item["successes"] / item["episodes"], 4)
    return stats


def _clip_psnr(predicted, actual):
    values = []
    for expected, observed in zip(predicted, actual):
        expected = np.asarray(expected, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float32)
        if expected.shape != observed.shape:
            continue
        mse = float(np.mean((expected - observed) ** 2))
        values.append(10.0 * np.log10((255.0 * 255.0) / max(mse, 1e-12)))
    return round(float(np.mean(values)), 3) if values else None


def _matched_policy_prefix(pending):
    """Return one aligned Flex-π prefix, or None if no future was executed."""

    count = min(len(pending["predicted"]), len(pending["actual"]))
    if count < 2:
        return None
    return {
        "predicted": pending["predicted"][:count],
        "actual": pending["actual"][:count],
        "interval": pending["interval"],
        "executed": pending["executed"],
        "metadata": pending["metadata"],
        "mode": pending["mode"],
    }


def _save_policy_prediction(session_dir, prefixes):
    """Compile all matched Flex-π prefixes without revealing them mid-rollout."""

    if not prefixes:
        return None
    predicted = [frame for prefix in prefixes for frame in prefix["predicted"]]
    actual = [frame for prefix in prefixes for frame in prefix["actual"]]
    output_dir = pathlib.Path(session_dir) / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_predicted_path = output_dir / "latest_policy_prediction_raw_composite.mp4"
    raw_actual_path = output_dir / "latest_policy_actual_raw_composite.mp4"
    imageio.mimwrite(raw_predicted_path, predicted, fps=2, macro_block_size=1)
    imageio.mimwrite(raw_actual_path, actual, fps=2, macro_block_size=1)

    # Flex-pi's released LIBERO checkpoint uses a model tensor layout rather
    # than a presentation layout: 512x288 wrist on top, 256x160 external at
    # bottom-left, and an intentionally black synthetic third-camera slot.
    # Preserve that exact tensor above for audit, but show the useful wrist
    # view in the dashboard so the black slot is not mistaken for corruption.
    wrist_height = 288
    display_predicted = [np.asarray(frame)[:wrist_height, :512] for frame in predicted]
    display_actual = [np.asarray(frame)[:wrist_height, :512] for frame in actual]
    predicted_path = output_dir / "latest_policy_prediction.mp4"
    actual_path = output_dir / "latest_policy_actual.mp4"
    imageio.mimwrite(predicted_path, display_predicted, fps=2, macro_block_size=1)
    imageio.mimwrite(actual_path, display_actual, fps=2, macro_block_size=1)
    timeline = []
    frame_offset = 0
    action_offset = 0
    for prefix_index, prefix in enumerate(prefixes, start=1):
        prefix_frames = len(prefix["predicted"])
        timeline.append(
            {
                "prefix": prefix_index,
                "start_frame": frame_offset,
                "end_frame_exclusive": frame_offset + prefix_frames,
                "start_action": action_offset,
                "executed_actions": prefix["executed"],
                "frame_interval_actions": prefix["interval"],
            }
        )
        frame_offset += prefix_frames
        action_offset += prefix["executed"]
    timeline_path = output_dir / "latest_policy_timeline.json"
    timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    metadata = prefixes[-1]["metadata"]
    return {
        "kind": metadata.get("kind", "joint_world_action"),
        "mode": prefixes[-1]["mode"],
        "prefix_count": len(prefixes),
        "frame_count": len(predicted),
        "frame_interval_actions": prefixes[-1]["interval"],
        "executed_actions": sum(prefix["executed"] for prefix in prefixes),
        "mean_rgb_psnr_db": _clip_psnr(display_predicted, display_actual),
        "raw_composite_rgb_psnr_db": _clip_psnr(predicted, actual),
        "display_view": "wrist",
        "display_width": 512,
        "display_height": wrist_height,
        "raw_composite_prediction": str(raw_predicted_path),
        "raw_composite_actual": str(raw_actual_path),
        "timeline": str(timeline_path),
        "caveat": metadata.get("caveat"),
        "matched_prefixes_completed": True,
        "revealed_after_rollout": False,
    }


def run(args):
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if args.task_id < 0 or args.task_id >= suite.n_tasks:
        raise ValueError("Invalid initial task ID")

    tasks = [
        {"id": task_id, "prompt": str(suite.get_task(task_id).language)}
        for task_id in range(suite.n_tasks)
    ]
    video_path = pathlib.Path(args.video_out_path)
    video_path.mkdir(parents=True, exist_ok=True)
    client = create_libero_policy_client(
        args.model, args.host, args.port, flexpi_mode=args.flexpi_mode
    )
    inbox = ControlInbox(args.session_dir)
    inference_audit_path = pathlib.Path(args.session_dir) / "inference-audit.jsonl"
    latencies = []
    history = []
    successes = 0
    completed_attempts = 0
    completed_episodes = 0
    attempt_number = 0
    current_task_id = args.task_id
    prompt_override = args.initial_prompt.strip() or None
    prompt_source = "typed" if prompt_override else "canonical"
    evaluation_mode = "exploratory" if prompt_override else "scored"
    model_request_count = 0
    base_max_steps = core._max_steps(args.task_suite_name, args.model)
    rollout_budget_multiplier = 3 if prompt_override else 2
    initial_max_steps = base_max_steps * rollout_budget_multiplier
    if args.max_policy_steps > 0:
        initial_max_steps = min(initial_max_steps, args.max_policy_steps)

    state = core.LiveState(
        args.session_dir,
        {
            "phase": "initializing",
            "interactive": True,
            "backend": "libero",
            "simulator": "LIBERO / robosuite / MuJoCo",
            "model_plugin": client.spec.key,
            "model": client.profile.model_name,
            "model_display_name": client.spec.display_name,
            "runtime": client.spec.runtime,
            "policy_transport": client.spec.transport,
            "policy_endpoint": client.endpoint,
            "network_audit": args.network_audit,
            "suite": args.task_suite_name,
            "task_ids": [args.task_id],
            "available_tasks": tasks,
            "total_tasks": suite.n_tasks,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "action_horizon": client.profile.action_horizon,
            "action_dimension": 7,
            "policy_mode": client.mode,
            "available_policy_modes": list(client.available_modes),
            "policy_prediction_status": (
                "available" if client.mode == "full-joint" else "disabled"
            ),
            "policy_prediction_result": None,
            "policy_prediction_video_url": None,
            "policy_actual_video_url": None,
            "model_image_width": client.model_image_width,
            "model_image_height": client.model_image_height,
            "camera_observation_width": core.LIBERO_ENV_RESOLUTION,
            "camera_observation_height": core.LIBERO_ENV_RESOLUTION,
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
            "base_max_steps": base_max_steps,
            "max_steps": initial_max_steps,
            "rollout_budget_multiplier": rollout_budget_multiplier,
            "evaluation_mode": evaluation_mode,
            "started_at": _timestamp(),
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

    def apply_rollout_settings(command):
        nonlocal current_task_id, prompt_override, prompt_source
        nonlocal evaluation_mode, rollout_budget_multiplier
        requested_task = int(command.get("task_id", current_task_id))
        if requested_task < 0 or requested_task >= suite.n_tasks:
            raise ValueError("Invalid task ID: {}".format(requested_task))
        current_task_id = requested_task
        canonical = str(suite.get_task(current_task_id).language)
        candidate = str(command.get("prompt", canonical)).strip()
        prompt_override = candidate if candidate and candidate != canonical else None
        requested_source = str(command.get("source", "typed"))
        prompt_source = (
            requested_source
            if prompt_override
            and requested_source in ("typed", "local_llm", "local_llm_exploratory")
            else "canonical"
        )
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
        requested_policy_mode = command.get("policy_mode")
        if requested_policy_mode is not None:
            client.set_mode(str(requested_policy_mode))

    initial_task = suite.get_task(current_task_id)
    initial_canonical_prompt = str(initial_task.language)
    state.update(
        phase="awaiting_command",
        task_id=current_task_id,
        task_position=current_task_id + 1,
        canonical_prompt=initial_canonical_prompt,
        prompt=prompt_override or initial_canonical_prompt,
        prompt_source=prompt_source,
        command_message=(
            "Model ready. Choose a task and instruction, then press Start rollout."
        ),
    )
    logging.info(
        "Policy and task catalog ready; waiting for Start rollout at the dashboard"
    )

    should_stop = False
    startup_command = (
        {"id": "auto-start", "action": "start_rollout"} if args.auto_start else None
    )
    while not should_stop and attempt_number == 0:
        command = startup_command or inbox.read()
        startup_command = None
        if not command:
            time.sleep(0.1)
            continue
        action = command.get("action")
        if action == "stop":
            should_stop = True
        elif action == "set_prompt":
            candidate = str(command.get("prompt", "")).strip()
            prompt_override = candidate or None
            prompt_source = (
                str(command.get("source", "typed")) if candidate else "canonical"
            )
            requested_evaluation = str(command.get("evaluation_mode", evaluation_mode))
            if requested_evaluation not in ("scored", "exploratory"):
                state.update(control_error="Invalid evaluation mode")
                continue
            evaluation_mode = requested_evaluation
        elif action == "set_task":
            try:
                apply_rollout_settings(command)
            except ValueError as error:
                state.update(control_error=str(error))
                continue
        elif action == "set_policy_mode":
            try:
                client.set_mode(str(command.get("policy_mode", "")))
            except ValueError as error:
                state.update(control_error=str(error))
                continue
            state.update(
                policy_mode=client.mode,
                policy_prediction_status=(
                    "available" if client.mode == "full-joint" else "disabled"
                ),
                command_ack=command.get("id"),
                command_message="Flex-π mode staged for the next rollout",
            )
        elif action in ("start", "reset", "start_rollout"):
            if action == "start_rollout":
                try:
                    apply_rollout_settings(command)
                except ValueError as error:
                    state.update(control_error=str(error))
                    continue
            state.update(
                command_ack=command.get("id"), command_message="Starting rollout"
            )
            break

    while not should_stop:
        attempt_task_id = current_task_id
        task = suite.get_task(current_task_id)
        canonical_prompt = str(task.language)
        active_prompt = prompt_override or canonical_prompt
        attempt_budget_multiplier = rollout_budget_multiplier
        attempt_evaluation_mode = evaluation_mode
        attempt_max_steps = base_max_steps * attempt_budget_multiplier
        if args.max_policy_steps > 0:
            attempt_max_steps = min(attempt_max_steps, args.max_policy_steps)
        initial_states = suite.get_task_init_states(current_task_id)
        env, _ = core._get_libero_env(
            task,
            core.LIBERO_ENV_RESOLUTION,
            args.seed,
            camera_depths=client.requires_depth,
        )
        env.reset()
        obs = env.set_init_state(initial_states[attempt_number % len(initial_states)])
        action_plan = collections.deque()
        replay_images = []
        prompt_timeline = []
        attempt_request_count = 0
        first_action_chunk_sha256 = None
        done = False
        selected_goal_reached = False
        aborted = False
        auto_start_next = False
        abort_reason = None
        step = 0
        pending_policy_prediction = None
        completed_policy_prefixes = []
        completed_policy_prediction = None
        attempt_number += 1
        attempt_started = time.perf_counter()

        state.update(
            phase="running",
            task_id=current_task_id,
            task_position=current_task_id + 1,
            canonical_prompt=canonical_prompt,
            prompt=active_prompt,
            prompt_source=prompt_source,
            attempt=attempt_number,
            step=0,
            max_steps=attempt_max_steps,
            rollout_budget_multiplier=attempt_budget_multiplier,
            evaluation_mode=attempt_evaluation_mode,
            progress=0.0,
            last_action_chunk=[],
            current_action=[],
            inference_latency_ms=None,
            command_message="Running attempt {}".format(attempt_number),
            control_error=None,
            policy_mode=client.mode,
            policy_prediction_status=(
                "waiting_for_joint_inference"
                if client.mode == "full-joint"
                else "disabled"
            ),
            policy_prediction_result=None,
            policy_prediction_video_url=None,
            policy_actual_video_url=None,
        )
        logging.info(
            "Interactive attempt %d, task %d: %s",
            attempt_number,
            current_task_id,
            active_prompt,
        )

        while step < attempt_max_steps + args.num_steps_wait:
            command = inbox.read()
            if command:
                action = command.get("action")
                if action == "stop":
                    should_stop = True
                    aborted = True
                    abort_reason = "stopped"
                    break
                if action == "set_prompt":
                    candidate = str(command.get("prompt", "")).strip()
                    prompt_override = candidate or None
                    prompt_source = (
                        str(command.get("source", "typed"))
                        if prompt_override
                        else "canonical"
                    )
                    requested_evaluation = str(
                        command.get("evaluation_mode", evaluation_mode)
                    )
                    if requested_evaluation not in ("scored", "exploratory"):
                        state.update(control_error="Invalid evaluation mode")
                        continue
                    evaluation_mode = requested_evaluation
                    active_prompt = prompt_override or canonical_prompt
                    action_plan.clear()
                    pending_policy_prediction = None
                    completed_policy_prefixes = []
                    completed_policy_prediction = None
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
                        requested_task = int(command.get("task_id", current_task_id))
                        if requested_task < 0 or requested_task >= suite.n_tasks:
                            state.update(
                                control_error="Invalid task ID: {}".format(
                                    requested_task
                                )
                            )
                            continue
                        current_task_id = requested_task
                        prompt_override = None
                        prompt_source = "canonical"
                        evaluation_mode = "scored"
                    aborted = True
                    abort_reason = action
                    state.update(
                        command_ack=command.get("id"),
                        command_message=(
                            "Starting a fresh rollout"
                            if action == "start_rollout"
                            else (
                                "Switching task"
                                if action == "set_task"
                                else "Resetting"
                            )
                        ),
                    )
                    break
                elif action == "set_policy_mode":
                    state.update(
                        control_error=(
                            "Policy mode cannot change mid-rollout; abort or wait for "
                            "completion, then select the next mode."
                        ),
                        command_ack=command.get("id"),
                    )

            if step < args.num_steps_wait:
                obs, _, done, _ = env.step(core.LIBERO_DUMMY_ACTION)
                step += 1
                continue

            external, wrist, robot_state, model_input = core._prepare_observation(
                obs, args.resize_size, active_prompt, client, env
            )
            state.frames(external, wrist)
            replay_images.append(external)

            if not action_plan:
                if (
                    not prompt_timeline
                    or prompt_timeline[-1]["prompt"] != active_prompt
                ):
                    prompt_timeline.append(
                        {
                            "step": max(0, step - args.num_steps_wait),
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
                policy_response = client.infer(model_input)
                action_chunk = np.asarray(policy_response["actions"])
                inference_latency = (time.perf_counter() - inference_started) * 1000.0
                action_chunk_sha256 = hashlib.sha256(action_chunk.tobytes()).hexdigest()
                if first_action_chunk_sha256 is None:
                    first_action_chunk_sha256 = action_chunk_sha256
                audit_event = {
                    "created_at": _timestamp(),
                    "request_id": request_id,
                    "attempt": attempt_number,
                    "task_id": attempt_task_id,
                    "step": max(0, step - args.num_steps_wait),
                    "prompt": request_prompt,
                    "prompt_source": prompt_source,
                    "evaluation_mode": attempt_evaluation_mode,
                    "prompt_sha256": prompt_sha256,
                    "action_chunk_sha256": action_chunk_sha256,
                    "inference_latency_ms": round(inference_latency, 2),
                    "policy_timing": policy_response.get("timing"),
                    "policy_artifact": policy_response.get("artifact"),
                    "policy_mode": client.mode,
                    "policy_prediction": policy_response.get("prediction"),
                    "max_steps": attempt_max_steps,
                }
                with inference_audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(audit_event) + "\n")
                latencies.append(inference_latency)
                action_plan.extend(action_chunk[: args.replan_steps])
                prediction_frames = policy_response.get("prediction_frames") or []
                prediction_metadata = policy_response.get("prediction") or {}
                if prediction_frames:
                    interval = int(prediction_metadata.get("frame_interval_actions", 4))
                    pending_policy_prediction = {
                        "predicted": prediction_frames,
                        "actual": [client.compose_preview(external, wrist)],
                        "capture_steps": set(
                            range(interval, args.replan_steps + 1, interval)
                        ),
                        "interval": interval,
                        "executed": 0,
                        "metadata": prediction_metadata,
                        "mode": client.mode,
                    }
                    state.update(policy_prediction_status="executing_actual_prefix")
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
                step=step - args.num_steps_wait,
                progress=min(
                    1.0, float(step - args.num_steps_wait) / attempt_max_steps
                ),
                current_action=np.round(current_action, 5).tolist(),
                robot_state=np.round(robot_state, 5).tolist(),
            )
            obs, _, done, _ = env.step(current_action.tolist())
            if pending_policy_prediction is not None:
                pending_policy_prediction["executed"] += 1
                executed = pending_policy_prediction["executed"]
                if executed in pending_policy_prediction["capture_steps"]:
                    actual_external, actual_wrist = core._display_images(obs)
                    pending_policy_prediction["actual"].append(
                        client.compose_preview(actual_external, actual_wrist)
                    )
                if done or not action_plan:
                    matched_prefix = _matched_policy_prefix(pending_policy_prediction)
                    if matched_prefix:
                        completed_policy_prefixes.append(matched_prefix)
                        state.update(
                            policy_prediction_status="buffered_for_post_rollout"
                        )
                    pending_policy_prediction = None
            selected_goal_reached = selected_goal_reached or bool(done)
            if args.realtime_delay_ms > 0:
                time.sleep(args.realtime_delay_ms / 1000.0)
            step += 1
            if done and attempt_evaluation_mode == "scored":
                final_external, final_wrist, _, _ = core._prepare_observation(
                    obs, args.resize_size, active_prompt, client, env
                )
                state.frames(final_external, final_wrist)
                replay_images.append(final_external)
                break

        if pending_policy_prediction is not None:
            matched_prefix = _matched_policy_prefix(pending_policy_prediction)
            if matched_prefix:
                completed_policy_prefixes.append(matched_prefix)
        env.close()
        duration = time.perf_counter() - attempt_started
        if completed_policy_prefixes:
            state.update(
                policy_prediction_status="compiling_full_rollout",
                command_message="Compiling the post-rollout Flex-π replay",
            )
        completed_policy_prediction = _save_policy_prediction(
            args.session_dir, completed_policy_prefixes
        )
        if completed_policy_prediction:
            completed_policy_prediction = dict(completed_policy_prediction)
            completed_policy_prediction["revealed_after_rollout"] = True
        if auto_start_next:
            # A reset/task switch immediately begins another rollout. Keep the
            # prior prefix as an artifact, but do not flash its result between
            # the two running states.
            completed_policy_prediction = None
        if aborted:
            status = "aborted"
        elif attempt_evaluation_mode == "exploratory":
            status = "unscored"
        else:
            status = "success" if selected_goal_reached else "failure"

        safe_task = canonical_prompt.replace(" ", "_")
        output_video = video_path / "attempt_{:03d}_{}_{}.mp4".format(
            attempt_number, safe_task, status
        )
        if replay_images:
            imageio.mimwrite(
                output_video, [np.asarray(frame) for frame in replay_images], fps=10
            )

        if not prompt_timeline:
            prompt_timeline.append(
                {"step": 0, "prompt": active_prompt, "source": prompt_source}
            )
        mixed_prompt = len({item["prompt"] for item in prompt_timeline}) > 1
        attempt_prompt_source = (
            "mixed" if mixed_prompt else prompt_timeline[0]["source"]
        )
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
                "canonical_prompt": canonical_prompt,
                "prompt": active_prompt,
                "prompt_source": attempt_prompt_source,
                "prompt_timeline": prompt_timeline,
                "mixed_prompt": mixed_prompt,
                "rate_eligible": rate_eligible,
                "max_steps": attempt_max_steps,
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
        warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
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
            prompt_stats=_prompt_stats(history),
            task_ids=sorted({item["task_id"] for item in history}),
            last_video=str(output_video) if replay_images else None,
            policy_prediction_status=(
                "ready"
                if completed_policy_prediction
                else ("unavailable" if client.mode == "full-joint" else "disabled")
            ),
            policy_prediction_result=completed_policy_prediction,
            policy_prediction_video_url=(
                f"/previews/latest_policy_prediction.mp4?attempt={attempt_number}"
                if completed_policy_prediction
                else None
            ),
            policy_actual_video_url=(
                f"/previews/latest_policy_actual.mp4?attempt={attempt_number}"
                if completed_policy_prediction
                else None
            ),
            progress=1.0
            if status in ("success", "failure", "unscored")
            else state.data.get("progress", 0.0),
            cold_inference_latency_ms=round(latencies[0], 2) if latencies else None,
            warm_mean_inference_latency_ms=round(float(np.mean(warm_latencies)), 2)
            if warm_latencies
            else None,
            median_inference_latency_ms=round(float(np.median(warm_latencies)), 2)
            if warm_latencies
            else None,
            p95_inference_latency_ms=round(float(np.percentile(warm_latencies, 95)), 2)
            if warm_latencies
            else None,
            command_message=(
                "Session finished" if should_stop else "Ready to start another rollout"
            ),
        )

        if auto_start_next and not should_stop:
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
                    command_ack=command.get("id"), command_message="Finishing session"
                )
            elif action == "set_prompt":
                candidate = str(command.get("prompt", "")).strip()
                prompt_override = candidate or None
                prompt_source = (
                    str(command.get("source", "typed"))
                    if prompt_override
                    else "canonical"
                )
                requested_evaluation = str(
                    command.get("evaluation_mode", evaluation_mode)
                )
                if requested_evaluation not in ("scored", "exploratory"):
                    state.update(control_error="Invalid evaluation mode")
                    continue
                evaluation_mode = requested_evaluation
                state.update(
                    prompt=prompt_override or canonical_prompt,
                    prompt_source=prompt_source,
                    evaluation_mode=evaluation_mode,
                    command_ack=command.get("id"),
                    command_message="Instruction staged for the next rollout",
                )
            elif action == "set_task":
                requested_task = int(command.get("task_id", current_task_id))
                if 0 <= requested_task < suite.n_tasks:
                    current_task_id = requested_task
                    prompt_override = None
                    prompt_source = "canonical"
                    evaluation_mode = "scored"
                    state.update(
                        command_ack=command.get("id"), command_message="Switching task"
                    )
                    canonical_prompt = str(suite.get_task(current_task_id).language)
                    state.update(
                        task_id=current_task_id,
                        task_position=current_task_id + 1,
                        canonical_prompt=canonical_prompt,
                        prompt=canonical_prompt,
                        prompt_source="canonical",
                        evaluation_mode=evaluation_mode,
                    )
                    continue
                state.update(control_error="Invalid task ID: {}".format(requested_task))
            elif action == "set_policy_mode":
                try:
                    client.set_mode(str(command.get("policy_mode", "")))
                except ValueError as error:
                    state.update(control_error=str(error))
                    continue
                state.update(
                    policy_mode=client.mode,
                    command_ack=command.get("id"),
                    command_message="Flex-π mode staged for the next rollout",
                    policy_prediction_status=(
                        "available" if client.mode == "full-joint" else "disabled"
                    ),
                )
            elif action == "start_rollout":
                try:
                    apply_rollout_settings(command)
                except ValueError as error:
                    state.update(control_error=str(error))
                    continue
                state.update(
                    command_ack=command.get("id"), command_message="Starting rollout"
                )
                break
            elif action in ("reset", "start"):
                state.update(
                    command_ack=command.get("id"), command_message="Starting rollout"
                )
                break

    state.update(
        phase="stopped",
        finished_at=_timestamp(),
        command_message="Review saved results",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))
