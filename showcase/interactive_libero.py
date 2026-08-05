"""Persistent interactive π0.5 + LIBERO session controlled by the dashboard."""

import collections
import dataclasses
import datetime
import json
import logging
import pathlib
import time

import imageio
from libero.libero import benchmark
import numpy as np
from openpi_client import websocket_client_policy
import tyro

import instrumented_libero as core


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_steps_wait: int = 10
    video_out_path: str = "showcase-runs/videos"
    session_dir: str = "showcase-runs/current"
    seed: int = 7
    realtime_delay_ms: int = 35
    network_audit: bool = True
    initial_prompt: str = ""


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
        if attempt["status"] not in ("success", "failure"):
            continue
        key = "task {} | {}".format(attempt["task_id"], attempt["prompt"])
        item = stats.setdefault(key, {"episodes": 0, "successes": 0})
        item["episodes"] += 1
        item["successes"] += int(attempt["status"] == "success")
        item["success_rate"] = round(item["successes"] / item["episodes"], 4)
    return stats


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
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    inbox = ControlInbox(args.session_dir)
    latencies = []
    history = []
    successes = 0
    completed_episodes = 0
    attempt_number = 0
    current_task_id = args.task_id
    prompt_override = args.initial_prompt.strip() or None
    prompt_source = "typed" if prompt_override else "canonical"
    max_steps = core._max_steps(args.task_suite_name)

    state = core.LiveState(
        args.session_dir,
        {
            "phase": "initializing",
            "interactive": True,
            "model": "pi05_libero",
            "runtime": "local JAX/CUDA",
            "policy_endpoint": "ws://{}:{}".format(args.host, args.port),
            "network_audit": args.network_audit,
            "suite": args.task_suite_name,
            "task_ids": [args.task_id],
            "available_tasks": tasks,
            "total_tasks": suite.n_tasks,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "action_horizon": 10,
            "started_at": _timestamp(),
            "successes": 0,
            "episodes": 0,
            "aborted_attempts": 0,
            "attempt_history": [],
            "prompt_stats": {},
            "inference_latencies_ms": [],
        },
    )

    should_stop = False
    while not should_stop:
        attempt_task_id = current_task_id
        task = suite.get_task(current_task_id)
        canonical_prompt = str(task.language)
        active_prompt = prompt_override or canonical_prompt
        initial_states = suite.get_task_init_states(current_task_id)
        env, _ = core._get_libero_env(task, core.LIBERO_ENV_RESOLUTION, args.seed)
        env.reset()
        obs = env.set_init_state(initial_states[attempt_number % len(initial_states)])
        action_plan = collections.deque()
        replay_images = []
        done = False
        aborted = False
        abort_reason = None
        step = 0
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
            max_steps=max_steps,
            progress=0.0,
            last_action_chunk=[],
            current_action=[],
            inference_latency_ms=None,
        )
        logging.info(
            "Interactive attempt %d, task %d: %s",
            attempt_number,
            current_task_id,
            active_prompt,
        )

        while step < max_steps + args.num_steps_wait:
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
                    active_prompt = prompt_override or canonical_prompt
                    action_plan.clear()
                    state.update(
                        prompt=active_prompt,
                        prompt_source=prompt_source,
                        command_ack=command.get("id"),
                    )
                elif action in ("reset", "set_task"):
                    if action == "set_task":
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
                    aborted = True
                    abort_reason = action
                    state.update(command_ack=command.get("id"))
                    break

            if step < args.num_steps_wait:
                obs, _, done, _ = env.step(core.LIBERO_DUMMY_ACTION)
                step += 1
                continue

            external, wrist, robot_state, model_input = core._prepare_observation(
                obs, args.resize_size, active_prompt
            )
            state.frames(external, wrist)
            replay_images.append(external)

            if not action_plan:
                inference_started = time.perf_counter()
                action_chunk = np.asarray(client.infer(model_input)["actions"])
                inference_latency = (time.perf_counter() - inference_started) * 1000.0
                latencies.append(inference_latency)
                action_plan.extend(action_chunk[: args.replan_steps])
                state.update(
                    last_action_chunk=np.round(action_chunk, 5).tolist(),
                    inference_latency_ms=round(inference_latency, 2),
                    inference_latencies_ms=[
                        round(value, 2) for value in latencies[-120:]
                    ],
                )

            current_action = np.asarray(action_plan.popleft())
            state.update(
                step=step - args.num_steps_wait,
                progress=min(1.0, float(step - args.num_steps_wait) / max_steps),
                current_action=np.round(current_action, 5).tolist(),
                robot_state=np.round(robot_state, 5).tolist(),
            )
            obs, _, done, _ = env.step(current_action.tolist())
            if args.realtime_delay_ms > 0:
                time.sleep(args.realtime_delay_ms / 1000.0)
            step += 1
            if done:
                final_external, final_wrist, _, _ = core._prepare_observation(
                    obs, args.resize_size, active_prompt
                )
                state.frames(final_external, final_wrist)
                replay_images.append(final_external)
                break

        env.close()
        duration = time.perf_counter() - attempt_started
        if aborted:
            status = "aborted"
        else:
            status = "success" if done else "failure"
            completed_episodes += 1
            successes += int(done)

        safe_task = canonical_prompt.replace(" ", "_")
        output_video = video_path / "attempt_{:03d}_{}_{}.mp4".format(
            attempt_number, safe_task, status
        )
        if replay_images:
            imageio.mimwrite(
                output_video, [np.asarray(frame) for frame in replay_images], fps=10
            )

        history.append(
            {
                "attempt": attempt_number,
                "task_id": attempt_task_id,
                "canonical_prompt": canonical_prompt,
                "prompt": active_prompt,
                "prompt_source": prompt_source,
                "status": status,
                "abort_reason": abort_reason,
                "duration_seconds": round(duration, 2),
                "video": str(output_video) if replay_images else None,
            }
        )
        warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
        state.update(
            phase="stopped" if should_stop else "awaiting_command",
            task_success=bool(done),
            successes=successes,
            episodes=completed_episodes,
            aborted_attempts=sum(item["status"] == "aborted" for item in history),
            success_rate=round(successes / completed_episodes, 4)
            if completed_episodes
            else None,
            attempt_history=history,
            prompt_stats=_prompt_stats(history),
            task_ids=sorted({item["task_id"] for item in history}),
            last_video=str(output_video) if replay_images else None,
            progress=1.0 if done else state.data.get("progress", 0.0),
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
        )

        while not should_stop and state.data["phase"] == "awaiting_command":
            command = inbox.read()
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
                    str(command.get("source", "typed"))
                    if prompt_override
                    else "canonical"
                )
                state.update(
                    prompt=prompt_override or canonical_prompt,
                    prompt_source=prompt_source,
                    command_ack=command.get("id"),
                )
            elif action == "set_task":
                requested_task = int(command.get("task_id", current_task_id))
                if 0 <= requested_task < suite.n_tasks:
                    current_task_id = requested_task
                    prompt_override = None
                    prompt_source = "canonical"
                    state.update(command_ack=command.get("id"))
                    break
                state.update(control_error="Invalid task ID: {}".format(requested_task))
            elif action in ("reset", "start"):
                state.update(command_ack=command.get("id"))
                break

    state.update(phase="stopped", finished_at=_timestamp())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))
