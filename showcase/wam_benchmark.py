"""Headless matched-protocol benchmark for released LIBERO WAM policies."""

from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import math
import os
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
RUN_SHOWCASE = PROJECT_DIR / "scripts" / "run_showcase.sh"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
# Shared local comparison budgets published by Flex-π. Fast-WAM's native
# evaluator instead uses 400/400/400/700; therefore even the full profile below
# is a matched local protocol, not a reproduction of Fast-WAM's paper protocol.
SUITE_BUDGETS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
TASKS_PER_SUITE = 10
PAPER_TRIALS_PER_TASK = 50
REPLAN_STEPS = 10
PROTOCOL_NAME = "matched-flexpi-libero-local-v1"
CONFIGS = (
    {"key": "fastwam", "model": "fastwam", "mode": "action-only"},
    {"key": "flexpi-action", "model": "flexpi", "mode": "action-only"},
    {"key": "flexpi-joint", "model": "flexpi", "mode": "full-joint"},
)
PUBLISHED_LIBERO = {
    "fastwam": {
        "libero_spatial": 98.2,
        "libero_object": 100.0,
        "libero_goal": 97.0,
        "libero_10": 95.2,
        "average": 97.6,
    },
    "flexpi-action": {
        "libero_spatial": 99.4,
        "libero_object": 99.6,
        "libero_goal": 98.0,
        "libero_10": 96.6,
        "average": 98.4,
    },
    "flexpi-joint": {
        "libero_spatial": 99.6,
        "libero_object": 99.8,
        "libero_goal": 98.6,
        "libero_10": 96.0,
        "average": 98.5,
    },
}
PUBLISHED_LATENCY_MS = {
    "fastwam_project_rtx_5090d": 190.0,
    "fastwam_compiled_in_flexpi_comparison_rtx_5090": 90.0,
    "flexpi_action_rtx_5090": 60.0,
    "flexpi_joint_rtx_5090": 193.0,
}


def _timestamp() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .astimezone()
        .strftime("%Y%m%d-%H%M%S")
    )


def _profile_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "suites": ("libero_spatial",),
            "task_ids": "2",
            "trials": 1,
            "max_policy_steps": 20,
            "latency_warmups": 1,
            "latency_calls": 2,
        }
    if profile == "pilot":
        return {
            "suites": SUITES,
            "task_ids": "0,1",
            "trials": 3,
            "max_policy_steps": None,
            "latency_warmups": 3,
            "latency_calls": 10,
        }
    if profile == "paper":
        return {
            "suites": SUITES,
            "task_ids": "all",
            "trials": PAPER_TRIALS_PER_TASK,
            "max_policy_steps": None,
            "latency_warmups": 3,
            "latency_calls": 20,
        }
    raise ValueError(f"Unknown benchmark profile: {profile}")


def _task_count(task_ids: str) -> int:
    if task_ids.strip().lower() == "all":
        return TASKS_PER_SUITE
    values = [int(value.strip()) for value in task_ids.split(",")]
    if len(values) != len(set(values)):
        raise ValueError("task IDs must not contain duplicates")
    invalid = [value for value in values if value < 0 or value >= TASKS_PER_SUITE]
    if invalid:
        raise ValueError(
            f"task IDs must be in [0, {TASKS_PER_SUITE}): {invalid}"
        )
    return len(values)


def build_plan(
    *,
    profile: str,
    output_dir: pathlib.Path,
    suites: tuple[str, ...] | None = None,
    task_ids: str | None = None,
    trials: int | None = None,
    max_policy_steps: int | None = None,
    latency_warmups: int | None = None,
    latency_calls: int | None = None,
    seed: int = 7,
    config_keys: tuple[str, ...] | None = None,
    common_runtime: bool = True,
) -> list[dict[str, Any]]:
    defaults = _profile_defaults(profile)
    selected_suites = suites or defaults["suites"]
    selected_task_ids = task_ids or defaults["task_ids"]
    selected_trials = trials if trials is not None else defaults["trials"]
    selected_max_steps = (
        max_policy_steps
        if max_policy_steps is not None
        else defaults["max_policy_steps"]
    )
    selected_warmups = (
        latency_warmups if latency_warmups is not None else defaults["latency_warmups"]
    )
    selected_latency_calls = (
        latency_calls if latency_calls is not None else defaults["latency_calls"]
    )
    if selected_trials <= 0:
        raise ValueError("trials must be positive")
    if selected_warmups < 0 or selected_latency_calls < 0:
        raise ValueError("latency probe counts must be non-negative")
    unknown_suites = sorted(set(selected_suites) - set(SUITES))
    if unknown_suites:
        raise ValueError(f"Unsupported suites: {', '.join(unknown_suites)}")
    if len(selected_suites) != len(set(selected_suites)):
        raise ValueError("benchmark suites must not contain duplicates")

    selected_configs = [
        config for config in CONFIGS if not config_keys or config["key"] in config_keys
    ]
    if config_keys and len(selected_configs) != len(set(config_keys)):
        known = {config["key"] for config in CONFIGS}
        unknown = sorted(set(config_keys) - known)
        raise ValueError(f"Unsupported configurations: {', '.join(unknown)}")

    plan = []
    for config in selected_configs:
        for suite_index, suite in enumerate(selected_suites):
            suite_max_steps = selected_max_steps or SUITE_BUDGETS[suite]
            session_dir = output_dir / "sessions" / config["key"] / suite
            probe_warmups = selected_warmups if suite_index == 0 else 0
            probe_calls = selected_latency_calls if suite_index == 0 else 0
            command = [
                str(RUN_SHOWCASE),
                "--backend",
                "libero",
                "--model",
                config["model"],
                "--batch",
                "--task-suite",
                suite,
                "--task-ids",
                selected_task_ids,
                "--trials",
                str(selected_trials),
                "--seed",
                str(seed),
                "--replan-steps",
                str(REPLAN_STEPS),
                "--max-policy-steps",
                str(suite_max_steps),
                "--benchmark-mode",
                "--no-save-videos",
                "--no-network-audit",
                "--no-open",
                "--no-hold-open",
                "--realtime-delay-ms",
                "0",
                "--latency-probe-warmups",
                str(probe_warmups),
                "--latency-probe-calls",
                str(probe_calls),
                "--session-dir",
                str(session_dir),
            ]
            if config["model"] == "flexpi":
                command.extend(("--flexpi-mode", config["mode"]))
            plan.append(
                {
                    "config": config["key"],
                    "model": config["model"],
                    "mode": config["mode"],
                    "suite": suite,
                    "task_ids": selected_task_ids,
                    "task_count": _task_count(selected_task_ids),
                    "trials_per_task": selected_trials,
                    "expected_episodes": _task_count(selected_task_ids)
                    * selected_trials,
                    "seed": seed,
                    "replan_steps": REPLAN_STEPS,
                    "settling_steps": 30 if config["model"] == "fastwam" else 10,
                    "max_policy_steps": suite_max_steps,
                    "latency_probe_warmups": probe_warmups,
                    "latency_probe_calls": probe_calls,
                    "session_dir": str(session_dir),
                    "common_libero_runtime": common_runtime,
                    "command": command,
                }
            )
    return plan


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _session_complete(run: dict[str, Any]) -> bool:
    state = _read_json(pathlib.Path(run["session_dir"]) / "state.json") or {}
    expected_task_ids = (
        list(range(TASKS_PER_SUITE))
        if run["task_ids"].strip().lower() == "all"
        else [int(value.strip()) for value in run["task_ids"].split(",")]
    )
    expected = {
        "suite": run["suite"],
        "task_ids": expected_task_ids,
        "seed": int(run["seed"]),
        "replan_steps": int(run["replan_steps"]),
        "num_steps_wait": int(run["settling_steps"]),
        "max_steps": int(run["max_policy_steps"]),
        "model_plugin": run["model"],
        "policy_mode": run["mode"],
        "benchmark_runtime": (
            "flexpi" if run["common_libero_runtime"] else "native"
        ),
    }
    return (
        state.get("phase") == "complete"
        and int(state.get("episodes") or 0) == int(run["expected_episodes"])
        and all(state.get(key) == value for key, value in expected.items())
    )


def _same_run_spec(planned: dict[str, Any], recorded: dict[str, Any]) -> bool:
    fields = (
        "config",
        "model",
        "mode",
        "suite",
        "task_ids",
        "expected_episodes",
        "max_policy_steps",
        "seed",
        "replan_steps",
        "settling_steps",
        "latency_probe_warmups",
        "latency_probe_calls",
        "common_libero_runtime",
    )
    return all(planned.get(field) == recorded.get(field) for field in fields)


def _manifest_runs(
    path: pathlib.Path, *, profile: str
) -> dict[tuple[str, str], dict[str, Any]]:
    manifest = _read_json(path) or {}
    if (
        manifest.get("schema_version") != 2
        or manifest.get("profile") != profile
        or manifest.get("protocol") != PROTOCOL_NAME
    ):
        return {}
    return {
        (run["config"], run["suite"]): run
        for run in manifest.get("runs", [])
        if isinstance(run, dict) and run.get("config") and run.get("suite")
    }


def _wait_for_session_lock(timeout_seconds: float = 30.0) -> None:
    """Wait for launcher descendants to release the single-session lock."""

    lock_path = PROJECT_DIR / "showcase-runs" / ".active-session.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with lock_path.open("a+") as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "The previous showcase process did not release the "
                        "single-session lock within 30 seconds."
                    )
                time.sleep(0.1)
                continue
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return


def execute_plan(
    plan: list[dict[str, Any]],
    *,
    output_dir: pathlib.Path,
    profile: str,
    resume: bool,
    keep_going: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "benchmark-manifest.json"
    prior_runs = _manifest_runs(manifest_path, profile=profile) if resume else {}
    statuses = []
    for run_index, planned_run in enumerate(plan, start=1):
        run = planned_run
        recorded = prior_runs.get((planned_run["config"], planned_run["suite"]))
        if recorded and _same_run_spec(planned_run, recorded):
            run = {**planned_run, **recorded}
        if resume and _session_complete(run):
            statuses.append({**run, "status": "skipped_complete", "returncode": 0})
            continue
        session_dir = pathlib.Path(run["session_dir"])
        if session_dir.exists() and any(session_dir.iterdir()):
            retry_dir = session_dir.with_name(
                f"{session_dir.name}-retry-{_timestamp()}"
            )
            run = dict(run)
            run["session_dir"] = str(retry_dir)
            command = list(run["command"])
            command[command.index("--session-dir") + 1] = str(retry_dir)
            run["command"] = command
        print(
            f"\n[{run_index}/{len(plan)}] {run['config']} · {run['suite']} · "
            f"{run['expected_episodes']} episodes",
            flush=True,
        )
        environment = os.environ.copy()
        environment["LIBERO_BENCHMARK_RUNTIME"] = (
            "flexpi" if run["common_libero_runtime"] else "native"
        )
        _wait_for_session_lock()
        completed = subprocess.run(
            run["command"], cwd=PROJECT_DIR, env=environment, check=False
        )
        _wait_for_session_lock()
        status = "complete" if completed.returncode == 0 else "failed"
        statuses.append({**run, "status": status, "returncode": completed.returncode})
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "profile": profile,
                    "protocol": PROTOCOL_NAME,
                    "updated_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "runs": statuses + plan[len(statuses) :],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode and not keep_going:
            raise RuntimeError(
                f"Benchmark run failed: {run['config']} / {run['suite']}"
            )
    return statuses


def _gpu_peak_mib(session_dir: pathlib.Path) -> float | None:
    try:
        with (session_dir / "gpu.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except FileNotFoundError:
        return None
    values = []
    for row in rows:
        try:
            values.append(float(row["memory.used [MiB]"].strip()))
        except (KeyError, TypeError, ValueError):
            continue
    return max(values) if values else None


def wilson_interval(successes: int, episodes: int) -> tuple[float, float] | None:
    if episodes <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / episodes
            + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )
    return center - margin, center + margin


def collect_results(runs: list[dict[str, Any]], *, profile: str) -> dict[str, Any]:
    records = []
    for run in runs:
        session_dir = pathlib.Path(run["session_dir"])
        state = _read_json(session_dir / "state.json") or {}
        successes = int(state.get("successes") or 0)
        episodes = int(state.get("episodes") or 0)
        interval = wilson_interval(successes, episodes)
        observed_protocol = {
            key: state.get(key)
            for key in (
                "suite",
                "task_ids",
                "seed",
                "replan_steps",
                "num_steps_wait",
                "max_steps",
                "model_plugin",
                "policy_mode",
                "benchmark_runtime",
            )
        }
        records.append(
            {
                **{key: value for key, value in run.items() if key != "command"},
                "phase": state.get("phase", "missing"),
                "successes": successes,
                "episodes": episodes,
                "success_rate_pct": round(100.0 * successes / episodes, 3)
                if episodes
                else None,
                "wilson_95_pct": [round(100.0 * value, 3) for value in interval]
                if interval
                else None,
                "warm_mean_inference_latency_ms": state.get(
                    "warm_mean_inference_latency_ms"
                ),
                "latency_probe": state.get("latency_probe")
                or _read_json(session_dir / "latency-probe.json"),
                "gpu_peak_mib": _gpu_peak_mib(session_dir),
                "mujoco_version": state.get("mujoco_version"),
                "libero_source": state.get("libero_source"),
                "libero_git": state.get("libero_git"),
                "lab_git": state.get("lab_git"),
                "python_version": state.get("python_version"),
                "policy_metadata": state.get("policy_metadata"),
                "observed_protocol": observed_protocol,
                "session_contract_matches_plan": _session_complete(run),
            }
        )

    aggregates = {}
    for config in (item["key"] for item in CONFIGS):
        selected = [record for record in records if record["config"] == config]
        successes = sum(record["successes"] for record in selected)
        episodes = sum(record["episodes"] for record in selected)
        interval = wilson_interval(successes, episodes)
        latency_probe = next(
            (record["latency_probe"] for record in selected if record["latency_probe"]),
            None,
        )
        aggregates[config] = {
            "successes": successes,
            "episodes": episodes,
            "success_rate_pct": round(100.0 * successes / episodes, 3)
            if episodes
            else None,
            "wilson_95_pct": [round(100.0 * value, 3) for value in interval]
            if interval
            else None,
            "latency_probe": latency_probe,
            "gpu_peak_mib": max(
                (
                    float(record["gpu_peak_mib"])
                    for record in selected
                    if record["gpu_peak_mib"] is not None
                ),
                default=None,
            ),
            "suites": {record["suite"]: record for record in selected},
        }

    expected_pairs = {
        (config["key"], suite) for config in CONFIGS for suite in SUITES
    }
    observed_pairs = {(record["config"], record["suite"]) for record in records}
    expected_config_contract = {
        config["key"]: (config["model"], config["mode"]) for config in CONFIGS
    }
    runtime_fingerprints = {
        (
            record.get("mujoco_version"),
            (record.get("libero_git") or {}).get("revision"),
        )
        for record in records
    }
    lab_revisions = {
        (record.get("lab_git") or {}).get("revision") for record in records
    }
    complete_matched_protocol = (
        bool(records)
        and profile == "paper"
        and observed_pairs == expected_pairs
        and len(records) == len(expected_pairs)
        and len({record["seed"] for record in records}) == 1
        and len(runtime_fingerprints) == 1
        and None not in next(iter(runtime_fingerprints))
        and len(lab_revisions) == 1
        and None not in lab_revisions
        and all(
            record["phase"] == "complete"
            and record.get("status") in ("complete", "skipped_complete")
            and record.get("returncode") == 0
            and record["session_contract_matches_plan"]
            and record["task_ids"] == "all"
            and record["task_count"] == TASKS_PER_SUITE
            and record["trials_per_task"] == PAPER_TRIALS_PER_TASK
            and record["episodes"]
            == TASKS_PER_SUITE * PAPER_TRIALS_PER_TASK
            and record["max_policy_steps"] == SUITE_BUDGETS[record["suite"]]
            and record["replan_steps"] == REPLAN_STEPS
            and record["settling_steps"]
            == (30 if record["model"] == "fastwam" else 10)
            and record["common_libero_runtime"] is True
            and (record["model"], record["mode"])
            == expected_config_contract[record["config"]]
            and (record.get("libero_git") or {}).get("tracked_dirty") is False
            and (record.get("lab_git") or {}).get("tracked_dirty") is False
            and bool((record.get("policy_metadata") or {}).get("upstream_revision"))
            and (record.get("policy_metadata") or {}).get(
                "upstream_tracked_dirty"
            )
            is False
            and bool((record.get("policy_metadata") or {}).get("checkpoint_sha256"))
            for record in records
        )
    )

    def aggregate_rate(key: str) -> float | None:
        return aggregates.get(key, {}).get("success_rate_pct")

    fast_rate = aggregate_rate("fastwam")
    action_rate = aggregate_rate("flexpi-action")
    joint_rate = aggregate_rate("flexpi-joint")

    def server_latency(key: str) -> float | None:
        probe = aggregates.get(key, {}).get("latency_probe") or {}
        return (probe.get("server_total") or {}).get("mean_ms")

    fast_latency = server_latency("fastwam")
    action_latency = server_latency("flexpi-action")
    joint_latency = server_latency("flexpi-joint")
    behavior_qualifier = (
        "matched_local_protocol"
        if complete_matched_protocol
        else ("wiring_only" if profile == "smoke" else "provisional")
    )
    compare_behavior = profile != "smoke"
    claims = {
        "flexpi_action_vs_fastwam_libero": {
            "status": behavior_qualifier,
            "observed_delta_points": round(action_rate - fast_rate, 3)
            if compare_behavior and action_rate is not None and fast_rate is not None
            else None,
            "direction_matches_paper": action_rate > fast_rate
            if compare_behavior and action_rate is not None and fast_rate is not None
            else None,
            "published_delta_points": 0.8,
        },
        "flexpi_joint_vs_fastwam_libero": {
            "status": behavior_qualifier,
            "observed_delta_points": round(joint_rate - fast_rate, 3)
            if compare_behavior and joint_rate is not None and fast_rate is not None
            else None,
            "direction_matches_paper": joint_rate > fast_rate
            if compare_behavior and joint_rate is not None and fast_rate is not None
            else None,
            "published_delta_points": 0.9,
        },
        "flexpi_joint_vs_action_libero": {
            "status": behavior_qualifier,
            "observed_delta_points": round(joint_rate - action_rate, 3)
            if compare_behavior and joint_rate is not None and action_rate is not None
            else None,
            "direction_matches_paper": joint_rate > action_rate
            if compare_behavior and joint_rate is not None and action_rate is not None
            else None,
            "published_delta_points": 0.1,
        },
        "flexpi_action_faster_than_fastwam": {
            "status": "local_stack" if action_latency and fast_latency else "pending",
            "observed_speedup": round(fast_latency / action_latency, 3)
            if action_latency and fast_latency
            else None,
            "direction_matches_paper": action_latency < fast_latency
            if action_latency and fast_latency
            else None,
        },
        "flexpi_joint_cost_vs_action": {
            "status": "local_stack" if action_latency and joint_latency else "pending",
            "observed_ratio": round(joint_latency / action_latency, 3)
            if action_latency and joint_latency
            else None,
            "published_ratio": round(193.0 / 60.0, 3),
        },
        "fastwam_video_cotraining_ablation": {
            "status": "not_testable_from_released_inference_checkpoints"
        },
        "flexpi_data_efficiency_and_real_robot_claims": {
            "status": "not_testable_in_libero_inference_benchmark"
        },
    }
    return {
        "schema_version": 1,
        "profile": profile,
        "protocol": {
            "name": PROTOCOL_NAME,
            "complete": complete_matched_protocol,
            "scope": "matched local LIBERO inference protocol",
            "fastwam_native_suite_budgets": {
                "libero_spatial": 400,
                "libero_object": 400,
                "libero_goal": 400,
                "libero_10": 700,
            },
            "shared_suite_budgets": SUITE_BUDGETS,
        },
        "complete_matched_protocol": complete_matched_protocol,
        "published_libero_success_pct": PUBLISHED_LIBERO,
        "published_latency_ms": PUBLISHED_LATENCY_MS,
        "records": records,
        "aggregates": aggregates,
        "claims": claims,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Fast-WAM vs Flex-π headless LIBERO benchmark",
        "",
        f"Profile: `{results['profile']}`",
        "",
        (
            "Status: **complete matched local protocol** — this covers the full "
            "local task/trial matrix under the shared Flex-π suite budgets; it "
            "is not a reproduction of either paper's complete evaluation."
            if results["complete_matched_protocol"]
            else (
                "Status: **wiring validation only** — the smoke profile is too "
                "short to evaluate behavioral claims."
                if results["profile"] == "smoke"
                else "Status: **provisional/local validation** — do not compare "
                "this run directly with 2,000-episode paper results."
            )
        ),
        "",
        "## Behavioral results",
        "",
        "| Configuration | Suite | Success | 95% Wilson interval | "
        "Published | Episodes |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in results["records"]:
        interval = record["wilson_95_pct"]
        interval_text = f"{interval[0]:.1f}–{interval[1]:.1f}%" if interval else "—"
        published = PUBLISHED_LIBERO.get(record["config"], {}).get(record["suite"])
        lines.append(
            f"| `{record['config']}` | `{record['suite']}` | "
            f"{_fmt(record['success_rate_pct'])}% | {interval_text} | "
            f"{_fmt(published)}% | {record['episodes']} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency results",
            "",
            "All local latency values are batch-1 means after the configured warmups. "
            "`server total` includes the released server's preprocessing and "
            "model pass; "
            "it excludes model startup and simulator time.",
            "",
            "| Configuration | Server total | Denoise core | Round trip | Peak VRAM |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, aggregate in results["aggregates"].items():
        probe = aggregate.get("latency_probe") or {}
        lines.append(
            f"| `{key}` | "
            f"{_fmt((probe.get('server_total') or {}).get('mean_ms'))} ms | "
            f"{_fmt((probe.get('denoise_core') or {}).get('mean_ms'))} ms | "
            f"{_fmt((probe.get('wall_round_trip') or {}).get('mean_ms'))} ms | "
            f"{_fmt(aggregate.get('gpu_peak_mib'), 0)} MiB |"
        )
    lines.extend(
        [
            "",
            "The papers' absolute latency numbers were measured on RTX 5090-class "
            "hardware with optimization stacks not present in this integration. The "
            "local test validates relative deployment cost, not those absolute "
            "numbers.",
            "",
            "## Claim checks",
            "",
            "```json",
            json.dumps(results["claims"], indent=2),
            "```",
            "",
            "## Scope limits",
            "",
            "- `smoke` validates wiring only. `pilot` gives provisional estimates; "
            "the `paper` profile covers 40 tasks × 50 rollouts per configuration "
            "but remains a matched local protocol rather than a paper reproduction.",
            "- Released checkpoints can test inference-time modes, but not "
            "video-co-training, cross-modality-training, or data-scaling ablations "
            "that require retraining.",
            "- LIBERO cannot reproduce RoboTwin, LIBERO-Plus, or bimanual "
            "real-robot claims.",
            "- Fast-WAM and Flex-π retain their native observations and "
            "preprocessing while sharing the same local LIBERO/MuJoCo runtime, "
            "task IDs, budgets, and action prefix.",
            "- Released evaluator settling is retained: Fast-WAM uses 30 initial "
            "no-op steps and Flex-π uses 10.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(
        component.strip() for component in value.split(",") if component.strip()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a matched headless Fast-WAM/Flex-π LIBERO benchmark."
    )
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "paper"), default="smoke"
    )
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--suites", help="Comma-separated suite override")
    parser.add_argument("--task-ids", help="Comma-separated IDs or all")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--max-policy-steps", type=int)
    parser.add_argument("--latency-warmups", type=int)
    parser.add_argument("--latency-calls", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--configs", help="fastwam,flexpi-action,flexpi-joint")
    parser.add_argument("--native-runtimes", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or (
        PROJECT_DIR / "benchmark-runs" / f"wam-libero-{_timestamp()}"
    )
    plan = build_plan(
        profile=args.profile,
        output_dir=output_dir,
        suites=_parse_csv_tuple(args.suites) if args.suites else None,
        task_ids=args.task_ids,
        trials=args.trials,
        max_policy_steps=args.max_policy_steps,
        latency_warmups=args.latency_warmups,
        latency_calls=args.latency_calls,
        seed=args.seed,
        config_keys=_parse_csv_tuple(args.configs) if args.configs else None,
        common_runtime=not args.native_runtimes,
    )
    if args.plan_only:
        print(f"Profile: {args.profile}")
        print(f"Output: {output_dir}")
        for run in plan:
            print(shlex.join(run["command"]))
        return 0

    statuses = execute_plan(
        plan,
        output_dir=output_dir,
        profile=args.profile,
        resume=args.resume,
        keep_going=args.keep_going,
    )
    results = collect_results(statuses, profile=args.profile)
    (output_dir / "benchmark-results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "benchmark-report.md").write_text(
        render_report(results), encoding="utf-8"
    )
    print(f"\nBenchmark report: {output_dir / 'benchmark-report.md'}")
    if any(status.get("returncode") for status in statuses):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
