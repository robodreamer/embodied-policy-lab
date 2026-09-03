import json

import pytest

from showcase.wam_benchmark import SUITE_BUDGETS
from showcase.wam_benchmark import build_plan
from showcase.wam_benchmark import collect_results
from showcase.wam_benchmark import execute_plan
from showcase.wam_benchmark import render_report


def test_smoke_plan_is_bounded_and_runs_all_three_configurations(tmp_path):
    plan = build_plan(profile="smoke", output_dir=tmp_path)

    assert [run["config"] for run in plan] == [
        "fastwam",
        "flexpi-action",
        "flexpi-joint",
    ]
    assert all(run["suite"] == "libero_spatial" for run in plan)
    assert all(run["task_ids"] == "2" for run in plan)
    assert all(run["expected_episodes"] == 1 for run in plan)
    assert all(run["max_policy_steps"] == 20 for run in plan)
    assert all("--benchmark-mode" in run["command"] for run in plan)
    assert all("--no-save-videos" in run["command"] for run in plan)


def test_paper_plan_matches_flexpi_libero_protocol(tmp_path):
    plan = build_plan(profile="paper", output_dir=tmp_path)

    assert len(plan) == 12
    for run in plan:
        assert run["task_ids"] == "all"
        assert run["trials_per_task"] == 50
        assert run["expected_episodes"] == 500
        assert run["max_policy_steps"] == SUITE_BUDGETS[run["suite"]]
    for config in {run["config"] for run in plan}:
        probes = [run for run in plan if run["config"] == config]
        assert sum(run["latency_probe_calls"] > 0 for run in probes) == 1
        assert (
            next(run for run in probes if run["latency_probe_calls"])[
                "latency_probe_calls"
            ]
            == 20
        )


@pytest.mark.parametrize("task_ids", ("0,0", "10", "-1"))
def test_plan_rejects_ambiguous_or_invalid_task_ids(tmp_path, task_ids):
    with pytest.raises(ValueError, match="task IDs"):
        build_plan(profile="smoke", output_dir=tmp_path, task_ids=task_ids)


def test_plan_rejects_duplicate_suites(tmp_path):
    with pytest.raises(ValueError, match="suites must not contain duplicates"):
        build_plan(
            profile="pilot",
            output_dir=tmp_path,
            suites=("libero_spatial", "libero_spatial"),
        )


@pytest.mark.parametrize(
    ("common_runtime", "expected_complete"), ((True, True), (False, False))
)
def test_full_matrix_requires_the_shared_runtime(
    tmp_path, common_runtime, expected_complete
):
    plan = build_plan(
        profile="paper", output_dir=tmp_path, common_runtime=common_runtime
    )
    completed_runs = []
    for run in plan:
        session = tmp_path / "sessions" / run["config"] / run["suite"]
        session.mkdir(parents=True)
        (session / "state.json").write_text(
            json.dumps(
                {
                    "phase": "complete",
                    "successes": run["expected_episodes"],
                    "episodes": run["expected_episodes"],
                    "suite": run["suite"],
                    "task_ids": list(range(10)),
                    "seed": run["seed"],
                    "replan_steps": run["replan_steps"],
                    "num_steps_wait": run["settling_steps"],
                    "max_steps": run["max_policy_steps"],
                    "model_plugin": run["model"],
                    "policy_mode": run["mode"],
                    "benchmark_runtime": (
                        "flexpi" if common_runtime else "native"
                    ),
                    "mujoco_version": "3.3.2",
                    "libero_git": {
                        "revision": "libero-fixture",
                        "tracked_dirty": False,
                    },
                    "lab_git": {
                        "revision": "lab-fixture",
                        "tracked_dirty": False,
                    },
                    "policy_metadata": {
                        "upstream_revision": f"{run['model']}-fixture",
                        "upstream_tracked_dirty": False,
                        "checkpoint_sha256": "fixture-sha256",
                        "auxiliary_asset_sha256": (
                            {"fixture": "fixture-sha256"}
                            if run["model"] == "flexpi"
                            else None
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        completed_runs.append({**run, "status": "complete", "returncode": 0})

    results = collect_results(completed_runs, profile="paper")

    assert results["complete_matched_protocol"] is expected_complete


def test_result_report_keeps_smoke_claims_provisional(tmp_path):
    plan = build_plan(profile="smoke", output_dir=tmp_path)
    observed = {
        "fastwam": (0, 1, 1800.0),
        "flexpi-action": (1, 1, 1300.0),
        "flexpi-joint": (1, 1, 2900.0),
    }
    for run in plan:
        session = tmp_path / "sessions" / run["config"] / run["suite"]
        session.mkdir(parents=True)
        successes, episodes, latency = observed[run["config"]]
        state = {
            "phase": "complete",
            "successes": successes,
            "episodes": episodes,
            "mujoco_version": "3.3.2",
            "libero_source": "/fixture/LIBERO",
            "latency_probe": {
                "server_total": {"mean_ms": latency},
                "denoise_core": {"mean_ms": latency - 100.0},
                "wall_round_trip": {"mean_ms": latency + 10.0},
            },
        }
        (session / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (session / "gpu.csv").write_text(
            "timestamp,name,memory.used [MiB],utilization.gpu [%],"
            "power.draw [W],temperature.gpu\n"
            "now,GPU,16000,50,100,60\n",
            encoding="utf-8",
        )

    results = collect_results(plan, profile="smoke")
    report = render_report(results)

    assert results["complete_matched_protocol"] is False
    behavior_claim = results["claims"]["flexpi_joint_vs_fastwam_libero"]
    assert behavior_claim["status"] == "wiring_only"
    assert behavior_claim["observed_delta_points"] is None
    assert behavior_claim["direction_matches_paper"] is None
    assert results["claims"]["flexpi_action_faster_than_fastwam"][
        "direction_matches_paper"
    ]
    assert "wiring validation only" in report
    assert "not_testable_from_released_inference_checkpoints" in report


def test_resume_reuses_a_completed_retry_directory(tmp_path, monkeypatch):
    plan = build_plan(
        profile="smoke", output_dir=tmp_path, config_keys=("fastwam",)
    )
    planned = plan[0]
    canonical = tmp_path / "sessions" / "fastwam" / "libero_spatial"
    canonical.mkdir(parents=True)
    (canonical / "partial.txt").write_text("interrupted", encoding="utf-8")
    retry = canonical.with_name("libero_spatial-retry-prior")
    retry.mkdir()
    state = {
        "phase": "complete",
        "suite": planned["suite"],
        "task_ids": [2],
        "seed": planned["seed"],
        "replan_steps": planned["replan_steps"],
        "num_steps_wait": planned["settling_steps"],
        "max_steps": planned["max_policy_steps"],
        "model_plugin": planned["model"],
        "policy_mode": planned["mode"],
        "benchmark_runtime": "flexpi",
        "episodes": planned["expected_episodes"],
    }
    (retry / "state.json").write_text(json.dumps(state), encoding="utf-8")
    retry_run = {**planned, "session_dir": str(retry)}
    retry_run["command"] = list(planned["command"])
    retry_run["command"][retry_run["command"].index("--session-dir") + 1] = str(
        retry
    )
    (tmp_path / "benchmark-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "profile": "smoke",
                "protocol": "matched-flexpi-libero-local-v1",
                "runs": [{**retry_run, "status": "complete"}],
            }
        ),
        encoding="utf-8",
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("a matching completed retry must not be relaunched")

    monkeypatch.setattr("showcase.wam_benchmark.subprocess.run", unexpected_run)
    statuses = execute_plan(
        plan,
        output_dir=tmp_path,
        profile="smoke",
        resume=True,
        keep_going=False,
    )

    assert statuses[0]["status"] == "skipped_complete"
    assert statuses[0]["session_dir"] == str(retry)
