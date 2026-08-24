import json

from showcase.wam_benchmark import SUITE_BUDGETS
from showcase.wam_benchmark import build_plan
from showcase.wam_benchmark import collect_results
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

    assert results["complete_paper_protocol"] is False
    behavior_claim = results["claims"]["flexpi_joint_vs_fastwam_libero"]
    assert behavior_claim["status"] == "wiring_only"
    assert behavior_claim["observed_delta_points"] is None
    assert behavior_claim["direction_matches_paper"] is None
    assert results["claims"]["flexpi_action_faster_than_fastwam"][
        "direction_matches_paper"
    ]
    assert "wiring validation only" in report
    assert "not_testable_from_released_inference_checkpoints" in report
