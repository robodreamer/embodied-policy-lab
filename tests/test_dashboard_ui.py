from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "showcase" / "static"


def test_dashboard_has_runtime_profile_status_bar():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    header = html.split("<header>", 1)[1].split("</header>", 1)[0]

    for element_id in (
        "profileModel",
        "profileEnvironment",
        "profileTaskSet",
        "profileTransport",
    ):
        assert f'id="{element_id}"' in header
        assert f'$("{element_id}")' in javascript


def test_dashboard_has_explicit_startup_indicator():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    header = html.split("<header>", 1)[1].split("</header>", 1)[0]

    assert 'id="loadingBanner"' in header
    assert 'role="status"' in header
    assert "updateLoadingStatus(state)" in javascript
    assert '["waiting", "initializing", "preparing_task"]' in javascript
    assert "READY — START A ROLLOUT" in javascript


def test_dashboard_control_configuration_retries_after_transient_failure():
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "configureControlsOnce" in javascript
    assert "controlsConfiguration = null" in javascript
    assert "configured = true" in javascript
    assert "DASHBOARD UPDATE FAILED" in javascript


def test_dashboard_reveals_opt_in_comparison_after_actual_execution():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "worldModelSelect",
        "compareWorldModel",
        "actualVideo",
        "previewVideo",
    ):
        assert f'id="{element_id}"' in html
        assert f'$("{element_id}")' in javascript
    assert "LIVE SIMULATOR STATE" in html
    assert 'action: "set_world_model_comparison"' in javascript
    assert "ACTUAL POLICY EXECUTION" in html
    assert "PREDICTOR OUTPUT" in html
    assert "SIMULATOR ORACLE REPLAY" in javascript
    assert "policy rollout continued without comparison" in javascript
    assert "loop playsinline" not in html
    assert 'action: "approve_preview"' not in javascript
    assert 'action: "reject_preview"' not in javascript


def test_dashboard_exposes_flexpi_mode_and_delayed_joint_future():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="policyModeSelect"' in html
    assert 'id="actualVideoLabel"' in html
    assert html.index('id="externalFrame"') < html.index('id="previewCard"')
    assert html.index('id="wristFrame"') < html.index('id="previewCard"')
    assert 'action: "set_policy_mode"' in javascript
    assert (
        'const rolloutFinished = ["awaiting_command", "stopped", "complete"]'
        in javascript
    )
    assert "rolloutFinished\n      && state.policy_prediction_status" in javascript
    assert 'policy_prediction_status === "ready"' in javascript
    assert "FLEX-π POST-ROLLOUT WORLD-ACTION CHECK" in javascript
    assert "REVEALED AFTER ROLLOUT" in javascript
    assert 'classList.toggle("policy-view", policyComparisonReady)' in javascript
    assert "GENERATED WRIST FUTURE" in javascript
    assert "full-rollout sample timeline" in javascript
    assert ".play().catch(() => {})" not in javascript


def test_flexpi_comparison_is_revealed_only_after_rollout_completion():
    client = (ROOT / "showcase" / "interactive_libero.py").read_text(encoding="utf-8")

    capture = client.index('policy_prediction_status="buffered_for_post_rollout"')
    rollout_close = client.index("env.close()")
    reveal = client.index(
        'completed_policy_prediction["revealed_after_rollout"] = True'
    )
    assert capture < rollout_close < reveal
    assert "completed_policy_prefixes.append(matched_prefix)" in client
    assert "latest_policy_timeline.json" in client
    assert 'policy_prediction_status="compiling_full_rollout"' in client
    assert "latest_policy_prediction.mp4?attempt={attempt_number}" in client


def test_dashboard_opens_before_heavy_policy_startup():
    launcher = (ROOT / "scripts" / "run_showcase.sh").read_text(encoding="utf-8")

    dashboard_start = launcher.index('setsid "${DASHBOARD_COMMAND[@]}"')
    policy_start = launcher.index('echo "Starting local $MODEL server for $BACKEND..."')
    assert dashboard_start < policy_start
    assert '"phase": "initializing"' in launcher
    assert "Loading policy weights into local accelerator memory" in launcher


def test_fastwam_startup_reports_cpu_progress_and_expected_duration():
    launcher = (ROOT / "scripts" / "run_showcase.sh").read_text(encoding="utf-8")

    assert "wait_for_policy_http" in launcher
    assert "startup_elapsed_seconds" in launcher
    assert "typical startup is 90-120 seconds" in launcher
    assert "GPU activity starts with the first policy request" in launcher
    assert "the loader process is still active" in launcher


def test_launcher_has_independent_concurrency_and_state_reconciliation_guards():
    launcher = (ROOT / "scripts" / "run_showcase.sh").read_text(encoding="utf-8")

    assert "ALLOW_CONCURRENT_LAB_RUNS" in launcher
    assert "flock -n" in launcher
    assert 'state["stop_reason"]' in launcher
    assert "client_exit_without_finalize" in launcher


def test_dashboard_static_copy_is_policy_and_simulator_agnostic():
    static_copy = "\n".join(
        path.read_text(encoding="utf-8") for path in STATIC.glob("*") if path.is_file()
    ).lower()

    for model_or_simulator in (
        "pi0.5",
        "pi-0.5",
        "π0.5",
        "physical intelligence",
        "robocasa is constructing",
        "mujoco / libero",
    ):
        assert model_or_simulator not in static_copy
