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
