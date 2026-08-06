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


def test_dashboard_opens_before_heavy_policy_startup():
    launcher = (ROOT / "scripts" / "run_showcase.sh").read_text(encoding="utf-8")

    dashboard_start = launcher.index('setsid "${DASHBOARD_COMMAND[@]}"')
    policy_start = launcher.index('echo "Starting local $MODEL server for $BACKEND..."')
    assert dashboard_start < policy_start
    assert '"phase": "initializing"' in launcher
    assert "Loading policy weights into local accelerator memory" in launcher


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
