from pathlib import Path

import pytest

from showcase import backend_registry
from showcase.interactive_robotwin import CAMERA_FILES, humanize_task, task_catalog
from showcase.robotwin_smoke import validate_observation


class ArrayLike:
    def __init__(self, shape):
        self.shape = shape


def test_robotwin_simulator_contract_matches_release_profiles():
    simulator = backend_registry.get_simulator("robotwin")
    assert simulator.simulator == "RoboTwin 2.0 / SAPIEN / Vulkan"
    assert simulator.state_dimension == 14
    assert simulator.action_dimension == 14
    assert simulator.cameras == ("head_camera", "left_camera", "right_camera")

    fastwam = backend_registry.get_profile("robotwin", "fast-wam")
    assert fastwam.model_name == "robotwin_uncond_3cam_384"
    assert fastwam.action_horizon == 32
    assert fastwam.default_replan_steps == 24

    flexpi = backend_registry.get_profile("robotwin", "flex-pi")
    assert flexpi.model_name == "flexpi_robotwin_3cam_384"
    assert flexpi.action_horizon == 32
    assert flexpi.default_replan_steps == 32


@pytest.mark.parametrize("model", ["pi05", "groot-n1.5"])
def test_robotwin_rejects_unvalidated_vla_pairings(model):
    with pytest.raises(ValueError, match="does not support backend"):
        backend_registry.get_profile("robotwin", model)


def test_model_free_observation_validator_accepts_three_camera_14d_contract():
    observation = {
        "joint_action": {"vector": ArrayLike((14,))},
        "observation": {
            "head_camera": {"rgb": ArrayLike((240, 320, 3))},
            "left_camera": {"rgb": ArrayLike((240, 320, 3))},
            "right_camera": {"rgb": ArrayLike((240, 320, 3))},
        },
    }
    assert validate_observation(observation) == {
        "head_camera": [240, 320, 3],
        "left_camera": [240, 320, 3],
        "right_camera": [240, 320, 3],
    }


def test_model_free_observation_validator_rejects_wrong_action_contract():
    observation = {
        "joint_action": {"vector": ArrayLike((7,))},
        "observation": {},
    }
    with pytest.raises(ValueError, match="expected 14D joint state"):
        validate_observation(observation)


def test_setup_pins_upstream_release_and_keeps_downloads_explicit():
    project = Path(__file__).resolve().parents[1]
    setup = (project / "scripts/setup_robotwin.sh").read_text(encoding="utf-8")
    assert 'ROBOTWIN_REVISION="bf44be51cf5717a5595ce59447f2cf5263d2aa95"' in setup
    assert "--download-assets" in setup
    assert "--download-checkpoints" in setup
    assert "download_assets=0" in setup
    assert "download_checkpoints=0" in setup


def test_robotwin_studio_preserves_three_distinct_camera_routes():
    assert CAMERA_FILES == {
        "head_camera": "external.jpg",
        "left_camera": "wrist.jpg",
        "right_camera": "right_wrist.jpg",
    }


def test_robotwin_studio_catalog_uses_named_tasks(tmp_path):
    config = tmp_path / "task_config"
    config.mkdir()
    (config / "_eval_step_limit.yml").write_text(
        "click_bell: 400\nturn_switch: 500\n", encoding="utf-8"
    )
    tasks, limits = task_catalog(tmp_path)
    assert [task["id"] for task in tasks] == ["click_bell", "turn_switch"]
    assert tasks[0]["position"] == 1
    assert limits == {"click_bell": 400, "turn_switch": 500}
    assert humanize_task("click_bell") == "click bell"


def test_dashboard_exposes_right_wrist_camera_route():
    project = Path(__file__).resolve().parents[1]
    server = (project / "showcase/dashboard_server.py").read_text(encoding="utf-8")
    app = (project / "showcase/static/app.js").read_text(encoding="utf-8")
    assert 'route == "/frames/right_wrist.jpg"' in server
    assert "/frames/right_wrist.jpg?t=" in app
