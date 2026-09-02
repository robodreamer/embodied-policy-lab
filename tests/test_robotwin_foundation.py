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
    assert backend_registry.get_simulator("robo_twin") is simulator
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
    assert 'CUROBO_REF="v0.7.8"' in setup
    assert 'CUROBO_REVISION="d64c4b005459db10c5dd867d8b30a87d5bda9bdb"' in setup
    assert 'CUDA_COMPILER_VERSION="12.8"' in setup
    assert 'WARP_VERSION="1.12.0"' in setup
    assert '"warp-lang==$WARP_VERSION"' in setup
    assert 'pixi exec --spec "cuda-compiler=$CUDA_COMPILER_VERSION"' in setup
    assert "--download-assets" in setup
    assert "--download-checkpoints" in setup
    assert "download_assets=0" in setup
    assert "download_checkpoints=0" in setup


def test_flexpi_asset_downloader_uses_file_hashes_not_modelscope_file_revisions():
    project = Path(__file__).resolve().parents[1]
    downloader = (project / "scripts/download_flexpi_assets.py").read_text(
        encoding="utf-8"
    )
    robotwin_setup = (project / "scripts/setup_robotwin.sh").read_text(
        encoding="utf-8"
    )
    libero_setup = (project / "scripts/setup_flexpi_libero.sh").read_text(
        encoding="utf-8"
    )

    assert 'revision="master"' in downloader
    assert "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996" in downloader
    assert "150f75d811d51f6c7760154aa7fec371dccda529" not in downloader
    assert '"$flex_python" "$PROJECT_DIR/scripts/download_flexpi_assets.py"' in robotwin_setup
    assert '"$PYTHON" "$PROJECT_DIR/scripts/download_flexpi_assets.py"' in libero_setup


def test_flexpi_robotwin_launch_uses_complete_checkpoint_without_training_bases():
    project = Path(__file__).resolve().parents[1]
    preparer = (
        project / "showcase/prepare_flexpi_inference_release.py"
    ).read_text(encoding="utf-8")
    studio = (project / "scripts/run_robotwin_studio.sh").read_text(
        encoding="utf-8"
    )
    batch = (project / "scripts/run_robotwin_evaluation.sh").read_text(
        encoding="utf-8"
    )

    assert 'config["model"]["skip_dit_load_from_pretrain"] = True' in preparer
    assert 'config["model"]["action_dit_pretrained_path"] = None' in preparer
    assert '"video": sum(key.startswith("mixtures.video.")' in preparer
    assert '"action": sum(key.startswith("mixtures.action.")' in preparer
    assert "os.link(checkpoint, runtime_checkpoint)" in preparer
    assert "prepare_flexpi_inference_release.py" in studio
    assert "prepare_flexpi_inference_release.py" in batch


def test_robotwin_launch_fails_fast_and_memory_maps_the_release_checkpoint():
    project = Path(__file__).resolve().parents[1]
    adapter = (project / "showcase/interactive_robotwin.py").read_text(
        encoding="utf-8"
    )
    setup = (project / "scripts/setup_robotwin.sh").read_text(encoding="utf-8")
    studio = (project / "scripts/run_robotwin_studio.sh").read_text(
        encoding="utf-8"
    )

    assert 'SETUPTOOLS_VERSION="80.10.2"' in setup
    assert "import pkg_resources, sapien" in setup
    assert "curobo.types.math, curobo.types.robot" in setup
    assert "version('warp-lang') == '$WARP_VERSION'" in setup
    assert "preflight_simulator(args.robotwin_root, state)" in adapter
    assert "from envs.robot.planner import CuroboPlanner" in adapter
    assert 'version("warp-lang") != "1.12.0"' in adapter
    assert "startup_progress(state, args.model_display_name, startup_note, started)" in adapter
    assert '"memory-mapped weights-only"' in adapter
    assert 'kwargs.setdefault("mmap", True)' in adapter
    assert 'kwargs.setdefault("weights_only", True)' in adapter
    assert "import pkg_resources, sapien" in studio


def test_robotwin_smoke_honors_runtime_root_overrides():
    project = Path(__file__).resolve().parents[1]
    smoke = (project / "scripts/run_robotwin_smoke.sh").read_text(encoding="utf-8")
    assert 'local override_name="$1" name="$2" candidate override=""' in smoke
    assert 'override="${!override_name:-}"' in smoke


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


def test_cli_robotwin_catalog_matches_pinned_upstream_task_contract():
    project = Path(__file__).resolve().parents[1]
    catalog = project / "showcase/robotwin_tasks.tsv"
    tasks = {}
    for line in catalog.read_text(encoding="utf-8").splitlines():
        task, steps = line.split("\t")
        tasks[task] = int(steps)

    assert len(tasks) == 50
    assert tasks["click_bell"] == 400
    assert tasks["turn_switch"] == 400
    assert tasks["put_bottles_dustbin"] == 1700


def test_dashboard_exposes_right_wrist_camera_route():
    project = Path(__file__).resolve().parents[1]
    server = (project / "showcase/dashboard_server.py").read_text(encoding="utf-8")
    app = (project / "showcase/static/app.js").read_text(encoding="utf-8")
    assert 'route == "/frames/right_wrist.jpg"' in server
    assert "/frames/right_wrist.jpg?t=" in app
