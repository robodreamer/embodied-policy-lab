from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from showcase import world_model_plugins
from showcase import world_model_registry


def test_registry_keeps_world_model_choice_independent_from_policy():
    simulator = world_model_registry.require_world_model("robocasa", "simulator")
    assert simulator.key == "robocasa-sim"
    assert simulator.action_schema == "robocasa-panda-omron-12d-v1"
    assert simulator.prediction_kind == "simulator_oracle"
    assert "not a learned world model" in simulator.description.lower()
    assert not simulator.is_learned

    learned = world_model_registry.get_world_model("dino-wm")
    assert learned.action_schema == "robocasa-panda-manip-7d-v1"
    assert not learned.available
    with pytest.raises(ValueError, match="12D mobile-manipulator"):
        world_model_registry.require_world_model("robocasa", learned.key)


@dataclasses.dataclass
class FakeState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray


class FakeSim:
    def __init__(self, position=0.0):
        self.state = FakeState(0.0, np.array([position]), np.array([0.0]))

    def get_state(self):
        return FakeState(
            self.state.time, self.state.qpos.copy(), self.state.qvel.copy()
        )

    def set_state(self, state):
        self.state = FakeState(state.time, state.qpos.copy(), state.qvel.copy())

    def forward(self):
        pass


class FakeInner:
    def __init__(self, position=0.0):
        self.sim = FakeSim(position)

    def _get_observations(self, force_update=False):
        value = int(self.sim.state.qpos[0])
        frame = np.full((6, 8, 3), value, dtype=np.uint8)
        return {
            "video.robot0_agentview_left": frame,
            "video.robot0_eye_in_hand": frame,
        }


class FakeEnv:
    def __init__(self, position=0.0):
        self.unwrapped = self
        self.env = FakeInner(position)
        self.closed = False

    def reset(self):
        self.env.sim = FakeSim()
        return {}, {}

    def get_observation(self, observation):
        return observation

    def close(self):
        self.closed = True


def test_simulator_preview_steps_only_the_branch(monkeypatch, tmp_path: Path):
    live = FakeEnv(position=3.0)
    branches = []

    def create_environment(task_name, split, seed):
        branch = FakeEnv()
        branches.append(branch)
        return branch

    def render_frames(env, *, width, height, observation=None):
        value = int(env.env.sim.state.qpos[0]) if observation is None else int(
            observation["video.robot0_agentview_left"][0, 0, 0]
        )
        return (np.full((height, width, 3), value, dtype=np.uint8),)

    def step_environment(env, action):
        env.env.sim.state.qpos[0] += float(action[0])
        env.env.sim.state.time += 1.0
        observation = env.get_observation(env.env._get_observations(force_update=True))
        return observation, 0.0, False, False, {"success": False}

    written = {}
    renderer_lifecycle = []
    monkeypatch.setattr(
        world_model_plugins.imageio,
        "mimwrite",
        lambda path, frames, fps, **kwargs: written.update(
            path=Path(path), frame_count=len(frames), fps=fps, options=kwargs
        ),
    )
    plugin = world_model_plugins.SimulatorCounterfactualPlugin(
        create_environment,
        render_frames,
        step_environment,
        resume_environment=lambda env: renderer_lifecycle.append("resume"),
        suspend_environment=lambda env: renderer_lifecycle.append("suspend"),
        restore_source_environment=lambda env: renderer_lifecycle.append(
            "restore_source"
        ),
    )
    result = plugin.preview(
        world_model_plugins.PreviewRequest(
            source_env=live,
            task_name="FakeTask",
            split="target",
            seed=7,
            action_chunk=np.ones((4, 12), dtype=np.float32),
            preview_steps=2,
            width=8,
            height=6,
            fps=4,
            artifact_path=tmp_path / "preview.mp4",
        )
    )

    assert live.env.sim.state.qpos.tolist() == [3.0]
    assert branches[0].env.sim.state.qpos.tolist() == [5.0]
    assert result.live_state_unchanged
    assert result.previewed_steps == 2
    assert result.predicted_state_sha256 == world_model_plugins.environment_state_sha256(
        branches[0]
    )
    comparison = world_model_plugins.compare_states(
        plugin.predicted_state(), branches[0].env.sim.get_state()
    )
    assert comparison["within_tolerance"]
    assert written == {
        "path": tmp_path / "preview.mp4",
        "frame_count": 3,
        "fps": 4,
        "options": {"macro_block_size": 1},
    }
    assert renderer_lifecycle == ["resume", "suspend", "restore_source"]
    plugin.close()
    assert branches[0].closed


def test_predictor_failure_is_returned_instead_of_raised(tmp_path: Path):
    class BrokenPredictor:
        def preview(self, request):
            raise RuntimeError("diagnostic branch failed")

    result, error = world_model_plugins.try_preview(
        BrokenPredictor(),
        world_model_plugins.PreviewRequest(
            source_env=object(),
            task_name="FakeTask",
            split="target",
            seed=7,
            action_chunk=np.zeros((2, 12), dtype=np.float32),
            preview_steps=2,
            width=8,
            height=6,
            fps=4,
            artifact_path=tmp_path / "never-written.mp4",
        ),
    )

    assert result is None
    assert isinstance(error, RuntimeError)
    assert str(error) == "diagnostic branch failed"


def test_action_contract_diagnostics_separate_base_and_control_mode():
    actions = np.zeros((3, 12), dtype=np.float32)
    actions[:, 11] = -1.0
    diagnostics = world_model_plugins.action_contract_diagnostics(actions)

    assert diagnostics["base_motion_max_abs"] == 0.0
    assert diagnostics["base_motion_within_tolerance"]
    assert diagnostics["control_mode_min"] == -1.0
    assert diagnostics["control_mode_max"] == -1.0
    assert "nonzero_base_motion" not in diagnostics["blocked_reasons"]
    assert not diagnostics["learned_projection_validated"]

    actions[0, 7] = 0.01
    diagnostics = world_model_plugins.action_contract_diagnostics(actions)
    assert not diagnostics["base_motion_within_tolerance"]
    assert "nonzero_base_motion" in diagnostics["blocked_reasons"]
