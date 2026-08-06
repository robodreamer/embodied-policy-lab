from __future__ import annotations

import numpy as np
import pytest

from showcase import backend_registry
from showcase import robocasa_policy_plugins as plugins


def observation() -> dict:
    return {
        "video.robot0_agentview_left": np.zeros((64, 96, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.ones((64, 96, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.full((64, 96, 3), 2, dtype=np.uint8),
        "state.end_effector_position_relative": np.arange(3, dtype=np.float32),
        "state.end_effector_rotation_relative": np.arange(3, dtype=np.float32),
        "state.gripper_qpos": np.arange(2, dtype=np.float32),
        "state.base_position": np.arange(3, dtype=np.float32),
        "state.base_rotation": np.arange(5, dtype=np.float32),
    }


class FakeGrootClient:
    def __init__(self, response: dict):
        self.response = response
        self.request = None

    def get_action(self, request: dict) -> dict:
        self.request = request
        return self.response


def groot_plugin(response: dict) -> plugins.GrootN15PolicyPlugin:
    plugin = object.__new__(plugins.GrootN15PolicyPlugin)
    plugin.spec = backend_registry.get_policy("groot-n1.5")
    plugin.profile = backend_registry.get_profile("robocasa", "groot-n1.5")
    plugin.host = "127.0.0.1"
    plugin.port = 8000
    plugin._client = FakeGrootClient(response)
    return plugin


def valid_response(horizon: int = 16) -> dict:
    widths = (3, 3, 1, 4, 1)
    return {
        key: np.full((horizon, width), index, dtype=np.float32)
        for index, (key, width) in enumerate(zip(plugins.GR00T_ACTION_KEYS, widths))
    }


def test_groot_contract_preserves_three_views_and_exact_prompt():
    plugin = groot_plugin(valid_response())
    prepared = plugin.prepare(observation(), "put the mug in the sink")

    assert prepared.prompt == "put the mug in the sink"
    assert prepared.robot_state.shape == (16,)
    assert prepared.payload[plugins.GR00T_LANGUAGE_KEY] == [prepared.prompt]
    for key in plugins.GR00T_VIDEO_KEYS:
        assert prepared.payload[key].shape == (1, 64, 96, 3)
    for key in plugins.GR00T_STATE_KEYS:
        assert prepared.payload[key].shape[0] == 1


def test_groot_action_fields_are_flattened_in_official_order():
    plugin = groot_plugin(valid_response())
    actions = plugin.infer(plugin.prepare(observation(), "close the lid"))

    assert actions.shape == (16, 12)
    assert actions[0].tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        3,
        3,
        3,
        3,
        4,
    ]


def test_groot_rejects_incomplete_or_nonfinite_actions():
    response = valid_response()
    del response["action.control_mode"]
    with pytest.raises(ValueError, match="missing action fields"):
        groot_plugin(response).infer(
            groot_plugin(valid_response()).prepare(observation(), "close the lid")
        )

    response = valid_response()
    response["action.base_motion"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        groot_plugin(response).infer(
            groot_plugin(valid_response()).prepare(observation(), "close the lid")
        )


def test_compatibility_matrix_rejects_unsupported_pair():
    with pytest.raises(ValueError, match="does not support"):
        backend_registry.require_compatible("libero", "groot")
