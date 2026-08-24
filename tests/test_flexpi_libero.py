import numpy as np
import pytest

from showcase import backend_registry
from showcase.flexpi_contracts import prediction_layout, split_prediction_frame
from showcase.flexpi_policy import axis_angle_to_quaternion, format_prompt
from showcase.libero_policy_plugins import (
    FlexPiLiberoClient,
    _uint16_payload,
    _uint8_payload,
    decode_uint16_payload,
    decode_uint8_payload,
    normalize_flexpi_mode,
)


def test_flexpi_profile_matches_released_libero_contract():
    policy = backend_registry.get_policy("flex-pi")
    profile = backend_registry.get_profile("libero", "flexpi")
    assert policy.transport == "http"
    assert profile.action_horizon == 32
    assert profile.default_replan_steps == 10
    with pytest.raises(ValueError, match="does not support"):
        backend_registry.get_profile("robocasa", "flexpi")


def test_flexpi_defaults_to_world_action_cogeneration():
    client = FlexPiLiberoClient("127.0.0.1", 8000)
    assert client.mode == "full-joint"
    assert client.available_modes[1]["display_name"] == "World-action co-generation"


def test_depth_transport_is_lossless_uint16_millimetres():
    depth = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    payload = _uint16_payload(depth, name="external_depth")
    decoded = decode_uint16_payload(payload, name="external_depth")
    assert decoded.dtype == np.uint16
    assert np.array_equal(decoded, depth)


def test_released_wrist_top_composite_layout_is_shared_and_lossless():
    composite = np.zeros((448, 512, 3), dtype=np.uint8)
    composite[:288] = (100, 110, 120)
    composite[288:, :256] = (10, 20, 30)
    layout = prediction_layout()

    external, wrist = split_prediction_frame(composite)

    assert layout["wrist"] == {
        "top": 0,
        "left": 0,
        "width": 512,
        "height": 288,
    }
    assert composite.shape == (448, 512, 3)
    assert np.array_equal(wrist, np.full((288, 512, 3), (100, 110, 120)))
    assert np.array_equal(external, np.full((160, 256, 3), (10, 20, 30)))
    assert not composite[288:, 256:].any()

    encoded = _uint8_payload(composite, name="composite")
    assert np.array_equal(
        decode_uint8_payload(encoded, name="composite"), composite
    )


def test_preview_uses_server_owned_upstream_preprocessing():
    external = np.zeros((128, 128, 3), dtype=np.uint8)
    wrist = np.ones((128, 128, 3), dtype=np.uint8)
    expected = np.full((448, 512, 3), 73, dtype=np.uint8)
    client = FlexPiLiberoClient("127.0.0.1", 8000)
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"composite": _uint8_payload(expected, name="composite")}

    client._post = fake_post
    actual = client.compose_preview(external, wrist)

    assert calls[0][0] == "/preprocess"
    assert np.array_equal(actual, expected)


def test_interactive_frame_limit_is_part_of_the_server_request():
    client = FlexPiLiberoClient("127.0.0.1", 8000)
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"actions": np.zeros((32, 7), dtype=np.float32).tolist()}

    client._post = fake_post
    observation = client.prepare_observation(
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros(8, dtype=np.float32),
        "pick up the bowl",
        {
            "external": np.zeros((16, 16), dtype=np.uint16),
            "wrist": np.zeros((16, 16), dtype=np.uint16),
        },
    )

    client.infer(
        observation,
        include_prediction_frames=True,
        prediction_frame_limit=3,
    )

    assert calls[0][0] == "/infer"
    assert calls[0][1]["prediction_frame_limit"] == 3


def test_mode_and_prompt_normalization_are_explicit():
    assert normalize_flexpi_mode("fast") == "action-only"
    assert normalize_flexpi_mode("joint") == "full-joint"
    with pytest.raises(ValueError, match="Unknown Flex-π mode"):
        normalize_flexpi_mode("mystery")
    wrapped = format_prompt("pick up the bowl")
    assert wrapped.endswith("pick up the bowl")
    assert format_prompt(wrapped) == wrapped


def test_axis_angle_conversion_uses_libero_xyzw_order():
    identity = axis_angle_to_quaternion(np.zeros(3, dtype=np.float32))
    assert np.allclose(identity, [0, 0, 0, 1])
    half_turn_x = axis_angle_to_quaternion(np.asarray([np.pi, 0, 0]))
    assert np.allclose(np.abs(half_turn_x), [1, 0, 0, 0], atol=1e-6)
