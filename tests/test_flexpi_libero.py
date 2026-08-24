import numpy as np
import pytest

from showcase import backend_registry
from showcase.flexpi_policy import axis_angle_to_quaternion, format_prompt
from showcase.libero_policy_plugins import (
    FlexPiLiberoClient,
    _uint16_payload,
    decode_uint16_payload,
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


def test_depth_transport_is_lossless_uint16_millimetres():
    depth = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    payload = _uint16_payload(depth, name="external_depth")
    decoded = decode_uint16_payload(payload, name="external_depth")
    assert decoded.dtype == np.uint16
    assert np.array_equal(decoded, depth)


def test_released_wrist_top_composite_layout():
    external = np.full((256, 256, 3), (10, 20, 30), dtype=np.uint8)
    wrist = np.full((256, 256, 3), (100, 110, 120), dtype=np.uint8)
    composite = FlexPiLiberoClient.compose_preview(external, wrist)
    assert composite.shape == (448, 512, 3)
    assert np.array_equal(composite[:288], np.full((288, 512, 3), (100, 110, 120)))
    assert np.array_equal(composite[288:, :256], np.full((160, 256, 3), (10, 20, 30)))
    assert not composite[288:, 256:].any()


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
