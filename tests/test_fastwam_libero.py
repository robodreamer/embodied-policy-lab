import numpy as np
import pytest

from showcase import backend_registry
from showcase.fastwam_policy import combine_cameras, format_prompt
from showcase.libero_policy_plugins import _uint8_payload, decode_uint8_payload


def test_fastwam_profile_matches_released_libero_contract():
    policy = backend_registry.get_policy("fast-wam")
    profile = backend_registry.get_profile("libero", "fastwam")
    assert policy.transport == "http"
    assert profile.action_horizon == 32
    assert profile.default_replan_steps == 10
    with pytest.raises(ValueError, match="does not support"):
        backend_registry.get_profile("robocasa", "fastwam")


def test_camera_composition_is_external_then_wrist_and_transport_is_lossless():
    external = np.full((256, 256, 3), (1, 2, 3), dtype=np.uint8)
    wrist = np.full((224, 224, 3), (101, 102, 103), dtype=np.uint8)
    payload = _uint8_payload(external, name="external")
    decoded = decode_uint8_payload(payload, name="external")
    assert np.array_equal(decoded, external)
    combined = combine_cameras(decoded, wrist)
    assert combined.shape == (224, 448, 3)
    assert np.array_equal(combined[:, :224], np.full((224, 224, 3), (1, 2, 3)))
    assert np.array_equal(combined[:, 224:], np.full((224, 224, 3), (101, 102, 103)))


def test_prompt_is_wrapped_exactly_once():
    task = "pick up the black bowl"
    wrapped = format_prompt(task)
    assert wrapped.endswith(task)
    assert format_prompt(wrapped) == wrapped
