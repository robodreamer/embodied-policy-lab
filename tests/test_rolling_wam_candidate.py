import pytest

from showcase import backend_registry
from showcase.rolling_wam_candidate import (
    ADMISSION_GATES,
    ALIASES,
    OUT_OF_SCOPE,
    PUBLISHER_REPORT,
    SCHEDULE,
    SIMULATOR_NOTES,
    UPSTREAM,
    ready_to_register,
)


def test_rolling_wam_stays_out_of_the_registry_until_admission():
    assert ready_to_register() is False
    assert all(not gate.satisfied for gate in ADMISSION_GATES)
    registered = "rollingwam" in backend_registry.POLICIES
    assert registered is ready_to_register()
    for alias in ALIASES:
        assert alias not in backend_registry.MODEL_ALIASES
        with pytest.raises(ValueError, match="Unknown model"):
            backend_registry.get_policy(alias)


def test_default_schedule_matches_the_paper_and_does_not_invent_action_width():
    assert SCHEDULE.denoising_steps_per_chunk == 10
    assert SCHEDULE.window_chunks == 5
    assert SCHEDULE.actions_per_executed_chunk == 16
    assert SCHEDULE.steady_state_denoising_steps == 2
    assert SCHEDULE.horizon_actions == 80
    assert SCHEDULE.guidance_scale == 1.0
    assert SCHEDULE.noise_shift_rho == 5.0
    assert SIMULATOR_NOTES["libero"].action_dimension is None
    assert SIMULATOR_NOTES["libero"].image_size is None
    assert SIMULATOR_NOTES["robotwin"].action_dimension is None
    assert SIMULATOR_NOTES["robotwin"].image_size == (384, 320)
    assert "robocasa" not in SIMULATOR_NOTES
    assert "robocasa" in OUT_OF_SCOPE
    assert "unitree_g1" in OUT_OF_SCOPE


def test_publisher_numbers_are_not_recorded_as_lab_results():
    assert UPSTREAM.code_released is False
    assert UPSTREAM.checkpoints_released is False
    assert UPSTREAM.commit == "9dbe8abcf36cbd11db2fc8d392a27c4a8a2a9c9a"
    assert PUBLISHER_REPORT["lab_measured"] is False
    assert PUBLISHER_REPORT["libero_average_success_percent"] == 98.1
    assert PUBLISHER_REPORT["robotwin_average_success_percent"] == 93.3
    assert PUBLISHER_REPORT["rolling_replan_ms"] == 215
