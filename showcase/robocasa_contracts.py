"""Dependency-light observation and action contracts shared by RoboCasa adapters."""

from __future__ import annotations

import numpy as np


CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_eye_in_hand",
    "video.robot0_agentview_right",
)


def robot_state_from_observation(observation: dict) -> np.ndarray:
    robot_state = np.concatenate(
        (
            observation["state.end_effector_position_relative"],
            observation["state.end_effector_rotation_relative"],
            observation["state.base_position"],
            observation["state.base_rotation"],
            observation["state.gripper_qpos"],
        ),
        axis=0,
    )
    if robot_state.shape != (16,):
        raise ValueError(f"Expected a 16D RoboCasa state; got {robot_state.shape}")
    return robot_state


def validate_action_chunk(value) -> np.ndarray:
    actions = np.asarray(value)
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError(f"Expected a [horizon, 12] action chunk; got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned non-finite RoboCasa actions")
    return actions
