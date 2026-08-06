"""Simulator and policy compatibility metadata for the local showcase.

Keep this module dependency-free: setup scripts and diagnostics import it before
any simulator environment has been installed.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class BackendSpec:
    key: str
    display_name: str
    simulator: str
    policy_name: str
    state_dimension: int
    action_dimension: int
    cameras: tuple[str, ...]
    task_collection: str
    checkpoint: str
    runtime_directory: str
    readiness: str


BACKENDS = {
    "libero": BackendSpec(
        key="libero",
        display_name="LIBERO",
        simulator="LIBERO / robosuite / MuJoCo",
        policy_name="pi05_libero",
        state_dimension=8,
        action_dimension=7,
        cameras=("agentview", "eye_in_hand"),
        task_collection="libero_spatial",
        checkpoint="gs://openpi-assets/checkpoints/pi05_libero",
        runtime_directory="upstream-openpi",
        readiness="showcase",
    ),
    "robocasa": BackendSpec(
        key="robocasa",
        display_name="RoboCasa365",
        simulator="RoboCasa365 / robosuite / MuJoCo",
        policy_name="pi05_pretrain_human300",
        state_dimension=16,
        action_dimension=12,
        cameras=(
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ),
        task_collection="atomic_seen",
        checkpoint=(
            "cache/robocasa365_checkpoints/"
            "pi05_pretrain_human300/multitask_learning/75000"
        ),
        runtime_directory="upstream-robocasa-openpi",
        readiness="showcase",
    ),
}


def get_backend(key: str) -> BackendSpec:
    try:
        return BACKENDS[key.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unknown backend {key!r}; choose one of: {supported}"
        ) from error


def as_dict(spec: BackendSpec) -> dict:
    return dataclasses.asdict(spec)
