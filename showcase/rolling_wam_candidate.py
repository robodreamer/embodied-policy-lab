"""Unpublished Rolling-WAM candidate.

This module records the paper contract that a future adapter must satisfy.
It is not a policy profile: ``showcase.backend_registry`` must not list
Rolling-WAM until ``ready_to_register`` is true. Publisher success rates and
latencies are citations, not lab measurements.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class UpstreamRecord:
    repository: str
    commit: str
    commit_date: str
    license: str
    code_released: bool
    checkpoints_released: bool


@dataclasses.dataclass(frozen=True)
class RollingSchedule:
    """Default schedule from arXiv:2609.30247v1, Section IV-B."""

    denoising_steps_per_chunk: int
    window_chunks: int
    actions_per_executed_chunk: int
    guidance_scale: float
    noise_shift_rho: float

    @property
    def steady_state_denoising_steps(self) -> int:
        return self.denoising_steps_per_chunk // self.window_chunks

    @property
    def horizon_actions(self) -> int:
        return self.window_chunks * self.actions_per_executed_chunk


@dataclasses.dataclass(frozen=True)
class SimulatorNote:
    cameras_stated: tuple[str, ...]
    action_dimension: int | None
    image_size: tuple[int, int] | None
    detail: str


@dataclasses.dataclass(frozen=True)
class AdmissionGate:
    key: str
    satisfied: bool
    detail: str


PAPER = {
    "title": "Rolling-WAM: World Action Models with Rolling Imagination",
    "arxiv": "2609.30247v1",
    "submitted": "2026-09-24",
    "license": "CC BY 4.0",
    "project_page": "https://rolling-wam.github.io/",
}

UPSTREAM = UpstreamRecord(
    repository="https://github.com/zyinghua/Rolling-WAM",
    commit="9dbe8abcf36cbd11db2fc8d392a27c4a8a2a9c9a",
    commit_date="2026-09-25",
    license="Apache-2.0",
    code_released=False,
    checkpoints_released=False,
)

# The inspected upstream commit contains the README, license, and assets only.
SCHEDULE = RollingSchedule(
    denoising_steps_per_chunk=10,
    window_chunks=5,
    actions_per_executed_chunk=16,
    guidance_scale=1.0,
    noise_shift_rho=5.0,
)

SIMULATOR_NOTES = {
    "libero": SimulatorNote(
        cameras_stated=("external", "wrist"),
        action_dimension=None,
        image_size=None,
        detail=(
            "One policy across Spatial, Object, Goal, and Long. "
            "The paper states external and wrist RGB, 50 rollouts per task, "
            "and does not state image size, state ordering, or action dimension."
        ),
    ),
    "robotwin": SimulatorNote(
        cameras_stated=("head", "wrist", "wrist"),
        action_dimension=None,
        image_size=(384, 320),
        detail=(
            "One policy across 50 tasks in Clean and Randomized, "
            "100 rollouts per task per setting. The 384×320 size is stated "
            "for the A100 replanning-latency setting, not as a released "
            "preprocessing spec. Action dimension is not stated."
        ),
    ),
}

OUT_OF_SCOPE = {
    "robocasa": "The paper does not evaluate RoboCasa.",
    "unitree_g1": (
        "Real-world only: one 320×224 egocentric RGB view, 43D state, "
        "and 78D actions (64D SONIC motion latent plus 7D per hand). "
        "This is not a LIBERO, RoboCasa, or RoboTwin contract."
    ),
}

# Cited from the paper. Do not copy these into results/ or describe them as
# a local or matched benchmark.
PUBLISHER_REPORT = {
    "lab_measured": False,
    "libero_average_success_percent": 98.1,
    "libero_rollouts_per_task": 50,
    "robotwin_average_success_percent": 93.3,
    "robotwin_rollouts_per_task_per_setting": 100,
    "latency_hardware": "NVIDIA A100",
    "latency_setting": "RoboTwin 2.0, 384×320, execute 16 actions, N=10, W=5",
    "rolling_replan_ms": 215,
    "joint_wam_replan_ms": 978,
    "fast_wam_replan_ms": 548,
}

ALIASES = ("rollingwam", "rolling-wam", "rolling_wam")

ADMISSION_GATES = (
    AdmissionGate(
        "inference_code",
        False,
        "Upstream commit has no sampler, server, or simulator client.",
    ),
    AdmissionGate(
        "checkpoint_identity",
        False,
        "No checkpoint, statistics file, or digest has been published.",
    ),
    AdmissionGate(
        "action_and_image_contract",
        False,
        "LIBERO and RoboTwin action dimension, normalization, state ordering, "
        "and LIBERO image size are not stated.",
    ),
    AdmissionGate(
        "stateful_runtime",
        False,
        "Steady-state replanning keeps a partially denoised video-action window. "
        "A stateless per-request action client cannot implement that schedule.",
    ),
    AdmissionGate(
        "local_wiring_check",
        False,
        "No audited local request or short rollout exists.",
    ),
)


def ready_to_register() -> bool:
    """True only when every admission gate has been satisfied."""
    return all(gate.satisfied for gate in ADMISSION_GATES)
