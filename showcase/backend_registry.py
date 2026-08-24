"""Dependency-free simulator, policy, and pairing metadata.

Model identity/runtime is separate from simulator-specific policy profiles.
Action horizons and checkpoints belong to a pairing: π0.5, for example, uses
different heads and checkpoints in LIBERO and RoboCasa.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SimulatorSpec:
    key: str
    display_name: str
    simulator: str
    state_dimension: int
    action_dimension: int
    cameras: tuple[str, ...]
    task_collection: str


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    key: str
    display_name: str
    runtime: str
    transport: str
    runtime_directory: str


@dataclasses.dataclass(frozen=True)
class ProfileSpec:
    backend: str
    policy: str
    model_name: str
    checkpoint: str
    action_horizon: int
    default_replan_steps: int


SIMULATORS = {
    "libero": SimulatorSpec(
        key="libero",
        display_name="LIBERO",
        simulator="LIBERO / robosuite / MuJoCo",
        state_dimension=8,
        action_dimension=7,
        cameras=("agentview", "eye_in_hand"),
        task_collection="libero_spatial",
    ),
    "robocasa": SimulatorSpec(
        key="robocasa",
        display_name="RoboCasa365",
        simulator="RoboCasa365 / robosuite / MuJoCo",
        state_dimension=16,
        action_dimension=12,
        cameras=(
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ),
        task_collection="atomic_seen",
    ),
}


POLICIES = {
    "pi05": PolicySpec(
        key="pi05",
        display_name="Physical Intelligence π0.5",
        runtime="local JAX/CUDA",
        transport="websocket",
        runtime_directory="upstream-openpi",
    ),
    "groot-n1.5": PolicySpec(
        key="groot-n1.5",
        display_name="NVIDIA Isaac GR00T N1.5",
        runtime="local PyTorch/CUDA",
        transport="zeromq",
        runtime_directory="upstream-robocasa-groot",
    ),
    "fastwam": PolicySpec(
        key="fastwam",
        display_name="Fast-WAM",
        runtime="local PyTorch/CUDA · staged 24 GB profile",
        transport="http",
        runtime_directory="../upstream-fastwam",
    ),
    "flexpi": PolicySpec(
        key="flexpi",
        display_name="Flex-π",
        runtime="local PyTorch/CUDA · flexible world-action profile",
        transport="http",
        runtime_directory="../upstream-flexpi",
    ),
}


PROFILES = {
    ("libero", "pi05"): ProfileSpec(
        backend="libero",
        policy="pi05",
        model_name="pi05_libero",
        checkpoint="gs://openpi-assets/checkpoints/pi05_libero",
        action_horizon=10,
        default_replan_steps=5,
    ),
    ("libero", "fastwam"): ProfileSpec(
        backend="libero",
        policy="fastwam",
        model_name="fastwam_libero_uncond_2cam224",
        checkpoint="checkpoints/fastwam_release/libero_uncond_2cam224.pt",
        action_horizon=32,
        default_replan_steps=10,
    ),
    ("libero", "flexpi"): ProfileSpec(
        backend="libero",
        policy="flexpi",
        model_name="flexpi_libero_stream_dropout",
        checkpoint="runs/flexpi-libero/checkpoints/weights/step_010860.pt",
        action_horizon=32,
        default_replan_steps=10,
    ),
    ("robocasa", "pi05"): ProfileSpec(
        backend="robocasa",
        policy="pi05",
        model_name="pi05_pretrain_human300",
        checkpoint=(
            "cache/robocasa365_checkpoints/"
            "pi05_pretrain_human300/multitask_learning/75000"
        ),
        action_horizon=50,
        default_replan_steps=5,
    ),
    ("robocasa", "groot-n1.5"): ProfileSpec(
        backend="robocasa",
        policy="groot-n1.5",
        model_name="gr00t_n1.5_robocasa365_120k",
        checkpoint=(
            "cache/robocasa365_checkpoints/gr00t_n1-5/"
            "multitask_learning/checkpoint-120000"
        ),
        action_horizon=16,
        default_replan_steps=16,
    ),
}


MODEL_ALIASES = {
    "pi": "pi05",
    "pi0.5": "pi05",
    "pi-0.5": "pi05",
    "groot": "groot-n1.5",
    "gr00t": "groot-n1.5",
    "groot-n1.5": "groot-n1.5",
    "gr00t-n1.5": "groot-n1.5",
    "groot_n1.5": "groot-n1.5",
    "gr00t_n1.5": "groot-n1.5",
    "fast-wam": "fastwam",
    "fast_wam": "fastwam",
    "flex-pi": "flexpi",
    "flex_pi": "flexpi",
    "flex-π": "flexpi",
}


def normalize_model(key: str) -> str:
    normalized = key.strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)


def get_simulator(key: str) -> SimulatorSpec:
    try:
        return SIMULATORS[key.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(SIMULATORS))
        raise ValueError(
            f"Unknown simulator backend {key!r}; choose one of: {supported}"
        ) from error


def get_policy(key: str) -> PolicySpec:
    normalized = normalize_model(key)
    try:
        return POLICIES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(POLICIES))
        raise ValueError(
            f"Unknown model {key!r}; choose one of: {supported}"
        ) from error


def get_profile(backend: str, model: str) -> ProfileSpec:
    simulator = get_simulator(backend)
    policy = get_policy(model)
    try:
        return PROFILES[(simulator.key, policy.key)]
    except KeyError as error:
        choices = ", ".join(
            model_key
            for backend_key, model_key in PROFILES
            if backend_key == simulator.key
        )
        raise ValueError(
            f"Model {policy.key!r} does not support backend {simulator.key!r}; "
            f"compatible models: {choices}"
        ) from error


def require_compatible(backend: str, model: str) -> tuple[SimulatorSpec, PolicySpec]:
    simulator = get_simulator(backend)
    policy = get_policy(model)
    get_profile(simulator.key, policy.key)
    return simulator, policy


# Backward-compatible names for callers written before policies became plugins.
BACKENDS = SIMULATORS
BackendSpec = SimulatorSpec
get_backend = get_simulator


def as_dict(spec: SimulatorSpec | PolicySpec | ProfileSpec) -> dict:
    return dataclasses.asdict(spec)
