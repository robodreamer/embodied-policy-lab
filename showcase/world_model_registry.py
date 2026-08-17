"""Dependency-free world-model metadata and compatibility checks.

Policies propose actions. World models preview consequences. Keeping those axes
independent lets the same preview implementation inspect action chunks from
either π0.5 or GR00T without pretending that a world model is a policy or a
safety authority.
"""

from __future__ import annotations

import dataclasses


ROBOCASA_ACTION_SCHEMA = "robocasa-panda-omron-12d-v1"
ROBOCASA_MANIP_ACTION_SCHEMA = "robocasa-panda-manip-7d-v1"


@dataclasses.dataclass(frozen=True)
class WorldModelSpec:
    key: str
    display_name: str
    runtime: str
    prediction_kind: str
    compatible_backends: tuple[str, ...]
    action_schema: str | None
    available: bool
    description: str
    unavailable_reason: str | None = None


WORLD_MODELS = {
    "none": WorldModelSpec(
        key="none",
        display_name="Direct execution (no preview)",
        runtime="disabled",
        prediction_kind="none",
        compatible_backends=("libero", "robocasa"),
        action_schema=None,
        available=True,
        description="Execute each policy chunk without a counterfactual preview.",
    ),
    "robocasa-sim": WorldModelSpec(
        key="robocasa-sim",
        display_name="RoboCasa simulator branch",
        runtime="local MuJoCo branch",
        prediction_kind="simulator_counterfactual",
        compatible_backends=("robocasa",),
        action_schema=ROBOCASA_ACTION_SCHEMA,
        available=True,
        description=(
            "Clone the current MuJoCo state, execute the proposed prefix in the "
            "clone, and leave the live episode untouched until approval."
        ),
    ),
    "dino-wm-droid": WorldModelSpec(
        key="dino-wm-droid",
        display_name="DINO-WM · DROID/RoboCasa",
        runtime="isolated PyTorch worker",
        prediction_kind="learned_latent_dynamics",
        compatible_backends=("robocasa",),
        action_schema=ROBOCASA_MANIP_ACTION_SCHEMA,
        available=False,
        description=(
            "Lightweight learned latent dynamics candidate trained with a 7D "
            "DROID/RoboCasa manipulation action contract."
        ),
        unavailable_reason=(
            "The current policies emit a 12D mobile-manipulator action. A tested "
            "7D manipulation projection and goal-frame scoring worker are still required."
        ),
    ),
    "jepa-wm-droid": WorldModelSpec(
        key="jepa-wm-droid",
        display_name="JEPA-WM · DROID/RoboCasa",
        runtime="isolated PyTorch worker",
        prediction_kind="learned_latent_dynamics",
        compatible_backends=("robocasa",),
        action_schema=ROBOCASA_MANIP_ACTION_SCHEMA,
        available=False,
        description=(
            "Higher-capacity learned latent dynamics candidate trained with a 7D "
            "DROID/RoboCasa manipulation action contract."
        ),
        unavailable_reason=(
            "The 7D action projection, temporal context packing, and goal-frame "
            "scoring path must be validated before it can gate 12D live actions."
        ),
    ),
}


ALIASES = {
    "off": "none",
    "direct": "none",
    "sim": "robocasa-sim",
    "simulator": "robocasa-sim",
    "simulator-oracle": "robocasa-sim",
    "dino": "dino-wm-droid",
    "dino-wm": "dino-wm-droid",
    "jepa": "jepa-wm-droid",
    "jepa-wm": "jepa-wm-droid",
}


def normalize_world_model(key: str) -> str:
    normalized = key.strip().lower()
    return ALIASES.get(normalized, normalized)


def get_world_model(key: str) -> WorldModelSpec:
    normalized = normalize_world_model(key)
    try:
        return WORLD_MODELS[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(WORLD_MODELS))
        raise ValueError(
            f"Unknown world model {key!r}; choose one of: {supported}"
        ) from error


def require_world_model(backend: str, key: str) -> WorldModelSpec:
    spec = get_world_model(key)
    if backend not in spec.compatible_backends:
        raise ValueError(
            f"World model {spec.key!r} does not support backend {backend!r}"
        )
    if not spec.available:
        raise ValueError(
            f"World model {spec.key!r} is not runnable yet: "
            f"{spec.unavailable_reason}"
        )
    return spec


def catalog(backend: str) -> list[dict]:
    return [
        dataclasses.asdict(spec)
        for spec in WORLD_MODELS.values()
        if backend in spec.compatible_backends
    ]
