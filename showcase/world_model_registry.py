"""Dependency-free predictor metadata and compatibility checks.

Policies propose actions. Predictors preview consequences. Keeping those axes
independent lets the same preview implementation inspect action chunks from
either π0.5 or GR00T without pretending that a deterministic simulator oracle
is a learned world model or a safety authority.
"""

from __future__ import annotations

import dataclasses


ROBOCASA_ACTION_SCHEMA = "robocasa-panda-omron-12d-v1"


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

    @property
    def is_learned(self) -> bool:
        return self.prediction_kind.startswith("learned_")


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
        display_name="RoboCasa simulator oracle (baseline)",
        runtime="local MuJoCo oracle branch",
        prediction_kind="simulator_oracle",
        compatible_backends=("robocasa",),
        action_schema=ROBOCASA_ACTION_SCHEMA,
        available=True,
        description=(
            "Deterministically replay the proposed prefix in a matching MuJoCo "
            "clone, then compare it with the live execution. This is an oracle "
            "sanity-check baseline, not a learned world model."
        ),
    ),
}


ALIASES = {
    "off": "none",
    "direct": "none",
    "sim": "robocasa-sim",
    "simulator": "robocasa-sim",
    "simulator-oracle": "robocasa-sim",
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
