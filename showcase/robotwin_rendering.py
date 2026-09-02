"""RoboTwin renderer compatibility helpers.

The pinned RoboTwin release hard-codes OIDN 2.0.1 for every SAPIEN scene.
That bundled CUDA denoiser fails on Blackwell GPUs, leaving noisy policy
observations and printing an error for each rendered camera. Keep the upstream
choice on older devices and redirect only Blackwell runtimes to OptiX.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

RENDER_DENOISERS = ("auto", "oidn", "optix", "none")


def resolve_render_denoiser(requested: str) -> str:
    """Resolve ``auto`` without importing SAPIEN or changing global state."""
    if requested not in RENDER_DENOISERS:
        raise ValueError(
            f"RoboTwin render denoiser must be one of {RENDER_DENOISERS}, got {requested!r}"
        )
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
        return "optix"
    return "oidn"


@contextlib.contextmanager
def override_upstream_oidn(selected: str) -> Iterator[None]:
    """Redirect RoboTwin's hard-coded OIDN choice for one scene setup."""
    if selected not in RENDER_DENOISERS[1:]:
        raise ValueError(f"render denoiser must be resolved before scene setup: {selected}")

    import sapien

    original = sapien.render.set_ray_tracing_denoiser

    def set_compatible_denoiser(requested: Any) -> None:
        original(selected if requested == "oidn" else requested)

    sapien.render.set_ray_tracing_denoiser = set_compatible_denoiser
    try:
        yield
    finally:
        sapien.render.set_ray_tracing_denoiser = original
