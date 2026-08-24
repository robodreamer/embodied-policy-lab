"""Policy clients for the shared LIBERO rollout engines.

The simulator process intentionally knows only this small interface.  π0.5
keeps its native websocket transport; Fast-WAM and Flex-π use a
dependency-free local HTTP protocol so their pinned PyTorch runtimes stay
isolated from the simulator client.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Protocol

import numpy as np
from PIL import Image

try:
    from . import backend_registry
    from .flexpi_contracts import (
        COMPOSITE_HEIGHT,
        COMPOSITE_WIDTH,
        FLEXPI_MODES,
        normalize_flexpi_mode,
    )
except ImportError:  # Direct script execution adds showcase/ to sys.path.
    import backend_registry
    from flexpi_contracts import (
        COMPOSITE_HEIGHT,
        COMPOSITE_WIDTH,
        FLEXPI_MODES,
        normalize_flexpi_mode,
    )


class LiberoPolicyClient(Protocol):
    spec: backend_registry.PolicySpec
    profile: backend_registry.ProfileSpec
    model_image_width: int
    model_image_height: int

    @property
    def endpoint(self) -> str: ...

    @property
    def requires_depth(self) -> bool: ...

    @property
    def available_modes(self) -> tuple[dict[str, str], ...]: ...

    @property
    def mode(self) -> str: ...

    def set_mode(self, mode: str) -> None: ...

    def prepare_observation(
        self,
        external: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        prompt: str,
        depth: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]: ...

    def infer(
        self,
        observation: dict[str, Any],
        *,
        include_prediction_frames: bool = False,
    ) -> dict[str, Any]: ...


def _uint8_payload(value: Any, *, name: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"{name} must be an HxWx3 uint8 image, got {array.shape} {array.dtype}"
        )
    return {
        "shape": list(array.shape),
        "dtype": "uint8",
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_uint8_payload(payload: dict[str, Any], *, name: str) -> np.ndarray:
    """Decode one transport image; public for the server and contract tests."""

    if payload.get("dtype") != "uint8":
        raise ValueError(f"{name} has unsupported dtype {payload.get('dtype')!r}")
    shape = tuple(int(value) for value in payload.get("shape", ()))
    if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
        raise ValueError(f"{name} has invalid shape {shape}")
    raw = base64.b64decode(payload["data"], validate=True)
    expected = int(np.prod(shape))
    if len(raw) != expected:
        raise ValueError(f"{name} contains {len(raw)} bytes; expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def _uint16_payload(value: Any, *, name: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype != np.uint16 or array.ndim != 2:
        raise ValueError(
            f"{name} must be an HxW uint16 depth map, got {array.shape} {array.dtype}"
        )
    return {
        "shape": list(array.shape),
        "dtype": "uint16",
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_uint16_payload(payload: dict[str, Any], *, name: str) -> np.ndarray:
    """Decode one lossless millimetre-depth transport payload."""

    if payload.get("dtype") != "uint16":
        raise ValueError(f"{name} has unsupported dtype {payload.get('dtype')!r}")
    shape = tuple(int(value) for value in payload.get("shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"{name} has invalid shape {shape}")
    raw = base64.b64decode(payload["data"], validate=True)
    expected = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    if len(raw) != expected:
        raise ValueError(f"{name} contains {len(raw)} bytes; expected {expected}")
    return np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()


def _resize_with_pad(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Exact local equivalent of OpenPI's PIL resize-with-pad preprocessing."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 image, got {array.shape} {array.dtype}")
    source = Image.fromarray(array)
    ratio = max(source.width / size, source.height / size)
    resized = source.resize(
        (max(1, int(source.width / ratio)), max(1, int(source.height / ratio))),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    canvas.paste(
        resized,
        (int((size - resized.width) / 2), int((size - resized.height) / 2)),
    )
    return np.asarray(canvas, dtype=np.uint8)


def _base_observation(
    external: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (8,) or not np.isfinite(state_array).all():
        raise ValueError(f"LIBERO state must be finite [8], got {state_array.shape}")
    return {
        "observation/image": np.ascontiguousarray(external),
        "observation/wrist_image": np.ascontiguousarray(wrist),
        "observation/state": state_array,
        "prompt": str(prompt),
    }


class _FixedModeClient:
    requires_depth = False
    model_image_width = 224
    model_image_height = 224
    available_modes: tuple[dict[str, str], ...] = ()
    mode = "action-only"

    def set_mode(self, mode: str) -> None:
        if mode not in ("", "action-only"):
            raise ValueError(f"{self.spec.display_name} does not expose inference modes")


class Pi05LiberoClient(_FixedModeClient):
    def __init__(self, host: str, port: int):
        from openpi_client import websocket_client_policy

        self.spec = backend_registry.get_policy("pi05")
        self.profile = backend_registry.get_profile("libero", "pi05")
        self.host = host
        self.port = port
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)

    @property
    def endpoint(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def prepare_observation(
        self, external, wrist, state, prompt, depth=None
    ) -> dict[str, Any]:
        del depth
        return _base_observation(
            _resize_with_pad(external), _resize_with_pad(wrist), state, prompt
        )

    def infer(
        self,
        observation: dict[str, Any],
        *,
        include_prediction_frames: bool = False,
    ) -> dict[str, Any]:
        del include_prediction_frames
        return self._client.infer(observation)


class FastWamLiberoClient(_FixedModeClient):
    def __init__(self, host: str, port: int, timeout_seconds: float = 600.0):
        self.spec = backend_registry.get_policy("fastwam")
        self.profile = backend_registry.get_profile("libero", "fastwam")
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def prepare_observation(
        self, external, wrist, state, prompt, depth=None
    ) -> dict[str, Any]:
        del depth
        # Preserve the already validated Fast-WAM branch contract: the
        # simulator client sends OpenPI-style 224-square views and the server
        # composes them without another effective resize.
        return _base_observation(
            _resize_with_pad(external), _resize_with_pad(wrist), state, prompt
        )

    def infer(
        self,
        observation: dict[str, Any],
        *,
        include_prediction_frames: bool = False,
    ) -> dict[str, Any]:
        del include_prediction_frames
        required = (
            "observation/image",
            "observation/wrist_image",
            "observation/state",
            "prompt",
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise ValueError(f"Fast-WAM observation is missing fields: {missing}")
        state = np.asarray(observation["observation/state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"Fast-WAM state must be finite [8], got {state.shape}")
        payload = {
            "schema_version": 1,
            "prompt": str(observation["prompt"]),
            "state": state.tolist(),
            "external": _uint8_payload(
                observation["observation/image"], name="external"
            ),
            "wrist": _uint8_payload(
                observation["observation/wrist_image"], name="wrist"
            ),
        }
        request = urllib.request.Request(
            f"{self.endpoint}/infer",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Fast-WAM server returned HTTP {error.code}: {detail}"
            ) from error
        if "actions" not in result:
            raise ValueError("Fast-WAM response is missing the 'actions' field")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (self.profile.action_horizon, 7):
            raise ValueError(
                "Fast-WAM action shape mismatch: "
                f"expected {(self.profile.action_horizon, 7)}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Fast-WAM returned a non-finite action")
        result["actions"] = actions
        return result


class FlexPiLiberoClient:
    requires_depth = True
    model_image_width = COMPOSITE_WIDTH
    model_image_height = COMPOSITE_HEIGHT
    available_modes = FLEXPI_MODES

    def __init__(
        self,
        host: str,
        port: int,
        timeout_seconds: float = 900.0,
        mode: str = "full-joint",
    ):
        self.spec = backend_registry.get_policy("flexpi")
        self.profile = backend_registry.get_profile("libero", "flexpi")
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.mode = normalize_flexpi_mode(mode)

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def set_mode(self, mode: str) -> None:
        self.mode = normalize_flexpi_mode(mode)

    def prepare_observation(
        self, external, wrist, state, prompt, depth=None
    ) -> dict[str, Any]:
        if depth is None:
            raise ValueError("Flex-π requires aligned agent and wrist depth maps")
        result = _base_observation(external, wrist, state, prompt)
        result["observation/depth"] = {
            "external": np.asarray(depth["external"], dtype=np.uint16),
            "wrist": np.asarray(depth["wrist"], dtype=np.uint16),
        }
        return result

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Flex-π server returned HTTP {error.code}: {detail}"
            ) from error

    def compose_preview(self, external: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        """Ask the pinned Flex-π runtime for its exact upstream input composite."""

        result = self._post(
            "/preprocess",
            {
                "schema_version": 1,
                "external": _uint8_payload(external, name="external"),
                "wrist": _uint8_payload(wrist, name="wrist"),
            },
        )
        return decode_uint8_payload(result["composite"], name="composite")

    def infer(
        self,
        observation: dict[str, Any],
        *,
        include_prediction_frames: bool = False,
    ) -> dict[str, Any]:
        required = (
            "observation/image",
            "observation/wrist_image",
            "observation/depth",
            "observation/state",
            "prompt",
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise ValueError(f"Flex-π observation is missing fields: {missing}")
        state = np.asarray(observation["observation/state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"Flex-π state must be finite [8], got {state.shape}")
        depth = observation["observation/depth"]
        payload = {
            "schema_version": 1,
            "mode": self.mode,
            "include_prediction_frames": bool(include_prediction_frames),
            "prompt": str(observation["prompt"]),
            "state": state.tolist(),
            "external": _uint8_payload(
                observation["observation/image"], name="external"
            ),
            "wrist": _uint8_payload(
                observation["observation/wrist_image"], name="wrist"
            ),
            "external_depth": _uint16_payload(
                depth["external"], name="external_depth"
            ),
            "wrist_depth": _uint16_payload(depth["wrist"], name="wrist_depth"),
        }
        result = self._post("/infer", payload)
        if "actions" not in result:
            raise ValueError("Flex-π response is missing the 'actions' field")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (self.profile.action_horizon, 7):
            raise ValueError(
                "Flex-π action shape mismatch: "
                f"expected {(self.profile.action_horizon, 7)}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Flex-π returned a non-finite action")
        result["actions"] = actions
        prediction = result.get("prediction")
        if isinstance(prediction, dict):
            encoded_input = prediction.pop("input_frame", None)
            if isinstance(encoded_input, dict):
                result["prediction_input_frame"] = decode_uint8_payload(
                    encoded_input, name="prediction_input_frame"
                )
            encoded_frames = prediction.pop("frames", [])
            result["prediction_frames"] = [
                decode_uint8_payload(item, name="prediction_frame")
                for item in encoded_frames
                if isinstance(item, dict)
            ]
        return result


def create_libero_policy_client(
    model: str,
    host: str,
    port: int,
    *,
    flexpi_mode: str = "full-joint",
) -> LiberoPolicyClient:
    key = backend_registry.normalize_model(model)
    backend_registry.require_compatible("libero", key)
    if key == "pi05":
        return Pi05LiberoClient(host, port)
    if key == "fastwam":
        return FastWamLiberoClient(host, port)
    if key == "flexpi":
        return FlexPiLiberoClient(host, port, mode=flexpi_mode)
    raise AssertionError(f"Compatible LIBERO model has no client: {key}")
