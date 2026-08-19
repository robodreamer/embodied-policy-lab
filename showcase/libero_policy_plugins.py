"""Policy clients for the shared LIBERO rollout engines.

The simulator process intentionally knows only this small interface.  π0.5
keeps its native websocket transport; Fast-WAM uses a dependency-free local
HTTP protocol so its pinned PyTorch runtime stays isolated from the LIBERO
client environment.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Protocol

import numpy as np

try:
    from . import backend_registry
except ImportError:  # Direct script execution adds showcase/ to sys.path.
    import backend_registry


class LiberoPolicyClient(Protocol):
    spec: backend_registry.PolicySpec
    profile: backend_registry.ProfileSpec

    @property
    def endpoint(self) -> str: ...

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...


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


class Pi05LiberoClient:
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

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._client.infer(observation)


class FastWamLiberoClient:
    def __init__(self, host: str, port: int, timeout_seconds: float = 600.0):
        self.spec = backend_registry.get_policy("fastwam")
        self.profile = backend_registry.get_profile("libero", "fastwam")
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
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


def create_libero_policy_client(model: str, host: str, port: int) -> LiberoPolicyClient:
    key = backend_registry.normalize_model(model)
    backend_registry.require_compatible("libero", key)
    if key == "pi05":
        return Pi05LiberoClient(host, port)
    if key == "fastwam":
        return FastWamLiberoClient(host, port)
    raise AssertionError(f"Compatible LIBERO model has no client: {key}")
