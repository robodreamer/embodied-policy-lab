"""Policy adapters for the shared RoboCasa rollout engine.

Each adapter translates native RoboCasa observations into one model's wire
contract and translates its response back to the canonical ``[horizon, 12]``
RoboCasa action matrix.  Adding another RoboCasa policy should only require a
new class here plus registry/launcher metadata.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

import numpy as np

try:
    from . import backend_registry
    from . import robocasa_contracts as contracts
except ImportError:  # Direct script execution adds showcase/ to sys.path.
    import backend_registry
    import robocasa_contracts as contracts


GR00T_VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
GR00T_STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
)
GR00T_ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)
GR00T_LANGUAGE_KEY = "annotation.human.task_description"


@dataclasses.dataclass(frozen=True)
class PreparedRequest:
    robot_state: np.ndarray
    payload: dict[str, Any]
    prompt: str


class RoboCasaPolicyPlugin(Protocol):
    spec: backend_registry.PolicySpec
    profile: backend_registry.ProfileSpec

    @property
    def endpoint(self) -> str: ...

    def prepare(self, observation: dict, prompt: str) -> PreparedRequest: ...

    def infer(self, request: PreparedRequest) -> np.ndarray: ...


class Pi05PolicyPlugin:
    def __init__(self, host: str, port: int, resize_size: int = 224):
        from openpi_client import image_tools, websocket_client_policy

        self.spec = backend_registry.get_policy("pi05")
        self.profile = backend_registry.get_profile("robocasa", "pi05")
        self.host = host
        self.port = port
        self.resize_size = resize_size
        self._image_tools = image_tools
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)

    @property
    def endpoint(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def prepare(self, observation: dict, prompt: str) -> PreparedRequest:
        images = [
            self._image_tools.convert_to_uint8(
                self._image_tools.resize_with_pad(
                    np.ascontiguousarray(observation[key]),
                    self.resize_size,
                    self.resize_size,
                )
            )
            for key in contracts.CAMERA_KEYS
        ]
        robot_state = contracts.robot_state_from_observation(observation)
        payload = {
            "observation/image": images[0],
            "observation/wrist_image": images[1],
            "observation/right_image": images[2],
            "observation/state": robot_state,
            "prompt": str(prompt),
        }
        return PreparedRequest(robot_state, payload, str(prompt))

    def infer(self, request: PreparedRequest) -> np.ndarray:
        response = self._client.infer(request.payload)
        if "actions" not in response:
            raise ValueError("π0.5 response is missing the 'actions' field")
        return contracts.validate_action_chunk(response["actions"])


class GrootN15PolicyPlugin:
    def __init__(self, host: str, port: int, resize_size: int = 224):
        del resize_size  # GR00T's checkpoint transform owns crop/resize.
        from gr00t.eval.robot import RobotInferenceClient

        self.spec = backend_registry.get_policy("groot-n1.5")
        self.profile = backend_registry.get_profile("robocasa", "groot-n1.5")
        self.host = host
        self.port = port
        self._client = RobotInferenceClient(host=host, port=port)

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def prepare(self, observation: dict, prompt: str) -> PreparedRequest:
        missing = [
            key
            for key in (*GR00T_VIDEO_KEYS, *GR00T_STATE_KEYS)
            if key not in observation
        ]
        if missing:
            raise ValueError(f"RoboCasa observation is missing GR00T fields: {missing}")
        payload = {
            key: np.expand_dims(np.ascontiguousarray(observation[key]), axis=0)
            for key in (*GR00T_VIDEO_KEYS, *GR00T_STATE_KEYS)
        }
        # The N1.5 policy's unbatched helper expands list values alongside the
        # one-frame video/state histories. A bare string becomes a 0-D NumPy
        # array inside the server and cannot be indexed by its batch transform.
        payload[GR00T_LANGUAGE_KEY] = [str(prompt)]
        return PreparedRequest(
            contracts.robot_state_from_observation(observation), payload, str(prompt)
        )

    def infer(self, request: PreparedRequest) -> np.ndarray:
        response = self._client.get_action(request.payload)
        missing = [key for key in GR00T_ACTION_KEYS if key not in response]
        if missing:
            raise ValueError(f"GR00T response is missing action fields: {missing}")
        components = []
        horizon = None
        for key in GR00T_ACTION_KEYS:
            value = np.asarray(response[key])
            if value.ndim == 1:
                value = value[:, None]
            if value.ndim != 2:
                raise ValueError(f"GR00T action {key!r} has invalid shape {value.shape}")
            horizon = len(value) if horizon is None else horizon
            if len(value) != horizon:
                raise ValueError("GR00T action components have different horizons")
            components.append(value)
        return contracts.validate_action_chunk(np.concatenate(components, axis=1))


def create_policy_plugin(
    model: str, host: str, port: int, resize_size: int = 224
) -> RoboCasaPolicyPlugin:
    key = backend_registry.normalize_model(model)
    backend_registry.require_compatible("robocasa", key)
    if key == "pi05":
        return Pi05PolicyPlugin(host, port, resize_size)
    if key == "groot-n1.5":
        return GrootN15PolicyPlugin(host, port, resize_size)
    raise AssertionError(f"Compatible RoboCasa model has no plugin: {key}")
