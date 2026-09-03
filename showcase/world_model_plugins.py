"""World-model preview plugins used by the RoboCasa interactive runtime."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import pathlib
import time
from collections.abc import Callable
from typing import Any, Protocol

import imageio.v2 as imageio
import numpy as np

try:
    from . import world_model_registry
except ImportError:
    import world_model_registry


@dataclasses.dataclass(frozen=True)
class PreviewRequest:
    source_env: Any
    task_name: str
    split: str
    seed: int
    action_chunk: np.ndarray
    preview_steps: int
    width: int
    height: int
    fps: float
    artifact_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class PreviewResult:
    model_key: str
    prediction_kind: str
    action_schema: str
    previewed_steps: int
    duration_ms: float
    artifact_path: str
    live_state_sha256_before: str
    live_state_sha256_after: str
    live_state_unchanged: bool
    predicted_state_sha256: str
    branch_success: bool
    branch_terminated: bool
    caveat: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class WorldModelPlugin(Protocol):
    key: str

    def preview(self, request: PreviewRequest) -> PreviewResult: ...


def try_preview(
    plugin: WorldModelPlugin, request: PreviewRequest
) -> tuple[PreviewResult | None, Exception | None]:
    """Run a diagnostic predictor without letting it abort policy execution."""
    try:
        return plugin.preview(request), None
    except Exception as error:  # Predictors are observability, never policy gates.
        return None, error


def state_sha256(state: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(state.time, dtype=np.float64).tobytes())
    digest.update(np.asarray(state.qpos, dtype=np.float64).tobytes())
    digest.update(np.asarray(state.qvel, dtype=np.float64).tobytes())
    return digest.hexdigest()


def environment_state_sha256(env: Any) -> str:
    return state_sha256(env.unwrapped.env.sim.get_state())


def compare_states(predicted: Any, actual: Any, *, atol: float = 1e-9) -> dict:
    """Compare MuJoCo states without treating floating-point noise as drift."""
    predicted_qpos = np.asarray(predicted.qpos, dtype=np.float64)
    actual_qpos = np.asarray(actual.qpos, dtype=np.float64)
    predicted_qvel = np.asarray(predicted.qvel, dtype=np.float64)
    actual_qvel = np.asarray(actual.qvel, dtype=np.float64)
    if predicted_qpos.shape != actual_qpos.shape or predicted_qvel.shape != actual_qvel.shape:
        return {
            "within_tolerance": False,
            "absolute_tolerance": atol,
            "shape_match": False,
            "time_error": None,
            "max_qpos_error": None,
            "max_qvel_error": None,
        }
    time_error = abs(float(predicted.time) - float(actual.time))
    qpos_error = float(np.max(np.abs(predicted_qpos - actual_qpos), initial=0.0))
    qvel_error = float(np.max(np.abs(predicted_qvel - actual_qvel), initial=0.0))
    return {
        "within_tolerance": bool(
            time_error <= atol and qpos_error <= atol and qvel_error <= atol
        ),
        "absolute_tolerance": atol,
        "shape_match": True,
        "time_error": time_error,
        "max_qpos_error": qpos_error,
        "max_qvel_error": qvel_error,
    }


class SimulatorCounterfactualPlugin:
    """Replay an action prefix in a separate RoboCasa simulator oracle.

    Only MuJoCo state is copied into the branch. The live environment is never
    stepped or restored, which makes the non-mutation check useful evidence and
    avoids contaminating the rollout with preview actions.
    """

    key = "robocasa-sim"

    def __init__(
        self,
        create_environment: Callable[[str, str, int], Any],
        render_frames: Callable[..., tuple[np.ndarray, ...]],
        step_environment: Callable[[Any, np.ndarray], tuple],
        resume_environment: Callable[[Any], None] | None = None,
        suspend_environment: Callable[[Any], None] | None = None,
        restore_source_environment: Callable[[Any], None] | None = None,
    ):
        self._create_environment = create_environment
        self._render_frames = render_frames
        self._step_environment = step_environment
        self._resume_environment = resume_environment or (lambda env: None)
        self._suspend_environment = suspend_environment or (lambda env: None)
        self._restore_source_environment = restore_source_environment or (
            lambda env: None
        )
        self._branch = None
        self._branch_identity: tuple[str, str, int] | None = None
        self._last_predicted_state = None

    def close(self) -> None:
        if self._branch is not None:
            self._branch.close()
            self._branch = None
            self._branch_identity = None
        self._last_predicted_state = None

    def predicted_state(self) -> Any:
        if self._last_predicted_state is None:
            raise RuntimeError("No simulator prediction has completed")
        return copy.deepcopy(self._last_predicted_state)

    def attach_branch(
        self, branch: Any, *, task_name: str, split: str, seed: int
    ) -> None:
        identity = (task_name, split, seed)
        if self._branch is branch and self._branch_identity == identity:
            return
        self.close()
        self._branch = branch
        self._branch_identity = identity

    def _get_branch(self, request: PreviewRequest):
        identity = (request.task_name, request.split, request.seed)
        if self._branch is None or self._branch_identity != identity:
            self.close()
            self._branch = self._create_environment(*identity)
            self._branch.reset()
            self._branch_identity = identity
        return self._branch

    def preview(self, request: PreviewRequest) -> PreviewResult:
        actions = np.asarray(request.action_chunk)
        if actions.ndim != 2 or actions.shape[1] != 12:
            raise ValueError(
                f"Simulator preview requires [horizon, 12] actions; got {actions.shape}"
            )
        previewed_steps = min(request.preview_steps, len(actions))
        if previewed_steps < 1:
            raise ValueError("Simulator preview requires at least one action")

        live_sim = request.source_env.unwrapped.env.sim
        live_state = copy.deepcopy(live_sim.get_state())
        live_hash_before = state_sha256(live_state)
        branch = self._get_branch(request)
        try:
            self._resume_environment(branch)
        except Exception as error:
            self.close()
            try:
                self._restore_source_environment(request.source_env)
            except Exception as restore_error:
                raise restore_error from error
            raise
        frames: list[np.ndarray] = []
        branch_success = False
        branch_terminated = False
        started = time.perf_counter()
        try:
            branch_sim = branch.unwrapped.env.sim
            branch_state = branch_sim.get_state()
            if (
                np.asarray(branch_state.qpos).shape != np.asarray(live_state.qpos).shape
                or np.asarray(branch_state.qvel).shape
                != np.asarray(live_state.qvel).shape
            ):
                raise ValueError(
                    "Counterfactual environment does not match the live MuJoCo model; "
                    "construct both environments with create_environment_pair()"
                )
            branch_sim.set_state(copy.deepcopy(live_state))
            branch_sim.forward()
            branch_observation = self._branch_observation(branch)
            frames.append(
                self._render_frames(
                    branch,
                    width=request.width,
                    height=request.height,
                    observation=branch_observation,
                )[0]
            )
            for action in actions[:previewed_steps]:
                branch_observation, _, terminated, truncated, info = self._step_environment(
                    branch, action
                )
                frames.append(
                    self._render_frames(
                        branch,
                        width=request.width,
                        height=request.height,
                        observation=branch_observation,
                    )[0]
                )
                branch_success = branch_success or bool(info.get("success", False))
                branch_terminated = bool(terminated or truncated)
                if branch_terminated:
                    break
        except Exception as error:
            # A failed branch may retain partially stepped controller state. Rebuild
            # it before the next preview instead of reusing uncertain state.
            self.close()
            try:
                self._restore_source_environment(request.source_env)
            except Exception as restore_error:
                raise restore_error from error
            raise

        self._last_predicted_state = copy.deepcopy(branch.unwrapped.env.sim.get_state())
        predicted_hash = state_sha256(self._last_predicted_state)
        live_hash_after = state_sha256(live_sim.get_state())
        unchanged = live_hash_before == live_hash_after
        if not unchanged:
            self.close()
            mutation_error = RuntimeError(
                "Counterfactual preview changed the live MuJoCo state"
            )
            try:
                self._restore_source_environment(request.source_env)
            except Exception as restore_error:
                raise restore_error from mutation_error
            raise mutation_error

        try:
            self._suspend_environment(branch)
            self._restore_source_environment(request.source_env)
            request.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                request.artifact_path,
                frames,
                fps=request.fps,
                macro_block_size=1,
            )
            result = PreviewResult(
                model_key=self.key,
                prediction_kind="simulator_oracle",
                action_schema=world_model_registry.ROBOCASA_ACTION_SCHEMA,
                previewed_steps=len(frames) - 1,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
                artifact_path=str(request.artifact_path),
                live_state_sha256_before=live_hash_before,
                live_state_sha256_after=live_hash_after,
                live_state_unchanged=unchanged,
                predicted_state_sha256=predicted_hash,
                branch_success=branch_success,
                branch_terminated=branch_terminated,
                caveat=(
                    "Deterministic ground-truth MuJoCo replay for this state and "
                    "action prefix. It is an oracle baseline, not a learned world "
                    "model or safety certificate."
                ),
            )
            return result
        except Exception:
            self.close()
            raise

    @staticmethod
    def _branch_observation(branch: Any) -> dict:
        wrapper = branch.unwrapped
        raw = wrapper.env._get_observations(force_update=True)
        return wrapper.get_observation(raw)


def create_world_model_plugin(
    key: str,
    *,
    create_environment: Callable[[str, str, int], Any],
    render_frames: Callable[..., tuple[np.ndarray, ...]],
    step_environment: Callable[[Any, np.ndarray], tuple],
    resume_environment: Callable[[Any], None] | None = None,
    suspend_environment: Callable[[Any], None] | None = None,
    restore_source_environment: Callable[[Any], None] | None = None,
) -> WorldModelPlugin | None:
    spec = world_model_registry.require_world_model("robocasa", key)
    if spec.key == "none":
        return None
    if spec.key == "robocasa-sim":
        return SimulatorCounterfactualPlugin(
            create_environment=create_environment,
            render_frames=render_frames,
            step_environment=step_environment,
            resume_environment=resume_environment,
            suspend_environment=suspend_environment,
            restore_source_environment=restore_source_environment,
        )
    raise ValueError(f"No runtime plugin is implemented for {spec.key!r}")
