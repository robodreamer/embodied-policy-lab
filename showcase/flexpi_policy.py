"""Pinned Flex-π LIBERO policy adapter with action-only and joint modes.

The released Flex-π environment is intentionally kept separate from the
simulator client.  This adapter mirrors the upstream evaluator's RGB, depth,
intrinsics, proprioception, normalization, and gripper contracts while adding
a small serialized localhost-serving boundary for Embodied Policy Lab.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

try:
    from .flexpi_contracts import (
        DEFAULT_FRAME_INTERVAL_ACTIONS,
        FLEXPI_MODES,
        normalize_flexpi_mode,
        prediction_layout,
    )
except ImportError:  # Direct script execution adds showcase/ to sys.path.
    from flexpi_contracts import (
        DEFAULT_FRAME_INTERVAL_ACTIONS,
        FLEXPI_MODES,
        normalize_flexpi_mode,
        prediction_layout,
    )

EXPECTED_UPSTREAM_REVISION = "20c1b2b71ea35a415d5d47c39b04443cfadad7a1"
CHECKPOINT_REPOSITORY = "flex-pi/flexpi-libero"
CHECKPOINT_REVISION = "f853cb49331aa0ab8124cbd1e1fb3a56e07a2523"
CHECKPOINT_FILENAME = "checkpoints/weights/step_010860.pt"
CONFIG_FILENAME = "config.yaml"
STATS_FILENAME = "dataset_stats.json"
CHECKPOINT_SHA256 = "1aca314666ffdd62ca1cb5b0e0b0e5f836b68b81b1ef455ae1641ddd12386211"
CONFIG_SHA256 = "46b00bf570f63bbe3465b9e81c476c8b9874c25fa278d74499fb4d1dc34e8650"
STATS_SHA256 = "8a7a12f54844e0ea1cb009d1e7db460be38dc39c6376e425c5cd2f428ef59880"
INTRINSICS_SHA256 = "f3acb280d90b37eaafc45ed435039721a8069ddbfc70bfb9dcaa3d29bbd184f2"
PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
PROMPT_PREFIX = PROMPT_TEMPLATE.partition("{task}")[0]
ACTION_HORIZON = 32
INFERENCE_STEPS = 4
MODEL_SEED = 42
SUPPORTED_MODES = tuple(item["key"] for item in FLEXPI_MODES)


def _git_revision(path: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _verify_sha256(path: pathlib.Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"Artifact checksum mismatch for {path}: got {actual}, expected {expected}"
        )


def format_prompt(task: str) -> str:
    value = str(task).strip()
    if not value:
        raise ValueError("Flex-π requires a non-empty task instruction")
    return value if value.startswith(PROMPT_PREFIX) else PROMPT_TEMPLATE.format(task=value)


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Convert LIBERO's XYZ rotation vector to its expected XYZW quaternion."""

    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"Axis-angle rotation must have shape (3,), got {value.shape}")
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    xyz = value / angle * np.sin(angle / 2.0)
    return np.asarray((*xyz, np.cos(angle / 2.0)), dtype=np.float32)


def _rgb_payload(frame: Image.Image | np.ndarray) -> dict[str, Any]:
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame))
    array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return {
        "shape": list(array.shape),
        "dtype": "uint8",
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


class FlexPiPolicy:
    """One memory-bounded, serialized worker for the released Flex-π weights."""

    def __init__(
        self,
        *,
        upstream_root: pathlib.Path,
        checkpoint: pathlib.Path,
        config: pathlib.Path,
        stats: pathlib.Path,
        intrinsics: pathlib.Path,
        artifact_dir: pathlib.Path | None = None,
        prompt_cache_size: int = 8,
    ) -> None:
        def startup(message: str) -> None:
            print(f"FLEXPI_STARTUP: {message}", flush=True)

        startup("Validating the pinned source and release files")
        self.upstream_root = upstream_root.resolve()
        self.checkpoint = checkpoint.resolve()
        self.config_path = config.resolve()
        self.stats_path = stats.resolve()
        self.intrinsics_path = intrinsics.resolve()
        self.artifact_dir = artifact_dir.resolve() if artifact_dir else None
        self.prompt_cache_size = int(prompt_cache_size)
        self._lock = threading.Lock()
        self._request_count = 0
        self._prompt_cache: collections.OrderedDict[str, tuple[Any, Any]] = (
            collections.OrderedDict()
        )

        revision = _git_revision(self.upstream_root)
        if revision != EXPECTED_UPSTREAM_REVISION:
            raise ValueError(
                f"Flex-π upstream revision is {revision}, expected "
                f"{EXPECTED_UPSTREAM_REVISION}"
            )
        for path in (
            self.checkpoint,
            self.config_path,
            self.stats_path,
            self.intrinsics_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        for path, expected in (
            (self.checkpoint, CHECKPOINT_SHA256),
            (self.config_path, CONFIG_SHA256),
            (self.stats_path, STATS_SHA256),
            (self.intrinsics_path, INTRINSICS_SHA256),
        ):
            _verify_sha256(path, expected)
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, str(self.upstream_root / "experiments" / "libero"))
        sys.path.insert(0, str(self.upstream_root / "src"))
        sys.path.insert(0, str(self.upstream_root))
        os.environ.setdefault(
            "DIFFSYNTH_MODEL_BASE_PATH", str(self.upstream_root / "checkpoints")
        )
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        startup("Importing the Flex-pi runtime")
        import torch

        unpatched_torch_load = torch.load
        from experiments.libero import eval_libero_single as upstream_eval
        from experiments.libero.libero_utils import invert_gripper_action
        from flexpi.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )
        from hydra.utils import instantiate
        from omegaconf import OmegaConf, open_dict

        # Importing the evaluator enables unrestricted pickle loading for
        # LIBERO init-state files. This policy server never loads those files,
        # so restore a checkpoint-safe default before any model asset is read.
        def weights_only_load(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("weights_only") is False:
                raise ValueError("Flex-π policy assets must use weights_only=True")
            kwargs["weights_only"] = True
            return unpatched_torch_load(*args, **kwargs)

        torch.load = weights_only_load

        if not torch.cuda.is_available():
            raise RuntimeError("Flex-π requires a CUDA device")
        self.torch = torch
        self.upstream_eval = upstream_eval
        self.invert_gripper_action = invert_gripper_action
        self.cfg = OmegaConf.load(self.config_path)
        with open_dict(self.cfg):
            self.cfg.ckpt = str(self.checkpoint)
            self.cfg.model.load_text_encoder = True
            self.cfg.model.skip_dit_load_from_pretrain = True
            self.cfg.model.action_dit_pretrained_path = None
            self.cfg.EVALUATION = {
                "camera_intrinsics_path": str(self.intrinsics_path),
                "tshape_eval_resize": "stretch",
                "num_inference_steps": INFERENCE_STEPS,
                "replan_steps": 10,
                "negative_prompt": "",
                "text_cfg_scale": 1.0,
                "rand_device": "cpu",
                "tiled": False,
                "binarize_gripper": True,
            }

        started = time.perf_counter()
        startup("Preparing preprocessing and normalization")
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(self.stats_path))
        )
        # Instantiate and deserialize on CPU.  Excluding T5 from model.to(cuda)
        # is required to leave enough of this workstation's 24 GB VRAM for the
        # future-stream denoising path.
        startup("Building the model on CPU")
        self.model = instantiate(
            self.cfg.model, model_dtype=torch.bfloat16, device="cpu"
        )
        startup("Deserializing the 12 GB checkpoint on CPU")
        self.model.load_checkpoint(str(self.checkpoint))
        self.model.eval()
        self.model.offload_text_encoder = True
        startup("Moving denoising and visual components to CUDA; keeping T5 on CPU")
        self.model.to("cuda")
        self.model.device = torch.device("cuda")
        # Mirror the official evaluator's inference initialization. With
        # compile/quantization disabled this still installs the deterministic
        # eager attention/solver settings expected by both runtime regimes.
        if hasattr(self.model, "prepare_for_inference"):
            self.model.prepare_for_inference(
                torch_compile=False,
                quantization=None,
            )
        startup("Installing inference settings and camera intrinsics")
        intrinsics_tensor = upstream_eval._load_eval_camera_intrinsics(self.cfg)
        self.model.set_camera_intrinsics(intrinsics_tensor.to("cuda"))
        self.input_height, self.input_width = map(
            int, self.cfg.data.train.video_size
        )
        self.num_video_frames = upstream_eval._get_num_video_frames(self.cfg)
        self.depth_target_hw = upstream_eval._per_cam_depth_target_hw(self.cfg)
        self.frame_interval_actions = int(
            self.cfg.data.train.action_video_freq_ratio
        )
        if self.frame_interval_actions != DEFAULT_FRAME_INTERVAL_ACTIONS:
            raise ValueError(
                "Flex-π release frame interval drifted: "
                f"got {self.frame_interval_actions}, expected "
                f"{DEFAULT_FRAME_INTERVAL_ACTIONS}"
            )
        self.load_seconds = time.perf_counter() - started
        startup(f"Model ready after {self.load_seconds:.1f} seconds")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model": "flexpi_libero_stream_dropout",
            "upstream_revision": EXPECTED_UPSTREAM_REVISION,
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint": str(self.checkpoint),
            "config": str(self.config_path),
            "stats": str(self.stats_path),
            "intrinsics": str(self.intrinsics_path),
            "action_horizon": ACTION_HORIZON,
            "inference_steps": INFERENCE_STEPS,
            "num_video_frames": self.num_video_frames,
            "model_seed": MODEL_SEED,
            "dtype": "bfloat16",
            "text_encoder": "CPU offload with prompt-context cache",
            "supported_modes": list(SUPPORTED_MODES),
            "prediction_layout": prediction_layout(),
            "load_seconds": round(self.load_seconds, 3),
            "requests": self._request_count,
        }

    def _context_for(self, prompt: str) -> tuple[Any, Any, bool, float]:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            self._prompt_cache.move_to_end(prompt)
            return cached[0], cached[1], True, 0.0
        started = time.perf_counter()
        context, context_mask = self.model.encode_prompt(prompt)
        cached = (context.cpu(), context_mask.cpu())
        self._prompt_cache[prompt] = cached
        self._prompt_cache.move_to_end(prompt)
        while len(self._prompt_cache) > self.prompt_cache_size:
            self._prompt_cache.popitem(last=False)
        self.torch.cuda.empty_cache()
        return cached[0], cached[1], False, time.perf_counter() - started

    def _model_observation(
        self,
        external: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
    ) -> tuple[Any, Any, dict[str, Any]]:
        # The transport sends display-correct RGB.  Upstream's helper applies
        # the canonical 180-degree LIBERO correction itself, so reverse here.
        obs = {
            "agentview_image": np.ascontiguousarray(external[::-1, ::-1]),
            "robot0_eye_in_hand_image": np.ascontiguousarray(wrist[::-1, ::-1]),
            "robot0_eef_pos": state[:3],
            "robot0_eef_quat": axis_angle_to_quaternion(state[3:6]),
            "robot0_gripper_qpos": state[6:8],
        }
        image, proprio, _, per_cam = self.upstream_eval._obs_to_model_input(
            obs,
            cfg=self.cfg,
            processor=self.processor,
            width=self.input_width,
            height=self.input_height,
            device="cuda",
            dtype=self.model.torch_dtype,
        )
        if per_cam is None:
            raise RuntimeError("Released Flex-π config did not produce per-camera inputs")
        return image, proprio, per_cam

    def _tensor_to_rgb(self, image: Any) -> np.ndarray:
        tensor = image.detach()[0].float().clamp(-1.0, 1.0)
        array = (
            ((tensor.permute(1, 2, 0) + 1.0) * 127.5)
            .clamp(0, 255)
            .to(dtype=self.torch.uint8)
            .cpu()
            .numpy()
        )
        return np.ascontiguousarray(array)

    def preprocess_composite(
        self, *, external: np.ndarray, wrist: np.ndarray
    ) -> np.ndarray:
        """Return the exact normalized-then-quantized model input composite."""

        with self._lock:
            image, _, _ = self._model_observation(
                external, wrist, np.zeros(8, dtype=np.float32)
            )
            return self._tensor_to_rgb(image)

    def _depth_inputs(
        self, external_depth: np.ndarray, wrist_depth: np.ndarray
    ) -> dict[str, Any]:
        from torch.nn import functional

        def convert(value: np.ndarray, target: tuple[int, int]) -> Any:
            tensor = self.torch.from_numpy(np.asarray(value, dtype=np.uint16).astype(np.int32))
            if tuple(tensor.shape) != tuple(target):
                tensor = functional.interpolate(
                    tensor[None, None].float(), size=target, mode="nearest"
                )[0, 0].to(self.torch.int32)
            return tensor.to("cuda").reshape(1, 1, *target)

        head_target = self.depth_target_hw["cam_high"]
        wrist_target = self.depth_target_hw["cam_left_wrist"]
        head = convert(external_depth, head_target)
        left = convert(wrist_depth, wrist_target)
        return {
            "cam_high": head,
            "cam_left_wrist": left,
            "cam_right_wrist": self.torch.zeros_like(left),
        }

    def _denormalize(self, action: Any) -> np.ndarray:
        value = self.upstream_eval._denormalize_action(action, self.processor)[0]
        value[..., -1] = value[..., -1] * 2.0 - 1.0
        value = self.invert_gripper_action(value)
        value[..., -1] = np.sign(value[..., -1])
        return np.asarray(value, dtype=np.float32)

    def infer(
        self,
        *,
        external: np.ndarray,
        wrist: np.ndarray,
        external_depth: np.ndarray,
        wrist_depth: np.ndarray,
        state: np.ndarray,
        task: str,
        mode: str,
        include_prediction_frames: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            return self._infer_locked(
                external=external,
                wrist=wrist,
                external_depth=external_depth,
                wrist_depth=wrist_depth,
                state=state,
                task=task,
                mode=mode,
                include_prediction_frames=include_prediction_frames,
            )

    def _infer_locked(self, **request: Any) -> dict[str, Any]:
        started = time.perf_counter()
        mode = normalize_flexpi_mode(request["mode"])
        include_prediction_frames = bool(request.get("include_prediction_frames"))
        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"State must be finite [8], got {state.shape}")
        self._request_count += 1
        request_id = self._request_count
        prompt = format_prompt(request["task"])
        context, context_mask, cache_hit, prompt_seconds = self._context_for(prompt)
        context = context.to("cuda")
        context_mask = context_mask.to("cuda")
        image, proprio, per_cam = self._model_observation(
            request["external"], request["wrist"], state
        )
        input_composite = self._tensor_to_rgb(image) if include_prediction_frames else None
        per_cam_depth = self._depth_inputs(
            request["external_depth"], request["wrist_depth"]
        )
        kwargs = {
            "prompt": None,
            "context": context,
            "context_mask": context_mask,
            "input_image": image,
            "action_horizon": ACTION_HORIZON,
            "num_video_frames": self.num_video_frames,
            "proprio": proprio,
            "negative_prompt": "",
            "text_cfg_scale": 1.0,
            "num_inference_steps": INFERENCE_STEPS,
            "sigma_shift": None,
            "seed": MODEL_SEED,
            "rand_device": "cpu",
            "tiled": False,
            "per_cam": per_cam,
            "per_cam_depth": per_cam_depth,
        }
        self.torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        if mode == "action-only":
            prediction = self.model.infer_action(
                **kwargs,
                joint_video=False,
                joint_dino=False,
                joint_pointmap=False,
            )
            frames: list[Any] = []
        else:
            saved = (
                self.model.joint_video,
                self.model.joint_dino,
                self.model.joint_pointmap,
            )
            self.model.joint_video = True
            self.model.joint_dino = True
            self.model.joint_pointmap = True
            try:
                prediction = self.model.infer_joint(
                    **kwargs, test_action_with_infer_action=False
                )
            finally:
                (
                    self.model.joint_video,
                    self.model.joint_dino,
                    self.model.joint_pointmap,
                ) = saved
            frames = list(prediction.get("video") or [])
        self.torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        actions = self._denormalize(prediction["action"])
        if actions.shape != (ACTION_HORIZON, 7):
            raise ValueError(f"Flex-π returned invalid action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Flex-π returned non-finite actions")
        timing = {
            "prompt_seconds": round(prompt_seconds, 4),
            "prompt_cache_hit": cache_hit,
            "inference_seconds": round(inference_seconds, 4),
            "cuda_peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
        }
        result: dict[str, Any] = {
            "schema_version": 1,
            "request_id": request_id,
            "mode": mode,
            "actions": actions.tolist(),
            "timing": timing,
        }
        if frames:
            encode_started = time.perf_counter()
            result["prediction"] = {
                "kind": "joint_rgb_dino_pointmap",
                "frame_interval_actions": self.frame_interval_actions,
                "generated_frame_count": len(frames),
                "layout": prediction_layout(),
                "caveat": (
                    "Flex-π jointly denoises future visual and action tokens; this is "
                    "not an independently scored counterfactual planner."
                ),
            }
            if include_prediction_frames:
                result["prediction"]["input_frame"] = _rgb_payload(input_composite)
                result["prediction"]["frames"] = [
                    _rgb_payload(frame) for frame in frames
                ]
            timing["presentation_encode_seconds"] = round(
                time.perf_counter() - encode_started, 4
            )
        timing["total_seconds"] = round(time.perf_counter() - started, 4)
        if self.artifact_dir:
            artifact = self.artifact_dir / f"request-{request_id:06d}.json"
            artifact.write_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "mode": mode,
                        "prompt": prompt,
                        "timing": timing,
                        "action_shape": list(actions.shape),
                        "prediction_frames": len(frames),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            result["artifact"] = str(artifact)
        return result
