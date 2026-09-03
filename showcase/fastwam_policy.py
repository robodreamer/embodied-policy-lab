"""Memory-bounded Fast-WAM LIBERO policy adapter.

Fast-WAM's released checkpoint fits the local 24 GB GPU only when its
text encoder, image VAE, and Mixture-of-Transformers are staged.  This module
keeps model semantics intact: every request encodes the *current* camera pair,
while only the deterministic text embedding is cached between replans.
"""

from __future__ import annotations

import collections
import gc
import hashlib
import json
import os
import pathlib
import subprocess
import threading
import time
import types
from typing import Any

import numpy as np
from PIL import Image


EXPECTED_UPSTREAM_REVISION = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
CHECKPOINT_FILENAME = "libero_uncond_2cam224.pt"
STATS_FILENAME = "libero_uncond_2cam224_dataset_stats.json"
CHECKPOINT_SHA256 = "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
STATS_SHA256 = "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
PROMPT_PREFIX = PROMPT_TEMPLATE.partition("{task}")[0]
ACTION_HORIZON = 32
INFERENCE_STEPS = 10
MODEL_SEED = 42


def _module_bytes(module: Any) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        identity = id(tensor)
        if identity not in seen:
            seen.add(identity)
            total += tensor.numel() * tensor.element_size()
    return total


def _git_revision(path: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _git_tracked_dirty(path: pathlib.Path) -> bool:
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--ignore-submodules=none",
        ],
        cwd=path,
        text=True,
    )
    return bool(status.strip())


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


def _center_crop_resize(image: np.ndarray, size: int = 224) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 image, got {array.shape} {array.dtype}")
    pil_image = Image.fromarray(array)
    source_width, source_height = pil_image.size
    scale = max(size / source_width, size / source_height)
    resized = pil_image.resize(
        (round(source_width * scale), round(source_height * scale)),
        resample=Image.Resampling.BILINEAR,
    )
    width, height = resized.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    return np.asarray(
        resized.crop((left, top, left + size, top + size)), dtype=np.uint8
    )


def combine_cameras(external: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Apply the released two-camera preprocessing contract (224x448 RGB)."""

    primary = _center_crop_resize(external)
    wrist_view = _center_crop_resize(wrist)
    combined = np.concatenate((primary, wrist_view), axis=1)
    if combined.shape != (224, 448, 3):
        raise AssertionError(f"Unexpected combined camera shape: {combined.shape}")
    return np.ascontiguousarray(combined)


def format_prompt(task: str) -> str:
    value = str(task).strip()
    if not value:
        raise ValueError("Fast-WAM requires a non-empty task instruction")
    return (
        value if value.startswith(PROMPT_PREFIX) else PROMPT_TEMPLATE.format(task=value)
    )


def _load_config(
    upstream_root: pathlib.Path,
    checkpoint: pathlib.Path,
    stats: pathlib.Path,
) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    resolvers = {
        "eval": eval,
        "max": lambda value: max(value),
        "split": lambda value, index: value.split("/")[int(index)],
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    with initialize_config_dir(
        version_base=None, config_dir=str(upstream_root / "configs")
    ):
        return compose(
            config_name="sim_libero",
            overrides=[
                f"ckpt={checkpoint}",
                f"EVALUATION.dataset_stats_path={stats}",
                "model.load_text_encoder=true",
                "model.skip_dit_load_from_pretrain=true",
                "model.action_dit_pretrained_path=null",
            ],
        )


class StagedFastWamPolicy:
    """One serialized Fast-WAM worker using component-level CPU/GPU staging."""

    def __init__(
        self,
        upstream_root: pathlib.Path,
        checkpoint: pathlib.Path,
        stats: pathlib.Path,
        artifact_dir: pathlib.Path | None = None,
        prompt_cache_size: int = 8,
    ) -> None:
        self.upstream_root = upstream_root.resolve()
        self.checkpoint = checkpoint.resolve()
        self.stats_path = stats.resolve()
        self.artifact_dir = artifact_dir.resolve() if artifact_dir else None
        self.prompt_cache_size = prompt_cache_size
        self._lock = threading.Lock()
        self._request_count = 0
        self._prompt_cache: collections.OrderedDict[str, tuple[Any, Any]] = (
            collections.OrderedDict()
        )
        self._mot_resident = False

        revision = _git_revision(self.upstream_root)
        if revision != EXPECTED_UPSTREAM_REVISION:
            raise ValueError(
                f"Fast-WAM upstream revision is {revision}, expected "
                f"{EXPECTED_UPSTREAM_REVISION}"
            )
        if _git_tracked_dirty(self.upstream_root):
            raise ValueError(
                "Fast-WAM upstream has tracked source or submodule changes; "
                "use a clean pinned checkout for validated inference"
            )
        for path in (self.checkpoint, self.stats_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        _verify_sha256(self.checkpoint, CHECKPOINT_SHA256)
        _verify_sha256(self.stats_path, STATS_SHA256)
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault(
            "DIFFSYNTH_MODEL_BASE_PATH", str(self.upstream_root / "checkpoints")
        )
        import torch
        from fastwam.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )
        from hydra.utils import instantiate

        if not torch.cuda.is_available():
            raise RuntimeError("Fast-WAM requires a CUDA device")
        self.torch = torch
        self.cfg = _load_config(self.upstream_root, self.checkpoint, self.stats_path)
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(self.stats_path))
        )
        self.state_key = self.processor.shape_meta["state"][0]["key"]
        self.action_key = self.processor.shape_meta["action"][0]["key"]

        started = time.perf_counter()
        self.model = instantiate(
            self.cfg.model, model_dtype=torch.bfloat16, device="cpu"
        )
        self.model.load_checkpoint(str(self.checkpoint))
        self.model.eval()
        self.model.device = torch.device("cuda")
        self._original_image_encoder = self.model._encode_input_image_latents_tensor
        gc.collect()
        self.load_seconds = time.perf_counter() - started
        self.component_bytes = {
            "mot": _module_bytes(self.model.mot),
            "text_encoder": _module_bytes(self.model.text_encoder),
            "vae": _module_bytes(self.model.vae),
            "proprio_encoder": _module_bytes(self.model.proprio_encoder),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model": "fastwam_libero_uncond_2cam224",
            "upstream_revision": EXPECTED_UPSTREAM_REVISION,
            "upstream_tracked_dirty": False,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "stats": str(self.stats_path),
            "stats_sha256": STATS_SHA256,
            "action_horizon": ACTION_HORIZON,
            "inference_steps": INFERENCE_STEPS,
            "model_seed": MODEL_SEED,
            "dtype": "bfloat16",
            "staging": "text CPU/GPU on prompt miss; VAE per frame; MoT resident",
            "load_seconds": round(self.load_seconds, 3),
            "component_bytes": self.component_bytes,
            "requests": self._request_count,
        }

    def _offload_mot(self) -> None:
        if not self._mot_resident:
            return
        self.model.mot.to(device="cpu", dtype=self.torch.bfloat16)
        if self.model.proprio_encoder is not None:
            self.model.proprio_encoder.to(device="cpu", dtype=self.torch.bfloat16)
        self._mot_resident = False
        gc.collect()
        self.torch.cuda.empty_cache()

    def _load_mot(self) -> None:
        if self._mot_resident:
            return
        self.model.mot.to(device="cuda", dtype=self.torch.bfloat16)
        if self.model.proprio_encoder is not None:
            self.model.proprio_encoder.to(device="cuda", dtype=self.torch.bfloat16)
        self.model.device = self.torch.device("cuda")
        self._mot_resident = True

    def _context_for(self, prompt: str) -> tuple[Any, Any, bool, float]:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            self._prompt_cache.move_to_end(prompt)
            return cached[0], cached[1], True, 0.0

        started = time.perf_counter()
        self._offload_mot()
        try:
            self.model.text_encoder.to(device="cuda", dtype=self.torch.bfloat16)
            context, context_mask = self.model.encode_prompt(prompt)
            cached = (context.cpu(), context_mask.cpu())
        finally:
            self.model.text_encoder.to(device="cpu", dtype=self.torch.bfloat16)
            gc.collect()
            self.torch.cuda.empty_cache()
        self._prompt_cache[prompt] = cached
        self._prompt_cache.move_to_end(prompt)
        while len(self._prompt_cache) > self.prompt_cache_size:
            self._prompt_cache.popitem(last=False)
        return cached[0], cached[1], False, time.perf_counter() - started

    def _normalize_state(self, state: np.ndarray) -> Any:
        batch = {
            "state": {
                self.state_key: self.torch.as_tensor(
                    state, dtype=self.torch.float32
                ).unsqueeze(0)
            }
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self.state_key]

    def _denormalize_action(self, action: Any) -> np.ndarray:
        normalizer = self.processor.normalizer.normalizers["action"][self.action_key]
        value = action
        if value.ndim == 2:
            value = value.unsqueeze(0)
        denormalized = normalizer.backward(value.float().cpu()).numpy()[0]
        # Official Fast-WAM LIBERO convention: dataset [0,1] gripper to env
        # [-1,+1], then invert and binarize.
        denormalized[..., -1] = (denormalized[..., -1] * 2.0 - 1.0) * -1.0
        denormalized[..., -1] = np.sign(denormalized[..., -1])
        return np.asarray(denormalized, dtype=np.float32)

    def infer(
        self,
        *,
        external: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._infer_locked(
                external=external, wrist=wrist, state=state, task=task
            )

    def _infer_locked(
        self,
        *,
        external: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> dict[str, Any]:
        request_started = time.perf_counter()
        self._request_count += 1
        request_id = self._request_count
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"State must be finite [8], got {state.shape}")
        combined = combine_cameras(external, wrist)
        prompt = format_prompt(task)
        image = self.torch.from_numpy(combined.copy()).permute(2, 0, 1).unsqueeze(0)
        image = image.to(dtype=self.torch.bfloat16) * (2.0 / 255.0) - 1.0
        proprio = self._normalize_state(state)

        self.torch.cuda.reset_peak_memory_stats()
        context, context_mask, prompt_cache_hit, prompt_seconds = self._context_for(
            prompt
        )
        self._load_mot()

        image_started = time.perf_counter()
        self.model.vae.to(device="cuda", dtype=self.torch.bfloat16)
        current_latents = None
        try:
            current_latents = self._original_image_encoder(
                input_image=image.to(device="cuda", dtype=self.torch.bfloat16),
                tiled=False,
            )
        finally:
            self.model.vae.to(device="cpu", dtype=self.torch.bfloat16)
            gc.collect()
            self.torch.cuda.empty_cache()
        if current_latents is None:
            raise RuntimeError("Fast-WAM VAE did not return image latents")
        image_seconds = time.perf_counter() - image_started

        def use_current_latents(_model: Any, input_image: Any, **_: Any) -> Any:
            del input_image
            return current_latents

        self.model._encode_input_image_latents_tensor = types.MethodType(
            use_current_latents, self.model
        )
        inference_started = time.perf_counter()
        try:
            output = self.model.infer_action(
                prompt=None,
                context=context,
                context_mask=context_mask,
                input_image=image,
                action_horizon=ACTION_HORIZON,
                proprio=proprio,
                num_inference_steps=INFERENCE_STEPS,
                seed=MODEL_SEED,
                rand_device="cpu",
                tiled=False,
            )["action"]
            self.torch.cuda.synchronize()
        finally:
            self.model._encode_input_image_latents_tensor = self._original_image_encoder
        inference_seconds = time.perf_counter() - inference_started
        actions = self._denormalize_action(output)
        if actions.shape != (ACTION_HORIZON, 7):
            raise ValueError(f"Fast-WAM returned invalid action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Fast-WAM returned non-finite actions")

        timings = {
            "prompt_seconds": round(prompt_seconds, 4),
            "image_encode_seconds": round(image_seconds, 4),
            "action_inference_seconds": round(inference_seconds, 4),
            "total_seconds": round(time.perf_counter() - request_started, 4),
            "prompt_cache_hit": prompt_cache_hit,
            "cuda_peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
        }
        artifact = self._write_artifact(
            request_id=request_id,
            prompt=prompt,
            state=state,
            combined=combined,
            normalized=output.numpy(),
            actions=actions,
            timings=timings,
        )
        current_latents = None
        self.torch.cuda.empty_cache()
        return {
            "schema_version": 1,
            "request_id": request_id,
            "actions": actions.tolist(),
            "timing": timings,
            "artifact": artifact,
        }

    def _write_artifact(
        self,
        *,
        request_id: int,
        prompt: str,
        state: np.ndarray,
        combined: np.ndarray,
        normalized: np.ndarray,
        actions: np.ndarray,
        timings: dict[str, Any],
    ) -> str | None:
        if self.artifact_dir is None:
            return None
        stem = f"request-{request_id:05d}"
        image_path = self.artifact_dir / f"{stem}-input.png"
        result_path = self.artifact_dir / f"{stem}.json"
        Image.fromarray(combined).save(image_path)
        record = {
            "schema_version": 1,
            "request_id": request_id,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_image": str(image_path),
            "input_image_sha256": hashlib.sha256(combined.tobytes()).hexdigest(),
            "state": state.tolist(),
            "normalized_actions": np.asarray(normalized).tolist(),
            "actions": actions.tolist(),
            "timing": timings,
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(result_path)
        return str(result_path)
