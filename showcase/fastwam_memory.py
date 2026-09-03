"""Memory-bounded Fast-WAM loading for 24 GB local GPUs.

The pinned publisher deploy wrapper always moves UMT5-XXL to CUDA alongside
the roughly 6B-parameter Fast-WAM checkpoint. That combination consumes the
entire 24 GB card before SAPIEN or inference can allocate working memory.
Keep the frozen text encoder on CPU and cache only its small prompt embedding
on CUDA, matching the offload boundary already exposed by Flex-pi.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Sequence
from typing import Any


def prompt_cache_key(prompt: str | Sequence[str]) -> str | tuple[str, ...]:
    """Return a stable key for a single prompt or prompt batch."""
    return prompt if isinstance(prompt, str) else tuple(prompt)


def load_fastwam_with_cpu_text_encoder(
    get_model: Callable[[dict[str, Any]], Any], policy_args: dict[str, Any]
) -> Any:
    """Load the publisher policy while keeping its frozen UMT5 encoder on CPU."""
    import torch
    from fastwam import runtime
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.helpers import loader

    original_factory = runtime.create_fastwam
    original_to = FastWAM.to
    original_encode_prompt = FastWAM.encode_prompt

    def offloaded_to(self, *args, **kwargs):
        if not getattr(self, "_epl_offload_text_encoder", False):
            return original_to(self, *args, **kwargs)

        # nn.Module.to() recursively moves every registered child. Temporarily
        # detach the frozen CPU encoder so only policy/vision components move.
        text_encoder = self._modules.pop("text_encoder", None)
        try:
            torch.nn.Module.to(self, *args, **kwargs)
        finally:
            if text_encoder is not None:
                self._modules["text_encoder"] = text_encoder
        return self

    @torch.no_grad()
    def offloaded_encode_prompt(self, prompt):
        if not getattr(self, "_epl_offload_text_encoder", False):
            return original_encode_prompt(self, prompt)
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Fast-WAM CPU text encoder/tokenizer is unavailable")

        key = prompt_cache_key(prompt)
        cached = getattr(self, "_epl_prompt_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]

        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to("cpu")
        mask = mask.to("cpu", dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        for index, length in enumerate(sequence_lengths):
            prompt_emb[index, length:] = 0
        mask = torch.ones_like(mask)
        context = prompt_emb.to(device=self.device, dtype=self.torch_dtype)
        context_mask = mask.to(device=self.device, dtype=torch.bool)
        self._epl_prompt_cache = (key, context, context_mask)
        return context, context_mask

    def offloaded_factory(*args, **kwargs):
        signature = inspect.signature(original_factory)
        bound = signature.bind_partial(*args, **kwargs)
        requested = bool(bound.arguments.get("load_text_encoder", True))
        if not requested:
            return original_factory(*args, **kwargs)

        bound.arguments["load_text_encoder"] = False
        model = original_factory(*bound.args, **bound.kwargs)

        model_id = str(bound.arguments.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
        tokenizer_model_id = str(
            bound.arguments.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
        )
        tokenizer_max_len = int(bound.arguments.get("tokenizer_max_len", 512))
        redirect_common_files = bool(bound.arguments.get("redirect_common_files", True))
        model_dtype = bound.arguments.get("model_dtype", torch.bfloat16)

        _, text_config, _, tokenizer_config = loader._resolve_configs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            redirect_common_files=redirect_common_files,
        )
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        text_encoder = loader._load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=model_dtype,
            device="cpu",
        )
        tokenizer = loader.HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=tokenizer_max_len,
            clean="whitespace",
        )
        model.text_encoder = text_encoder.eval()
        model.tokenizer = tokenizer
        model._epl_offload_text_encoder = True
        model._epl_prompt_cache = None
        if hasattr(model, "model_paths"):
            model.model_paths["text_encoder"] = str(text_config.path)
            model.model_paths["tokenizer"] = str(tokenizer_config.path)
        return model

    runtime.create_fastwam = offloaded_factory
    FastWAM.to = offloaded_to
    FastWAM.encode_prompt = offloaded_encode_prompt
    try:
        policy = get_model(policy_args)
        # The publisher wrapper has completed its final `.to(cuda).eval()`.
        # Keep offload behavior on this instance after restoring globals.
        policy.model.to = types.MethodType(offloaded_to, policy.model)
        policy.model.encode_prompt = types.MethodType(
            offloaded_encode_prompt, policy.model
        )
        return policy
    finally:
        runtime.create_fastwam = original_factory
        FastWAM.to = original_to
        FastWAM.encode_prompt = original_encode_prompt
