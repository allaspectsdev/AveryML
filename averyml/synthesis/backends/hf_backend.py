"""HuggingFace Transformers synthesis backend."""

from __future__ import annotations

from typing import Any

import torch

from averyml.config.synthesis import DecodingConfig
from averyml.synthesis.backends.base import SynthesisBackend
from averyml.utils.registry import synthesis_backend_registry


@synthesis_backend_registry.register("hf")
class HFSynthesisBackend(SynthesisBackend):
    """Uses HuggingFace Transformers for synthesis (slower but more accessible)."""

    def __init__(self, device: str = "auto"):
        self._device = device
        self._model = None
        self._tokenizer = None

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
            **kwargs,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        prompts: list[str],
        decoding: DecodingConfig,
        max_tokens: int,
        seed: int,
    ) -> list[str]:
        assert self._model is not None and self._tokenizer is not None, "Call load_model() first"

        torch.manual_seed(seed)
        results = []

        # Process one at a time to avoid OOM on large models
        for prompt in prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True).to(self._model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=decoding.temperature,
                    top_k=decoding.top_k if decoding.top_k > 0 else None,
                    top_p=decoding.top_p,
                    do_sample=True,
                )

            generated = output_ids[0][input_len:]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text)

        return results

    @property
    def tokenizer(self):
        assert self._tokenizer is not None, "Call load_model() first"
        return self._tokenizer

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
