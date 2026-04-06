"""vLLM-based synthesis backend for high-throughput inference."""

from __future__ import annotations

from typing import Any

from averyml.config.synthesis import DecodingConfig
from averyml.synthesis.backends.base import SynthesisBackend
from averyml.utils.registry import synthesis_backend_registry


@synthesis_backend_registry.register("vllm")
class VLLMSynthesisBackend(SynthesisBackend):
    """Uses vLLM offline inference for high-throughput data synthesis."""

    def __init__(self, tensor_parallel_size: int = 1):
        self._tensor_parallel_size = tensor_parallel_size
        self._llm = None
        self._tokenizer = None

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=self._tensor_parallel_size,
            **kwargs,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate(
        self,
        prompts: list[str],
        decoding: DecodingConfig,
        max_tokens: int,
        seed: int,
    ) -> list[str]:
        from vllm import SamplingParams

        assert self._llm is not None, "Call load_model() first"

        stop_token_ids = []
        if self._tokenizer.eos_token_id is not None:
            stop_token_ids = [self._tokenizer.eos_token_id]

        params = SamplingParams(
            max_tokens=max_tokens,
            seed=seed,
            stop_token_ids=stop_token_ids,
            temperature=decoding.temperature,
            top_k=decoding.top_k,
            top_p=decoding.top_p,
            min_p=decoding.min_p,
        )

        outputs = self._llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Call load_model() before accessing tokenizer")
        return self._tokenizer

    def cleanup(self) -> None:
        self._llm = None
        self._tokenizer = None
