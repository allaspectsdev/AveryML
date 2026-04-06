"""Configuration for the data synthesis pipeline (Step 1 of SSD)."""

from __future__ import annotations

from pydantic import Field

from averyml.config.base import BaseConfig


class DecodingConfig(BaseConfig):
    """Decoding parameters shared between synthesis (T_train) and evaluation (T_eval).

    Controls temperature and truncation (top-k, top-p) for sampling.
    """

    temperature: float = Field(default=0.6, ge=0.0, description="Sampling temperature")
    top_k: int = Field(default=20, ge=-1, description="Top-k truncation (-1 for disabled)")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p (nucleus) truncation")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Min-p truncation")


class SynthesisConfig(BaseConfig):
    """Configuration for data synthesis: sampling solutions from a frozen base model.

    The paper samples N=1 solution per prompt with T_train and rho_train,
    then uses those raw, unverified outputs as SFT training data.
    """

    model_id: str = Field(description="HuggingFace model ID for the frozen base model")
    prompt_source: str = Field(default="rstarcoder", description="Prompt source: 'rstarcoder' or 'custom'")
    prompt_dataset: str = Field(
        default="",
        description="HF dataset name or local path for prompts",
    )
    max_prompts: int | None = Field(default=None, description="Limit number of prompts (None = use all)")
    n_samples: int = Field(default=1, ge=1, description="Samples per prompt (paper: N=1 suffices)")
    decoding: DecodingConfig = Field(default_factory=DecodingConfig, description="T_train and rho_train")
    backend: str = Field(default="vllm", description="Inference backend: 'vllm' or 'hf'")
    max_tokens: int = Field(default=32768, ge=1, description="Maximum generation length")
    tensor_parallel_size: int = Field(default=1, ge=1, description="GPUs for vLLM tensor parallelism")
    output_path: str = Field(default="./data/synthesis", description="Output directory")
    output_format: str = Field(default="jsonl", description="Output format: 'jsonl' or 'hf_dataset'")
    checkpoint_every: int = Field(
        default=0, ge=0,
        description="Save partial results every N prompts (0 to disable). Enables resume on crash.",
    )
    seed: int = Field(default=42, description="Random seed")
