"""Configuration for the evaluation pipeline (Step 3 of SSD)."""

from __future__ import annotations

from pydantic import Field

from averyml.config.base import BaseConfig
from averyml.config.synthesis import DecodingConfig


class EvaluationConfig(BaseConfig):
    """Configuration for evaluating a fine-tuned model on code benchmarks.

    Decodes with T_eval and rho_eval, evaluates on LiveCodeBench v5/v6.
    """

    model_id: str = Field(description="Fine-tuned model path or HF model ID")
    benchmark: str = Field(
        default="livecodebench_v6",
        description="Benchmark: 'livecodebench_v5' or 'livecodebench_v6'",
    )
    decoding: DecodingConfig = Field(default_factory=DecodingConfig, description="T_eval and rho_eval")
    max_tokens: int = Field(default=32768, ge=1, description="Maximum generation length")
    n_repeat: int = Field(default=20, ge=1, description="Samples per problem for pass@k")
    k_values: list[int] = Field(default=[1, 5, 10], description="k values for pass@k metrics")
    tensor_parallel_size: int = Field(default=1, ge=1, description="GPUs for vLLM tensor parallelism")
    output_path: str = Field(default="./results", description="Results output directory")
    seeds: list[int] = Field(default=[0, 1234, 1234, 1234], description="Random seeds per repeat")
    timeout_per_test: float = Field(default=6.0, gt=0, description="Timeout per test case in seconds")
    max_workers: int = Field(default=32, ge=1, description="Max parallel evaluation workers")
