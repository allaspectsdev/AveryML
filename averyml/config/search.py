"""Configuration for hyperparameter grid search over temperature space."""

from __future__ import annotations

from pydantic import Field

from averyml.config.base import BaseConfig
from averyml.config.synthesis import DecodingConfig


class SearchConfig(BaseConfig):
    """Configuration for grid search over (T_train, T_eval) and truncation configs.

    The paper shows that performance is well-governed by T_eff = T_train * T_eval,
    with a quadratic peak near T_eff ~ 1.2 (R^2=0.75). Truncation adds a second
    improvement channel on top of temperature composition.
    """

    base_model_id: str = Field(description="Base model to use for all grid cells")
    prompt_source: str = Field(default="rstarcoder", description="Prompt source for synthesis")
    prompt_dataset: str = Field(default="", description="HF dataset or local path for prompts")

    # Grid axes
    t_train_values: list[float] = Field(
        default=[0.5, 0.7, 1.0, 1.5, 2.0],
        description="Training temperatures to sweep",
    )
    t_eval_values: list[float] = Field(
        default=[0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5],
        description="Evaluation temperatures to sweep",
    )
    truncation_configs: list[DecodingConfig] | None = Field(
        default=None,
        description="Truncation configs to sweep (None = use default per model)",
    )

    # Per-cell pipeline defaults
    n_samples: int = Field(default=1, ge=1, description="Samples per prompt during synthesis")
    train_iterations: int = Field(default=2500, ge=1, description="Training iterations per cell")
    warmup_iterations: int = Field(default=250, ge=0, description="Warmup iterations per cell")
    n_repeat: int = Field(default=20, ge=1, description="Evaluation repeats per cell")
    benchmark: str = Field(default="livecodebench_v6", description="Benchmark for evaluation")

    # Resource management
    max_parallel_jobs: int = Field(default=1, ge=1, description="Max parallel grid cells")
    output_path: str = Field(default="./search_results", description="Search results directory")
