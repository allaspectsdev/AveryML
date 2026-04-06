"""Top-level experiment configuration that wires the full SSD pipeline."""

from __future__ import annotations

from pydantic import Field

from averyml.config.base import BaseConfig
from averyml.config.evaluation import EvaluationConfig
from averyml.config.search import SearchConfig
from averyml.config.synthesis import SynthesisConfig
from averyml.config.training import TrainingConfig


class ExperimentConfig(BaseConfig):
    """Top-level config that wires synthesis -> training -> evaluation.

    Used by the `averyml run-pipeline` command to execute the full SSD pipeline.
    """

    name: str = Field(description="Experiment name for tracking and output paths")
    synthesis: SynthesisConfig = Field(description="Data synthesis configuration")
    training: TrainingConfig = Field(description="Fine-tuning configuration")
    evaluation: EvaluationConfig = Field(description="Evaluation configuration")
    search: SearchConfig | None = Field(default=None, description="Optional grid search configuration")
