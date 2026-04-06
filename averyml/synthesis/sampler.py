"""Sampler orchestrator: Step 1 of the SSD pipeline.

Samples solutions from a frozen base model with T_train and rho_train,
applies minimal filtering, and writes the output dataset.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tqdm import tqdm

from averyml.config.synthesis import SynthesisConfig
from averyml.synthesis.dataset_writer import DatasetWriter
from averyml.synthesis.filters import apply_minimal_filters
from averyml.utils.registry import prompt_source_registry, synthesis_backend_registry

logger = logging.getLogger(__name__)


class Sampler:
    """Orchestrates data synthesis (Step 1 of SSD).

    Pipeline:
    1. Load prompts from the configured source
    2. Load the base model via the configured backend
    3. For each prompt, generate N samples with T_train and rho_train
    4. Apply minimal syntactic filtering (no correctness checks)
    5. Write output dataset
    """

    def __init__(self, config: SynthesisConfig):
        self.config = config

    def run(self) -> Path:
        """Execute the full synthesis pipeline. Returns path to output dataset."""
        prompt_source = self._build_prompt_source()
        backend = self._build_backend()

        # Load prompts
        prompts = prompt_source.load(self.config.max_prompts)
        logger.info(f"Loaded {len(prompts)} prompts from '{self.config.prompt_source}'")

        # Load model
        logger.info(f"Loading model: {self.config.model_id} (backend={self.config.backend})")
        backend.load_model(self.config.model_id)
        logger.info("Model loaded")

        # Generate samples
        all_samples = []
        total_rounds = self.config.n_samples
        start_time = time.time()

        for sample_idx in tqdm(range(total_rounds), desc="Sampling rounds", disable=total_rounds <= 1):
            seed = self.config.seed + sample_idx
            logger.info(f"Generating sample {sample_idx + 1}/{total_rounds} (seed={seed})...")

            # Format prompts for the model
            formatted = [prompt_source.format_for_model(p, backend.tokenizer) for p in prompts]

            # Generate
            round_start = time.time()
            responses = backend.generate(
                formatted,
                self.config.decoding,
                self.config.max_tokens,
                seed,
            )
            round_elapsed = time.time() - round_start
            logger.info(
                f"Round {sample_idx + 1} complete: {len(responses)} responses "
                f"in {round_elapsed:.1f}s ({len(responses) / max(round_elapsed, 0.1):.1f} prompts/s)"
            )

            for prompt, response in zip(prompts, responses):
                all_samples.append({
                    "prompt_id": prompt["prompt_id"],
                    "prompt_text": prompt["prompt_text"],
                    "response": response,
                    "sample_idx": sample_idx,
                    "decoding_config": self.config.decoding.model_dump(),
                })

        total_elapsed = time.time() - start_time
        logger.info(f"Generated {len(all_samples)} total samples in {total_elapsed:.1f}s")

        # Apply minimal filtering
        filtered = apply_minimal_filters(all_samples)
        removed = len(all_samples) - len(filtered)
        pct = (removed / len(all_samples) * 100) if all_samples else 0
        logger.info(f"After filtering: {len(filtered)} samples (removed {removed}, {pct:.1f}%)")
        if pct > 20:
            logger.warning(
                f"High filter rate ({pct:.1f}%). This may indicate a problem with generation "
                f"(e.g., wrong prompt format, model producing empty outputs). "
                f"SSD typically filters <5% of samples."
            )

        # Write output
        output_path = Path(self.config.output_path)
        result_path = DatasetWriter.write(filtered, output_path, self.config.output_format)

        backend.cleanup()
        logger.info(f"Synthesis complete: {result_path}")
        return result_path

    def _build_prompt_source(self):
        cls = prompt_source_registry.get(self.config.prompt_source)
        return cls(self.config.prompt_dataset)

    def _build_backend(self):
        cls = synthesis_backend_registry.get(self.config.backend)
        if self.config.backend == "vllm":
            return cls(tensor_parallel_size=self.config.tensor_parallel_size)
        return cls()
