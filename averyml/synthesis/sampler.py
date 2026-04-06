"""Sampler orchestrator: Step 1 of the SSD pipeline.

Samples solutions from a frozen base model with T_train and rho_train,
applies minimal filtering, and writes the output dataset.
Supports checkpointing for crash recovery.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from averyml.config.synthesis import SynthesisConfig
from averyml.synthesis.dataset_writer import DatasetWriter
from averyml.synthesis.filters import apply_minimal_filters
from averyml.utils.registry import prompt_source_registry, synthesis_backend_registry

logger = logging.getLogger(__name__)


def compute_cache_key(config: SynthesisConfig) -> str:
    """Compute a content-addressed cache key for synthesis outputs."""
    key_data = {
        "model_id": config.model_id,
        "prompt_source": config.prompt_source,
        "prompt_dataset": config.prompt_dataset,
        "max_prompts": config.max_prompts,
        "n_samples": config.n_samples,
        "temperature": config.decoding.temperature,
        "top_k": config.decoding.top_k,
        "top_p": config.decoding.top_p,
        "min_p": config.decoding.min_p,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
    }
    raw = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Sampler:
    """Orchestrates data synthesis (Step 1 of SSD).

    Pipeline:
    1. Load prompts from the configured source
    2. Load the base model via the configured backend
    3. For each prompt, generate N samples with T_train and rho_train
    4. Apply minimal syntactic filtering (no correctness checks)
    5. Write output dataset

    Supports checkpointing: if checkpoint_every > 0, saves partial results
    to a checkpoint file. On resume, skips already-completed prompts.
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

        # Check for existing checkpoint
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / "_checkpoint.jsonl"
        existing_samples = self._load_checkpoint(checkpoint_path)
        completed_ids = {(s["prompt_id"], s["sample_idx"]) for s in existing_samples}

        if existing_samples:
            logger.info(f"Resuming from checkpoint: {len(existing_samples)} samples already completed")

        # Generate samples
        all_samples = list(existing_samples)
        total_rounds = self.config.n_samples
        start_time = time.time()
        checkpoint_counter = 0

        for sample_idx in tqdm(range(total_rounds), desc="Sampling rounds", disable=total_rounds <= 1):
            # Figure out which prompts still need generation for this round
            remaining_prompts = []
            remaining_indices = []
            for i, p in enumerate(prompts):
                if (p["prompt_id"], sample_idx) not in completed_ids:
                    remaining_prompts.append(p)
                    remaining_indices.append(i)

            if not remaining_prompts:
                logger.info(f"Round {sample_idx + 1}/{total_rounds}: all prompts already completed")
                continue

            seed = self.config.seed + sample_idx
            logger.info(
                f"Generating round {sample_idx + 1}/{total_rounds}: "
                f"{len(remaining_prompts)} prompts (seed={seed})..."
            )

            # Format and generate
            formatted = [prompt_source.format_for_model(p, backend.tokenizer) for p in remaining_prompts]

            round_start = time.time()
            responses = backend.generate(formatted, self.config.decoding, self.config.max_tokens, seed)
            round_elapsed = time.time() - round_start

            logger.info(
                f"Round {sample_idx + 1} complete: {len(responses)} responses "
                f"in {round_elapsed:.1f}s ({len(responses) / max(round_elapsed, 0.1):.1f} prompts/s)"
            )

            # Collect results
            new_samples = []
            for prompt, response in zip(remaining_prompts, responses):
                sample = {
                    "prompt_id": prompt["prompt_id"],
                    "prompt_text": prompt["prompt_text"],
                    "response": response,
                    "sample_idx": sample_idx,
                    "decoding_config": self.config.decoding.model_dump(),
                }
                new_samples.append(sample)
                all_samples.append(sample)

            # Checkpoint if configured
            if self.config.checkpoint_every > 0:
                checkpoint_counter += len(new_samples)
                if checkpoint_counter >= self.config.checkpoint_every:
                    self._save_checkpoint(all_samples, checkpoint_path)
                    checkpoint_counter = 0

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

        # Write final output
        result_path = DatasetWriter.write(filtered, output_path, self.config.output_format)

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint file removed (synthesis complete)")

        backend.cleanup()
        logger.info(f"Synthesis complete: {result_path}")
        return result_path

    def _load_checkpoint(self, path: Path) -> list[dict]:
        """Load existing checkpoint samples."""
        if not path.exists():
            return []
        try:
            from averyml.utils.io import read_jsonl
            return read_jsonl(path)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            return []

    def _save_checkpoint(self, samples: list[dict], path: Path) -> None:
        """Save current samples to checkpoint file."""
        from averyml.utils.io import write_jsonl
        write_jsonl(samples, path)
        logger.info(f"Checkpoint saved: {len(samples)} samples to {path}")

    def _build_prompt_source(self):
        cls = prompt_source_registry.get(self.config.prompt_source)
        return cls(self.config.prompt_dataset)

    def _build_backend(self):
        cls = synthesis_backend_registry.get(self.config.backend)
        if self.config.backend == "vllm":
            return cls(tensor_parallel_size=self.config.tensor_parallel_size)
        return cls()
