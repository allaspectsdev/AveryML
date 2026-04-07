"""SSD Trainer orchestrator: Step 2 of the SSD pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from averyml.config.training import TrainingConfig
from averyml.training.data import SFTDataset
from averyml.utils.registry import training_backend_registry

logger = logging.getLogger(__name__)

# Known thinking model patterns (these use 300 iterations, 50 warmup)
THINKING_MODEL_PATTERNS = ["thinking", "think", "reasoning"]


def is_thinking_model(model_id: str) -> bool:
    """Detect if a model is a thinking/reasoning variant."""
    lower = model_id.lower()
    return any(p in lower for p in THINKING_MODEL_PATTERNS)


class SSDTrainer:
    """Orchestrates fine-tuning (Step 2 of SSD).

    Pipeline:
    1. Load synthesis output as SFT dataset
    2. Tokenize with prompt masking (loss on completion only)
    3. Fine-tune with the configured backend (HF Trainer or torchtune)
    4. Save final checkpoint

    Auto-detects thinking models and warns if iterations look wrong.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def run(self) -> Path:
        """Execute the fine-tuning pipeline. Returns path to final checkpoint."""
        from transformers import AutoTokenizer

        # Validate config for common mistakes
        self._validate_config()

        logger.info(f"Preparing SFT dataset from {self.config.dataset_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = SFTDataset(
            dataset_path=self.config.dataset_path,
            tokenizer=tokenizer,
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
        ).load()

        logger.info(f"Dataset ready: {len(dataset)} samples")

        backend_cls = training_backend_registry.get(self.config.backend)
        backend = backend_cls()

        checkpoint_path = backend.train(self.config, dataset, tokenizer=tokenizer)
        logger.info(f"Training complete. Checkpoint: {checkpoint_path}")
        return checkpoint_path

    def _validate_config(self):
        """Check for common config mistakes and log warnings."""
        model_id = self.config.model_id
        iterations = self.config.num_train_iterations
        warmup = self.config.warmup_iterations

        if is_thinking_model(model_id):
            if iterations > 500:
                logger.warning(
                    f"{'='*60}\n"
                    f"THINKING MODEL DETECTED: {model_id}\n"
                    f"You have num_train_iterations={iterations}, but the paper uses 300\n"
                    f"for thinking models (Section 3.1). This will use ~8x more compute\n"
                    f"than needed. Consider setting num_train_iterations=300, warmup=50.\n"
                    f"{'='*60}"
                )
            if warmup > 100:
                logger.warning(
                    f"Thinking model with warmup_iterations={warmup}. "
                    f"Paper uses 50 for thinking models."
                )
        else:
            if iterations < 500:
                logger.warning(
                    f"Instruct model with only {iterations} iterations. "
                    f"Paper uses 2500 for instruct models. Results may be suboptimal."
                )

        if self.config.max_seq_length < 32768:
            logger.warning(
                f"max_seq_length={self.config.max_seq_length} is unusually low. "
                f"Paper uses 65536. Short sequences may hurt performance."
            )
