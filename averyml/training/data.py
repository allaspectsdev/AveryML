"""SFT dataset preparation for the training pipeline.

Loads synthesis output and prepares it for supervised fine-tuning.
Each sample is a (prompt, completion) pair. The loss is computed
only on the completion tokens (standard SFT).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from averyml.utils.io import read_jsonl

logger = logging.getLogger(__name__)


class SFTDataset:
    """Prepares synthesis output for supervised fine-tuning.

    Tokenizes prompt+completion pairs, masking prompt tokens in labels
    so cross-entropy loss only applies to the completion.
    """

    def __init__(self, dataset_path: str, tokenizer: Any, max_seq_length: int = 65536):
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def load(self):
        """Load and tokenize the dataset. Returns a HuggingFace Dataset."""
        from datasets import Dataset

        # Load raw samples
        if self.dataset_path.suffix == ".jsonl":
            raw_samples = read_jsonl(self.dataset_path)
        elif self.dataset_path.is_dir():
            ds = Dataset.load_from_disk(str(self.dataset_path))
            raw_samples = list(ds)
        else:
            raise ValueError(f"Unsupported dataset path: {self.dataset_path}")

        logger.info(f"Loaded {len(raw_samples)} samples from {self.dataset_path}")

        # Format as chat conversations
        formatted = []
        for sample in raw_samples:
            prompt_text = sample.get("prompt_text", "")
            response = sample.get("response", "")
            if not prompt_text or not response:
                continue
            formatted.append({
                "messages": [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": response},
                ]
            })

        logger.info(f"Formatted {len(formatted)} samples as chat conversations")

        ds = Dataset.from_list(formatted)
        ds = ds.map(
            self._tokenize_and_mask,
            remove_columns=["messages"],
            desc="Tokenizing",
        )
        ds = ds.filter(lambda x: len(x["input_ids"]) > 0, desc="Filtering empty")

        logger.info(f"Final dataset: {len(ds)} samples")
        return ds

    def _tokenize_and_mask(self, example: dict) -> dict:
        """Tokenize a chat conversation and mask prompt tokens in labels.

        The loss is computed only on assistant (completion) tokens.
        Prompt tokens get label=-100 (ignored by cross-entropy).
        """
        messages = example["messages"]

        # Full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_length, return_tensors=None
        )

        # Prompt-only (to find where completion starts)
        prompt_messages = [messages[0]]  # just the user message
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_seq_length, return_tensors=None)

        input_ids = full_ids["input_ids"]
        prompt_len = len(prompt_ids["input_ids"])

        # Labels: -100 for prompt tokens, actual ids for completion tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Ensure same length
        labels = labels[: len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": full_ids["attention_mask"],
            "labels": labels,
        }
