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

EXPECTED_FIELDS = {"prompt_text", "response"}


class SFTDataset:
    """Prepares synthesis output for supervised fine-tuning.

    Tokenizes prompt+completion pairs, masking prompt tokens in labels
    so cross-entropy loss only applies to the completion.
    """

    def __init__(self, dataset_path: str, tokenizer: Any, max_seq_length: int = 65536, packing: bool = False):
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.packing = packing

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

        logger.info(f"Loaded {len(raw_samples)} raw samples from {self.dataset_path}")

        if not raw_samples:
            raise ValueError(f"Dataset is empty: {self.dataset_path}")

        # Validate fields
        first_keys = set(raw_samples[0].keys())
        missing = EXPECTED_FIELDS - first_keys
        if missing:
            raise ValueError(
                f"Dataset is missing required fields: {missing}. "
                f"Found fields: {sorted(first_keys)}. "
                f"Expected at least: {sorted(EXPECTED_FIELDS)}. "
                f"Make sure the synthesis output matches the training data format."
            )

        # Validate tokenizer has chat template
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError(
                f"Tokenizer for {getattr(self.tokenizer, 'name_or_path', '?')} "
                f"does not support apply_chat_template. Use a chat/instruct model."
            )

        # Format as chat conversations
        formatted = []
        skipped = 0
        for sample in raw_samples:
            prompt_text = sample.get("prompt_text", "")
            response = sample.get("response", "")
            if not prompt_text or not response:
                skipped += 1
                continue
            formatted.append({
                "messages": [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": response},
                ]
            })

        if skipped > 0:
            pct = skipped / len(raw_samples) * 100
            logger.warning(
                f"Skipped {skipped}/{len(raw_samples)} samples ({pct:.1f}%) "
                f"with empty prompt_text or response"
            )

        if not formatted:
            raise ValueError(
                f"All {len(raw_samples)} samples were filtered out. "
                f"Check that the dataset has non-empty 'prompt_text' and 'response' fields."
            )

        logger.info(f"Formatted {len(formatted)} samples as chat conversations")

        # Validate first sample tokenizes correctly
        try:
            test_result = self._tokenize_and_mask(formatted[0])
            if len(test_result["input_ids"]) == 0:
                logger.warning("First sample tokenized to 0 tokens — check chat template compatibility")
        except Exception as e:
            raise ValueError(
                f"Failed to tokenize first sample: {e}. "
                f"This usually means the tokenizer's chat template is incompatible."
            ) from e

        ds = Dataset.from_list(formatted)
        ds = ds.map(
            self._tokenize_and_mask,
            remove_columns=["messages"],
            desc="Tokenizing",
        )
        ds = ds.filter(lambda x: len(x["input_ids"]) > 0, desc="Filtering empty")

        logger.info(f"Tokenized dataset: {len(ds)} samples")

        if self.packing:
            ds = self._pack_dataset(ds)
            logger.info(f"Packed dataset: {len(ds)} packed sequences")

        logger.info(f"Final dataset: {len(ds)} samples")
        return ds

    def _pack_dataset(self, ds):
        """Pack multiple short samples into max_seq_length sequences.

        Concatenates samples end-to-end with EOS boundaries. Produces
        ~3-5x fewer but longer sequences, eliminating padding waste.
        """
        from datasets import Dataset as HFDataset

        eos_id = self.tokenizer.eos_token_id or 0
        packed = []
        current_ids = []
        current_labels = []

        for sample in ds:
            sample_ids = sample["input_ids"]
            sample_labels = sample["labels"]

            # Would overflow — finalize current bin
            if current_ids and len(current_ids) + len(sample_ids) + 1 > self.max_seq_length:
                packed.append(self._finalize_packed(current_ids, current_labels))
                current_ids = []
                current_labels = []

            # Add separator between documents
            if current_ids:
                current_ids.append(eos_id)
                current_labels.append(-100)  # don't compute loss on separator

            current_ids.extend(sample_ids)
            current_labels.extend(sample_labels)

        # Finalize last bin
        if current_ids:
            packed.append(self._finalize_packed(current_ids, current_labels))

        unpacked_count = len(ds)
        pack_ratio = unpacked_count / max(len(packed), 1)
        logger.info(f"Packing: {unpacked_count} samples -> {len(packed)} sequences ({pack_ratio:.1f}x compression)")

        return HFDataset.from_list(packed)

    def _finalize_packed(self, ids: list[int], labels: list[int]) -> dict:
        """Pad a packed sequence to max_seq_length."""
        pad_id = self.tokenizer.pad_token_id or 0
        pad_len = self.max_seq_length - len(ids)
        return {
            "input_ids": ids + [pad_id] * pad_len,
            "attention_mask": [1] * len(ids) + [0] * pad_len,
            "labels": labels + [-100] * pad_len,
        }

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
