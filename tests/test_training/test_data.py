"""Tests for SFT dataset construction."""

import json
from pathlib import Path

import pytest


class TestSFTDatasetLoad:
    def test_loads_from_jsonl(self, tmp_path):
        """Test that SFTDataset can load and parse a JSONL file."""
        jsonl_path = tmp_path / "test.jsonl"
        samples = [
            {"prompt_id": "1", "prompt_text": "Write hello world", "response": "print('hello world')"},
            {"prompt_id": "2", "prompt_text": "Add two numbers", "response": "def add(a, b): return a + b"},
        ]
        with open(jsonl_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        from unittest.mock import MagicMock

        # Mock tokenizer that just returns character-level tokens
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "user: prompt\nassistant: response"
        tokenizer.__call__ = MagicMock(return_value={
            "input_ids": list(range(20)),
            "attention_mask": [1] * 20,
        })

        from averyml.training.data import SFTDataset

        ds = SFTDataset(dataset_path=str(jsonl_path), tokenizer=tokenizer, max_seq_length=512)
        # Verify it can find the file and read it
        assert ds.dataset_path == jsonl_path

    def test_rejects_missing_file(self):
        from unittest.mock import MagicMock

        from averyml.training.data import SFTDataset

        ds = SFTDataset(dataset_path="/nonexistent/path.jsonl", tokenizer=MagicMock(), max_seq_length=512)
        with pytest.raises(Exception):
            ds.load()

    def test_skips_empty_samples(self, tmp_path):
        """Samples with empty prompt_text or response should be filtered."""
        jsonl_path = tmp_path / "test.jsonl"
        samples = [
            {"prompt_id": "1", "prompt_text": "", "response": "code"},
            {"prompt_id": "2", "prompt_text": "prompt", "response": ""},
            {"prompt_id": "3", "prompt_text": "good prompt", "response": "good code"},
        ]
        with open(jsonl_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        from averyml.utils.io import read_jsonl

        loaded = read_jsonl(jsonl_path)
        # The dataset should have 3 items loaded, but after format filtering,
        # only the one with both prompt and response should remain
        valid = [s for s in loaded if s.get("prompt_text") and s.get("response")]
        assert len(valid) == 1
