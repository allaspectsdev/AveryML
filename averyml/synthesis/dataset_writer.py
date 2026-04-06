"""Write synthesis output to various formats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from averyml.utils.io import write_jsonl

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Writes synthesis output to JSONL or HuggingFace Dataset format."""

    @staticmethod
    def write(samples: list[dict[str, Any]], output_path: Path, fmt: str = "jsonl") -> Path:
        """Write samples to the specified format. Returns the output path."""
        if fmt == "jsonl":
            return DatasetWriter.write_jsonl(samples, output_path)
        elif fmt == "hf_dataset":
            return DatasetWriter.write_hf_dataset(samples, output_path)
        else:
            raise ValueError(f"Unknown output format: {fmt}. Use 'jsonl' or 'hf_dataset'.")

    @staticmethod
    def write_jsonl(samples: list[dict[str, Any]], output_path: Path) -> Path:
        """Write samples as JSONL."""
        file_path = output_path / "synthesis.jsonl"
        write_jsonl(samples, file_path)
        logger.info(f"Wrote {len(samples)} samples to {file_path}")
        return file_path

    @staticmethod
    def write_hf_dataset(samples: list[dict[str, Any]], output_path: Path) -> Path:
        """Write samples as a HuggingFace Dataset."""
        from datasets import Dataset

        ds = Dataset.from_list(samples)
        ds.save_to_disk(str(output_path / "synthesis_dataset"))
        logger.info(f"Wrote {len(samples)} samples as HF dataset to {output_path / 'synthesis_dataset'}")
        return output_path / "synthesis_dataset"
