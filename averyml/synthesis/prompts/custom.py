"""Custom JSONL-based prompt source."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from averyml.synthesis.prompts.base import PromptSource
from averyml.utils.io import read_jsonl
from averyml.utils.registry import prompt_source_registry

logger = logging.getLogger(__name__)


@prompt_source_registry.register("custom")
class CustomPromptSource(PromptSource):
    """Loads prompts from a local JSONL file.

    Each line should have at minimum: {"prompt_id": "...", "prompt_text": "..."}
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self, max_prompts: int | None = None) -> list[dict[str, Any]]:
        items = read_jsonl(self.file_path)
        if max_prompts is not None:
            items = items[:max_prompts]
        logger.info(f"Loaded {len(items)} prompts from {self.file_path}")
        return items

    def format_for_model(self, prompt: dict[str, Any], tokenizer: Any) -> str:
        content = prompt["prompt_text"]
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
