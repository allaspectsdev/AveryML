"""Minimal filtering for synthesized data.

IMPORTANT: SSD deliberately uses NO correctness filtering. The only filters
applied are syntactic: removing empty responses and single-line stubs.
This is a core insight of the paper -- even training on gibberish outputs
improves the model through distribution reshaping.
"""

from __future__ import annotations

import re
from typing import Any


def extract_code_block(text: str) -> str:
    """Extract the last code block from markdown-formatted text."""
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return text


def is_empty_response(text: str) -> bool:
    """Returns True if response is empty or whitespace-only."""
    return not text or not text.strip()


def is_single_line_stub(text: str) -> bool:
    """Returns True if extracted code is a single-line stub like 'pass' or '...'."""
    code = extract_code_block(text)
    lines = [line for line in code.strip().splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return True
    if len(lines) == 1:
        stub_patterns = {"pass", "...", "return", "return None", "return 0", "return []", "return ''", 'return ""'}
        return lines[0].strip() in stub_patterns
    return False


def apply_minimal_filters(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply ONLY the minimal syntactic filters described in the paper.

    Removes:
    - Empty responses
    - Single-line stub responses

    Does NOT filter by:
    - Correctness (no execution, no test cases)
    - Code quality
    - Language detection
    """
    filtered = []
    removed_empty = 0
    removed_stub = 0

    for sample in samples:
        response = sample.get("response", "")
        if is_empty_response(response):
            removed_empty += 1
            continue
        if is_single_line_stub(response):
            removed_stub += 1
            continue
        filtered.append(sample)

    total_removed = removed_empty + removed_stub
    if total_removed > 0:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Filtered {total_removed}/{len(samples)} samples "
            f"(empty={removed_empty}, stub={removed_stub})"
        )

    return filtered
