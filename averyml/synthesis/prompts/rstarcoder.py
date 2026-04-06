"""rSTARcoder prompt source for competitive programming problems."""

from __future__ import annotations

import logging
from typing import Any

from averyml.synthesis.prompts.base import PromptSource
from averyml.utils.registry import prompt_source_registry

logger = logging.getLogger(__name__)

CODING_PROMPT = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.
```python
  # YOUR CODE HERE
```"""


@prompt_source_registry.register("rstarcoder")
class RStarCoderPromptSource(PromptSource):
    """Loads competitive programming problems from rSTARcoder seed subset.

    The paper uses ~10K unique competitive programming problems,
    de-duplicated from the rSTARcoder dataset.
    """

    def __init__(self, dataset_name: str = ""):
        self.dataset_name = dataset_name

    def load(self, max_prompts: int | None = None) -> list[dict[str, Any]]:
        from datasets import load_dataset

        logger.info(f"Loading rSTARcoder prompts from: {self.dataset_name}")

        if self.dataset_name:
            logger.info(f"Loading dataset: {self.dataset_name}")
            try:
                ds = load_dataset(self.dataset_name, split="train")
            except Exception as e:
                raise ValueError(
                    f"Failed to load rSTARcoder dataset '{self.dataset_name}': {e}\n"
                    f"Check that the dataset ID is correct and accessible. "
                    f"You may need to accept the dataset's license on HuggingFace."
                ) from e
        else:
            logger.warning(
                "=" * 60 + "\n"
                "WARNING: No prompt_dataset specified for rSTARcoder source.\n"
                "Falling back to LiveCodeBench problems as prompts.\n"
                "This uses a DIFFERENT distribution than the paper.\n"
                "For faithful SSD reproduction, set prompt_dataset in your config.\n"
                + "=" * 60
            )
            ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)

        prompts = []
        for i, row in enumerate(ds):
            if max_prompts is not None and i >= max_prompts:
                break

            prompt_text = row.get("question_content") or row.get("prompt") or row.get("problem", "")
            prompt_id = row.get("question_id") or row.get("id") or str(i)

            prompts.append({
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "metadata": {k: v for k, v in row.items() if k not in ("question_content", "prompt", "problem")},
            })

        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts

    def format_for_model(self, prompt: dict[str, Any], tokenizer: Any) -> str:
        content = CODING_PROMPT.format(problem_description=prompt["prompt_text"])
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
