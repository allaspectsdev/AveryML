"""HumanEval benchmark implementation.

164 hand-written Python programming problems from OpenAI.
Each has a function signature, docstring, and test cases.
"""

from __future__ import annotations

import logging
from typing import Any

from averyml.evaluation.benchmarks.base import Benchmark
from averyml.evaluation.benchmarks.livecodebench_utils import (
    compute_metrics_from_results,
    reliability_guard,
)
from averyml.utils.registry import benchmark_registry

logger = logging.getLogger(__name__)

HUMANEVAL_PROMPT = """Complete the following Python function. Return ONLY the function implementation, no explanation.

```python
{prompt}
```"""


@benchmark_registry.register("humaneval")
class HumanEvalBenchmark(Benchmark):
    """OpenAI HumanEval benchmark: 164 Python function completion problems."""

    def __init__(self, max_workers: int = 32, timeout_per_test: float = 10.0):
        self.max_workers = max_workers
        self.timeout = timeout_per_test

    def load_problems(self) -> list[dict]:
        from datasets import load_dataset

        logger.info("Loading HumanEval from openai/openai_humaneval...")
        ds = load_dataset("openai_humaneval", split="test")
        problems = []
        for row in ds:
            problems.append({
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "difficulty": "medium",  # HumanEval doesn't have difficulty labels
            })
        logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems

    def format_prompt(self, problem: dict, tokenizer: Any = None) -> str:
        content = HUMANEVAL_PROMPT.format(prompt=problem["prompt"])
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": content}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return content

    def evaluate_solution(self, problem: dict, code: str, timeout: float = 10.0) -> dict:
        import multiprocessing
        import re

        result_entry = {
            "task_id": problem["task_id"],
            "difficulty": problem.get("difficulty", "medium"),
            "correctness": None,
            "reason": None,
            "test_results": [],
            "num_tests_passed": 0,
            "num_tests_failed": 0,
        }

        # Extract code from markdown if present
        code_match = re.findall(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        completion = code_match[-1] if code_match else code

        # Build full program: prompt + completion + tests
        full_program = problem["prompt"] + completion + "\n" + problem["test"] + f"\ncheck({problem['entry_point']})\n"

        # Run in isolated process
        manager = multiprocessing.Manager()
        result = manager.list()

        def run_test(program, result_list):
            reliability_guard()
            try:
                exec(program, {})  # noqa: S102
                result_list.append(True)
            except Exception as e:
                result_list.append(False)

        p = multiprocessing.Process(target=run_test, args=(full_program, result))
        p.start()
        p.join(timeout=timeout or self.timeout)
        if p.is_alive():
            p.kill()
            result.append(False)

        passed = bool(result) and result[0] is True
        result_entry["correctness"] = passed
        result_entry["test_results"] = [1 if passed else 0]
        result_entry["num_tests_passed"] = 1 if passed else 0
        result_entry["num_tests_failed"] = 0 if passed else 1
        result_entry["reason"] = "" if passed else "Tests failed"

        return result_entry

    def compute_metrics(self, results: dict[str, list[list[int]]], k_values: list[int]) -> dict[str, Any]:
        return compute_metrics_from_results(results, k_list=k_values)
