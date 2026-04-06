"""MBPP (Mostly Basic Programming Problems) benchmark implementation.

974 crowd-sourced Python programming problems from Google Research.
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

MBPP_PROMPT = """Write a Python function to solve the following problem. Return ONLY the function, no explanation.

{description}

Your function should pass these test cases:
{test_list}

```python
"""


@benchmark_registry.register("mbpp")
class MBPPBenchmark(Benchmark):
    """Google MBPP benchmark: 974 basic Python programming problems."""

    def __init__(self, max_workers: int = 32, timeout_per_test: float = 10.0, split: str = "test"):
        self.max_workers = max_workers
        self.timeout = timeout_per_test
        self.split = split

    def load_problems(self) -> list[dict]:
        from datasets import load_dataset

        logger.info(f"Loading MBPP (split={self.split})...")
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=self.split)
        problems = []
        for row in ds:
            problems.append({
                "task_id": str(row["task_id"]),
                "prompt": row["prompt"],
                "code": row["code"],
                "test_list": row["test_list"],
                "difficulty": "easy",  # MBPP problems are generally easier
            })
        logger.info(f"Loaded {len(problems)} MBPP problems")
        return problems

    def format_prompt(self, problem: dict, tokenizer: Any = None) -> str:
        test_str = "\n".join(problem["test_list"][:3])  # show first 3 test cases
        content = MBPP_PROMPT.format(description=problem["prompt"], test_list=test_str)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": content}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return content

    def evaluate_solution(self, problem: dict, code: str, timeout: float = 10.0) -> dict:
        import multiprocessing
        import re

        result_entry = {
            "task_id": problem["task_id"],
            "difficulty": problem.get("difficulty", "easy"),
            "correctness": None,
            "reason": None,
            "test_results": [],
            "num_tests_passed": 0,
            "num_tests_failed": 0,
        }

        # Extract code
        code_match = re.findall(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        completion = code_match[-1] if code_match else code

        # Build test program
        test_code = completion + "\n" + "\n".join(problem["test_list"])

        manager = multiprocessing.Manager()
        result = manager.list()

        def run_test(program, result_list):
            reliability_guard()
            try:
                exec(program, {})  # noqa: S102
                result_list.append(True)
            except Exception:
                result_list.append(False)

        p = multiprocessing.Process(target=run_test, args=(test_code, result))
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
