"""LiveCodeBench v5/v6 benchmark implementation."""

from __future__ import annotations

import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from averyml.evaluation.benchmarks.base import Benchmark
from averyml.evaluation.benchmarks.livecodebench_utils import (
    compute_metrics_from_results,
    has_code,
    lcb_run,
    map_to_example,
    post_process_code,
    translate_private_test_cases,
)
from averyml.utils.registry import benchmark_registry

logger = logging.getLogger(__name__)

# Prompt templates from the Apple reference
LCB_PROMPT_WITHOUT_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
  # YOUR CODE HERE
```"""

LCB_PROMPT_WITH_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
```python
{entry_point}
"""

# Date ranges for each LCB version
VERSION_DATE_RANGES: dict[str, list[str]] = {
    "livecodebench_v5": ["2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02"],
    "livecodebench_v6": ["2025-02", "2025-03", "2025-04", "2025-05"],
}


@benchmark_registry.register("livecodebench_v5")
@benchmark_registry.register("livecodebench_v6")
class LiveCodeBench(Benchmark):
    """LiveCodeBench benchmark supporting both v5 and v6."""

    def __init__(
        self,
        version: str = "livecodebench_v6",
        max_workers: int = 32,
        timeout_per_test: float = 6.0,
    ):
        self.version = version
        self.max_workers = max_workers
        self.timeout = timeout_per_test
        self.date_range = VERSION_DATE_RANGES.get(version, VERSION_DATE_RANGES["livecodebench_v6"])

    def load_problems(self) -> list[dict]:
        """Load LiveCodeBench problems from HuggingFace, filtered by version date range."""
        from datasets import concatenate_datasets, load_dataset

        logger.info("Loading LiveCodeBench problems from livecodebench/code_generation_lite...")
        lcb = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
        logger.info(f"Loaded {len(lcb)} total problems")

        def filter_by_date(example):
            return example["contest_date"][:7] in self.date_range

        ds = lcb.filter(filter_by_date)
        logger.info(f"{len(ds)} problems after date filter ({self.version})")

        # Sharded processing to avoid Arrow overflow
        processed_shards = []
        num_shards = 4
        cpu_count = os.cpu_count() or 1
        for i in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=i)
            shard = shard.map(
                lambda ex: {"private_test_cases": translate_private_test_cases(ex["private_test_cases"])},
                num_proc=cpu_count,
            )
            shard = shard.map(map_to_example, remove_columns=ds.column_names, load_from_cache_file=False)
            processed_shards.append(shard)

        ds = concatenate_datasets(processed_shards)
        return list(ds)

    def evaluate_solution(self, problem: dict, code: str, timeout: float | None = None) -> dict:
        """Evaluate a single code solution against the problem's test cases."""
        timeout = timeout or self.timeout
        result_entry = {
            "task_id": problem.get("task_id"),
            "difficulty": problem.get("difficulty"),
            "correctness": None,
            "reason": None,
            "num_tests_passed": 0,
            "num_tests_failed": 0,
            "test_results": [],
        }

        code_blocks = has_code(code) if isinstance(code, str) else code
        if not code_blocks:
            # Fallback: try the raw response as code (model may not use markdown fences)
            stripped = code.strip() if isinstance(code, str) else ""
            if stripped and any(kw in stripped for kw in ("def ", "import ", "print(", "for ", "while ", "if ")):
                code_blocks = [stripped]
                logger.debug(f"No code fences found for {problem.get('task_id')}, using raw response as code")
            else:
                result_entry["correctness"] = False
                result_entry["reason"] = "No code block found."
                return result_entry

        try:
            last_code = code_blocks[-1]
            problem_copy = copy.deepcopy(problem)
            result_list = lcb_run(
                problem=problem_copy,
                completion=post_process_code(last_code),
                timeout=timeout,
                is_extracted=not problem_copy["is_stdin"],
            )

            num_passed = sum(1 for r in result_list if r[0])
            result_entry["test_results"] = [1 if r[0] else 0 for r in result_list]
            result_entry["num_tests_passed"] = num_passed
            result_entry["num_tests_failed"] = len(result_list) - num_passed
            result_entry["correctness"] = all(r[0] for r in result_list)
            result_entry["reason"] = "" if result_entry["correctness"] else "Code is incorrect."

        except Exception as e:
            logger.error(f"Error evaluating solution: {e}")
            result_entry["correctness"] = False
            result_entry["reason"] = f"Evaluation error: {e}"

        return result_entry

    def evaluate_batch(self, problems: list[dict], solutions: list[list[str]]) -> list[list[dict]]:
        """Evaluate multiple solutions per problem in parallel.

        Args:
            problems: List of problem dicts.
            solutions: solutions[i][j] is the j-th solution for the i-th problem (as raw model output).

        Returns:
            results[i][j] is the evaluation result for the j-th solution of the i-th problem.
        """
        all_results: list[list[dict]] = [[] for _ in problems]

        for repeat_idx in range(len(solutions[0]) if solutions else 0):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for i, problem in enumerate(problems):
                    future = executor.submit(self.evaluate_solution, problem, solutions[i][repeat_idx])
                    futures[future] = i

                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "task_id": problems[i].get("task_id"),
                            "difficulty": problems[i].get("difficulty"),
                            "correctness": False,
                            "reason": f"Future error: {e}",
                            "test_results": [],
                            "num_tests_passed": 0,
                            "num_tests_failed": 0,
                        }
                    all_results[i].append(result)

        return all_results

    def compute_metrics(self, results: dict[str, list[list[int]]], k_values: list[int]) -> dict[str, Any]:
        """Compute pass@k overall and per-difficulty."""
        return compute_metrics_from_results(results, k_list=k_values)

    def format_prompt(self, problem: dict, tokenizer=None) -> str:
        """Format a problem into the prompt string."""
        if problem["is_stdin"]:
            prompt_text = LCB_PROMPT_WITHOUT_STARTER_CODE.format(problem_description=problem["prompt"])
        else:
            prompt_text = LCB_PROMPT_WITH_STARTER_CODE.format(
                problem_description=problem["prompt"],
                entry_point=problem["entry_point"],
            )

        if tokenizer is not None:
            messages = [{"role": "user", "content": prompt_text}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt_text
