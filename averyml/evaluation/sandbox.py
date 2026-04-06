"""Sandboxed code execution for evaluation.

Wraps the multiprocessing-based execution from livecodebench_utils
with a cleaner interface.
"""

from __future__ import annotations

import copy

from averyml.evaluation.benchmarks.livecodebench_utils import (
    has_code,
    lcb_run,
    post_process_code,
)


class CodeSandbox:
    """Multiprocessing-based sandboxed code execution."""

    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout

    def execute(self, problem: dict, code: str, is_extracted: bool) -> dict:
        """Execute code against test cases in an isolated subprocess.

        Returns dict with keys: all_passed, result_list, test_cases.
        """
        problem_copy = copy.deepcopy(problem)
        result_list = lcb_run(
            problem=problem_copy,
            completion=post_process_code(code),
            timeout=self.timeout,
            is_extracted=is_extracted,
        )
        return {
            "all_passed": all(r[0] for r in result_list),
            "result_list": result_list,
            "test_cases": problem_copy.get("test", []),
        }
