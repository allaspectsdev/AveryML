"""LiveCodeBench evaluation utilities.

Adapted from Apple's ml-ssd reference implementation and the official
LiveCodeBench testing_util.py. Provides sandboxed code execution, stdin
mocking, and correctness comparison for code generation evaluation.
"""

from __future__ import annotations

import ast
import base64
import copy
import faulthandler
import io
import json
import multiprocessing
import pickle
import re
import sys
import time
import zlib
from decimal import Decimal
from types import ModuleType
from unittest.mock import mock_open, patch

import numpy as np

BASE_IMPORTS = (
    "from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat\n"
    "from copy import deepcopy\n"
    "from string import ascii_lowercase\n"
    "from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt, log, prod\n"
    "from collections import defaultdict, deque, Counter, OrderedDict\n"
    "from bisect import bisect, bisect_left, bisect_right, insort\n"
    "from heapq import heappush, heappop, heapify, merge\n"
    "from functools import reduce, cache, lru_cache, partial\n"
    "from random import randrange, shuffle\n"
    "from operator import itemgetter, sub, iand\n"
    "from re import search as re_search\n"
    "from os.path import commonprefix\n"
    "from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator\n"
    "from itertools import zip_longest, cycle\n"
    "import copy, string, math, collections, bisect, heapq, functools, random, itertools, operator, re, sys\n"
    "import numpy as np\n"
    "import pandas as pd\n"
)


# ---------------------------------------------------------------------------
# Stdin/stdout capture helpers
# ---------------------------------------------------------------------------


class Capturing(list):
    """Context manager to capture stdout as a list of strings."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        self._stringio.close = lambda: 1  # type: ignore[assignment]
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    """Mock for sys.stdin.buffer with byte string support."""

    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    """Custom mock for sys.stdin that supports .buffer, .read(), .readline(), iteration."""

    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = io.StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __iter__(self):
        return iter(self._stringio)

    def __next__(self):
        return next(self._stringio)

    def __getattr__(self, name):
        return getattr(self._stringio, name)


# ---------------------------------------------------------------------------
# String comparison
# ---------------------------------------------------------------------------


def get_stripped_lines(val: str) -> list[str]:
    """Strip value and each line individually."""
    val = val.strip()
    return [line.strip() for line in val.split("\n")]


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    """Convert space-separated values to Decimal for precise numeric comparison."""
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def compare_strings_with_decimal_fallback(prediction_str: str, expected_str: str) -> bool:
    """Compare outputs with exact match first, then Decimal fallback for numerics."""
    stripped_pred = get_stripped_lines(prediction_str)
    stripped_exp = get_stripped_lines(expected_str)

    if len(stripped_pred) != len(stripped_exp):
        return False

    for pred_line, exp_line in zip(stripped_pred, stripped_exp):
        if pred_line == exp_line:
            continue
        ok_pred, dec_pred = convert_line_to_decimals(pred_line)
        if not ok_pred:
            return False
        ok_exp, dec_exp = convert_line_to_decimals(exp_line)
        if not ok_exp:
            return False
        if dec_pred != dec_exp:
            return False

    return True


# ---------------------------------------------------------------------------
# Security guard
# ---------------------------------------------------------------------------


def reliability_guard():
    """Disable destructive functions to protect the host during code execution.

    WARNING: This is NOT a security sandbox. Do not run untrusted code without
    proper containerization.
    """
    faulthandler.disable()

    import builtins

    builtins.exit = None  # type: ignore[assignment]
    builtins.quit = None  # type: ignore[assignment]

    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    for attr in [
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ]:
        if hasattr(os, attr):
            setattr(os, attr, None)

    import shutil

    shutil.rmtree = None  # type: ignore[assignment]
    shutil.move = None  # type: ignore[assignment]
    shutil.chown = None  # type: ignore[assignment]

    import subprocess

    subprocess.Popen = None  # type: ignore[assignment]

    sys.modules["ipdb"] = None  # type: ignore[assignment]
    sys.modules["joblib"] = None  # type: ignore[assignment]
    sys.modules["resource"] = None  # type: ignore[assignment]
    sys.modules["psutil"] = None  # type: ignore[assignment]
    sys.modules["tkinter"] = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Test case helpers
# ---------------------------------------------------------------------------


def has_test_type(tests: str, test_type: str) -> bool:
    """Check if any test case has the given testtype."""
    for test in json.loads(tests):
        if test.get("testtype") == test_type:
            return True
    return False


def translate_private_test_cases(encoded_data: str) -> list[dict]:
    """Decode base64 + zlib + pickle encoded test cases."""
    decoded = base64.b64decode(encoded_data)
    decompressed = zlib.decompress(decoded)
    original = pickle.loads(decompressed)  # noqa: S301
    return json.loads(original)


def map_to_example(row: dict) -> dict:
    """Map a HuggingFace dataset row to our standardized example format."""
    metadata_raw = row.get("metadata", "{}")
    try:
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    except json.JSONDecodeError:
        metadata = {}

    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
        "metadata": metadata,
    }


def post_process_code(code: str) -> str:
    """Extract code from markdown code blocks."""
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    return code


def has_code(response: str) -> list[str]:
    """Extract code blocks from a model response."""
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    return re.findall(pattern, response, re.DOTALL)


# ---------------------------------------------------------------------------
# AST-based code transformation
# ---------------------------------------------------------------------------


def parse_function_name_from_starter_code(starter_code: str) -> str | None:
    """Extract function name from starter code using AST parsing."""
    try:
        code_to_parse = starter_code
        if not code_to_parse.strip().endswith(("pass", "...", "return")):
            lines = code_to_parse.rstrip().split("\n")
            if lines and lines[-1].rstrip().endswith(":"):
                indent = len(lines[-1]) - len(lines[-1].lstrip()) + 4
                code_to_parse = code_to_parse + "\n" + " " * indent + "pass"

        tree = ast.parse(code_to_parse)
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fn = node.name
        return fn
    except Exception:
        return None


def clean_if_name(code: str) -> str:
    """Remove 'if __name__ == \"__main__\":' wrapper from code."""
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    """Wrap code inside a wrapped_function() for controlled execution with stdin mocking."""
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts if all_other_stmts else [ast.Pass()],
            decorator_list=[],
            lineno=-1,
        )
        return BASE_IMPORTS + "\n" + ast.unparse(import_stmts) + "\n" + ast.unparse(function_ast)
    except Exception:
        return code


def compile_code(code: str) -> ModuleType | None:
    """Compile code into a module, handling class Solution wrapping."""
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)  # noqa: S102
        if "class Solution" in code:
            return tmp_sol.Solution()
        return tmp_sol
    except Exception:
        return None


def get_function(compiled_sol, fn_name: str):
    """Safely extract function from compiled module."""
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def call_method(method, inputs):
    """Call method with comprehensive stdin mocking."""
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass

    return _inner_call_method(method)


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


def prepare_test_input_output_std(test_case: dict) -> tuple[str, str]:
    """Prepare test input/output for stdin-based tests."""
    return test_case["input"], test_case["output"].strip()


def run_test_func(completion: str, is_extracted: bool, test_input, test_output, func_name: str):
    """Run function-based test with string comparison + Decimal fallback."""
    assert func_name is not None, "func_name must be provided"

    namespace = {}
    exec(completion, namespace)  # noqa: S102

    is_class_based = "class Solution:" in completion or "class Solution(" in completion

    saved_stdout = sys.stdout
    output = io.StringIO()
    sys.stdout = output

    try:
        if is_class_based:
            callable_func = getattr(namespace["Solution"](), func_name)
        else:
            callable_func = namespace[func_name]

        if not is_extracted:
            prediction = callable_func(**test_input) if isinstance(test_input, dict) else callable_func(test_input)
        else:
            prediction = callable_func(*test_input)

        if isinstance(prediction, tuple):
            prediction = list(prediction)

        prediction_str = str(prediction) if not isinstance(prediction, str) else prediction
        expected_str = str(test_output) if not isinstance(test_output, str) else test_output

        if compare_strings_with_decimal_fallback(prediction_str, expected_str):
            return True, prediction
        return False, prediction

    except Exception as e:
        return False, f"Error: {e}" if not is_extracted else str(e)
    finally:
        sys.stdout = saved_stdout


def run_test_std(completion: str, test_input: str, test_output: str):
    """Run stdin-based test using AST-based code transformation."""
    completion = clean_if_name(completion)
    completion = make_function(completion)

    compiled_sol = compile_code(completion)
    if compiled_sol is None:
        return False, "Compilation failed"

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return False, "Could not find wrapped_function"

    with Capturing() as captured_output:
        try:
            call_method(method, test_input)
        except Exception as e:
            return False, f"Runtime error: {e}"

    prediction = captured_output[0] if captured_output else ""

    if compare_strings_with_decimal_fallback(prediction, test_output):
        return True, prediction.strip()
    return False, prediction.strip()


def prepare_test_input_output_functional(test_case: dict, is_extracted: bool):
    """Prepare test input/output for function-based tests."""
    if not is_extracted:
        return test_case["input"], test_case["output"]

    input_str = test_case["input"]
    expected_output = test_case["output"].strip()
    inputs = []

    if "=" in input_str:
        parts = input_str.split(",") if "," in input_str else [input_str]
        for part in parts:
            key, value = map(str.strip, part.split("=", 1))
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    value = value.strip('"')
            inputs.append(value)
    else:
        for line in input_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                inputs.append(line.strip('"'))
                continue
            if line.startswith("[") and line.endswith("]"):
                inputs.append(json.loads(line))
                continue
            try:
                inputs.append(int(line))
            except ValueError:
                try:
                    inputs.append(float(line))
                except ValueError:
                    inputs.append(line)

    try:
        expected_output = json.loads(expected_output)
    except json.JSONDecodeError:
        expected_output = expected_output.strip()

    return inputs, expected_output


def run_tests_for_one_example(problem, test_cases, completion, result_list, is_extracted):
    """Run all test cases for a single problem in an isolated process."""
    reliability_guard()
    completion = BASE_IMPORTS + "\n" + completion

    func_name = None
    test_type = test_cases[0]["testtype"]
    if test_type == "functional":
        metadata = problem.get("metadata", {})
        func_name = metadata.get("func_name")
        if not func_name and "entry_point" in problem:
            func_name = parse_function_name_from_starter_code(problem["entry_point"])

    for test_case in test_cases:
        output_error = ""
        output_value = ""
        test_input = None
        test_output = None
        try:
            time_start = time.time()
            if test_type == "functional":
                test_input, test_output = prepare_test_input_output_functional(test_case, is_extracted)
                passed, output_value = run_test_func(
                    completion, is_extracted, copy.deepcopy(test_input), copy.deepcopy(test_output), func_name
                )
            else:
                test_input, test_output = prepare_test_input_output_std(test_case)
                passed, output_value = run_test_std(
                    completion, copy.deepcopy(test_input), copy.deepcopy(test_output)
                )
            time_elapsed = time.time() - time_start

            if not passed:
                output_error = (
                    f"For test input: {test_input}. Expected: {test_output}, got: {output_value}."
                )
        except Exception as e:
            passed = False
            time_elapsed = float("inf")
            output_error = f"For test input: {test_input}. Expected: {test_output}, error: {e}."
            output_value = f"Error: {e}."

        if not output_error:
            output_error = (
                f"For test input: {test_input}. Expected: {test_output}, "
                f"correctly produced: {output_value}."
            )

        result_list.append((passed, output_error, output_value, time_elapsed))
        if not passed:
            return


def lcb_run(problem: dict, completion: str, timeout: float, is_extracted: bool) -> list:
    """Run tests in an isolated subprocess with timeout."""
    test_cases = problem["test"]
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(
        target=run_tests_for_one_example,
        args=(problem, test_cases, completion, result, is_extracted),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(test_cases) + 5)
    if p.is_alive():
        p.kill()

    # Fill in timeout results for remaining test cases
    for _ in range(len(test_cases) - len(result)):
        result.append((False, "Timed out.", "Error: Timed out!", float("inf")))

    return list(result)


# ---------------------------------------------------------------------------
# pass@k metric
# ---------------------------------------------------------------------------


def estimate_pass_at_k(num_samples, num_correct, k) -> np.ndarray:
    """Unbiased estimator for pass@k: 1 - C(n-c, k) / C(n, k)."""

    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        import itertools

        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def compute_metrics_from_results(
    results: dict[str, list[list[int]]], k_list: list[int] = [1, 5]
) -> dict:
    """Compute pass@k metrics from per-task, per-generation test results."""
    total = []
    correct = []
    task_ids = []

    for task_id, res in results.items():
        all_correct = [np.all(np.array(gen) > 0) for gen in res]
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))

    total_arr = np.array(total)
    correct_arr = np.array(correct)

    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total_arr, correct_arr, k).tolist()
        for k in k_list
        if (total_arr >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total_arr, correct_arr, k).mean()
        for k in k_list
        if (total_arr >= k).all()
    }
    pass_at_k["detail"] = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    return pass_at_k
