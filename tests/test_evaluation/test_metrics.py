"""Tests for pass@k metric computation."""

import numpy as np
import pytest

from averyml.evaluation.benchmarks.livecodebench_utils import (
    compare_strings_with_decimal_fallback,
    compute_metrics_from_results,
    estimate_pass_at_k,
    has_code,
    post_process_code,
)


class TestEstimatePassAtK:
    def test_all_correct(self):
        result = estimate_pass_at_k(np.array([10]), np.array([10]), 1)
        assert result[0] == pytest.approx(1.0)

    def test_none_correct(self):
        result = estimate_pass_at_k(np.array([10]), np.array([0]), 1)
        assert result[0] == pytest.approx(0.0)

    def test_half_correct_pass_at_1(self):
        result = estimate_pass_at_k(np.array([10]), np.array([5]), 1)
        assert 0.0 < result[0] < 1.0
        assert result[0] == pytest.approx(0.5)

    def test_pass_at_5_more_than_pass_at_1(self):
        p1 = estimate_pass_at_k(np.array([10]), np.array([3]), 1)
        p5 = estimate_pass_at_k(np.array([10]), np.array([3]), 5)
        assert p5[0] > p1[0]


class TestComputeMetrics:
    def test_perfect_results(self):
        results = {
            "task_1": [[1, 1], [1, 1]],
            "task_2": [[1, 1], [1, 1]],
        }
        metrics = compute_metrics_from_results(results, k_list=[1])
        assert metrics["pass@1"] == pytest.approx(1.0)

    def test_mixed_results(self):
        results = {
            "task_1": [[1, 1], [0, 1]],  # 1 correct, 1 incorrect
            "task_2": [[0, 0], [0, 0]],  # 0 correct
        }
        metrics = compute_metrics_from_results(results, k_list=[1])
        assert 0.0 < metrics["pass@1"] < 1.0


class TestCompareStrings:
    def test_exact_match(self):
        assert compare_strings_with_decimal_fallback("42", "42") is True

    def test_no_match(self):
        assert compare_strings_with_decimal_fallback("42", "43") is False

    def test_decimal_match(self):
        assert compare_strings_with_decimal_fallback("3.14", "3.14") is True

    def test_multiline(self):
        assert compare_strings_with_decimal_fallback("1\n2\n3", "1\n2\n3") is True

    def test_different_line_count(self):
        assert compare_strings_with_decimal_fallback("1\n2", "1\n2\n3") is False


class TestHasCode:
    def test_python_block(self):
        text = '```python\nprint("hello")\n```'
        result = has_code(text)
        assert len(result) == 1
        assert 'print("hello")' in result[0]

    def test_no_code(self):
        assert has_code("just some text") == []

    def test_multiple_blocks(self):
        text = '```python\nfirst\n```\n```python\nsecond\n```'
        result = has_code(text)
        assert len(result) == 2


class TestPostProcessCode:
    def test_strips_markdown(self):
        code = '```python\nprint("hello")\n```'
        result = post_process_code(code)
        assert "```" not in result
        assert 'print("hello")' in result
