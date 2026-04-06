"""Tests for significance testing utilities."""

import numpy as np
import pytest

from averyml.analysis.significance import (
    bootstrap_ci,
    bootstrap_delta_ci,
    cohens_d,
    compare_metrics,
    permutation_test,
)


class TestBootstrapCI:
    def test_narrow_ci_for_constant(self):
        values = np.ones(100)
        result = bootstrap_ci(values)
        assert result["mean"] == pytest.approx(1.0)
        assert result["ci_lower"] == pytest.approx(1.0)
        assert result["ci_upper"] == pytest.approx(1.0)

    def test_ci_contains_mean(self):
        rng = np.random.RandomState(42)
        values = rng.normal(0.5, 0.1, 200)
        result = bootstrap_ci(values)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_wider_ci_for_noisy_data(self):
        rng = np.random.RandomState(42)
        tight = bootstrap_ci(rng.normal(0, 0.01, 100))
        wide = bootstrap_ci(rng.normal(0, 1.0, 100))
        assert (wide["ci_upper"] - wide["ci_lower"]) > (tight["ci_upper"] - tight["ci_lower"])


class TestBootstrapDeltaCI:
    def test_significant_difference(self):
        a = np.zeros(100)
        b = np.ones(100)
        result = bootstrap_delta_ci(a, b)
        assert result["delta"] == pytest.approx(1.0)
        assert result["significant"] is True
        assert result["ci_lower"] > 0

    def test_no_difference(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0.5, 0.1, 100)
        b = rng.normal(0.5, 0.1, 100)
        result = bootstrap_delta_ci(a, b)
        assert abs(result["delta"]) < 0.1


class TestPermutationTest:
    def test_obvious_difference(self):
        a = np.zeros(50)
        b = np.ones(50)
        result = permutation_test(a, b)
        assert result["p_value"] < 0.01

    def test_no_difference(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        result = permutation_test(a, b)
        assert result["p_value"] > 0.05


class TestCohensD:
    def test_zero_effect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_large_effect(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0.0, 0.1, 100)
        b = rng.normal(1.0, 0.1, 100)
        d = cohens_d(a, b)
        assert d > 0.8  # large effect


class TestCompareMetrics:
    def test_full_comparison(self):
        rng = np.random.RandomState(42)
        base = {f"t{i}": rng.normal(0.3, 0.05) for i in range(20)}
        ssd = {f"t{i}": rng.normal(0.6, 0.05) for i in range(20)}
        result = compare_metrics(base, ssd)
        assert result["n_tasks"] == 20
        assert result["delta"]["significant"] is True
        assert result["cohens_d"] > 0

    def test_no_shared_tasks(self):
        base = {"a": 1.0}
        ssd = {"b": 1.0}
        result = compare_metrics(base, ssd)
        assert "error" in result
