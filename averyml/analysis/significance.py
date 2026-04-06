"""Statistical significance testing for SSD experiments.

Provides bootstrap confidence intervals and permutation tests
for comparing base vs SSD model metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for the mean.

    Returns dict with: mean, ci_lower, ci_upper, std.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "std": float(values.std()),
    }


def bootstrap_delta_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for the difference in means (B - A).

    Returns dict with: delta, ci_lower, ci_upper, significant.
    """
    rng = np.random.RandomState(seed)
    n_a, n_b = len(values_a), len(values_b)
    deltas = []
    for _ in range(n_bootstrap):
        boot_a = rng.choice(values_a, size=n_a, replace=True).mean()
        boot_b = rng.choice(values_b, size=n_b, replace=True).mean()
        deltas.append(boot_b - boot_a)

    deltas = np.array(deltas)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    return {
        "delta": float(values_b.mean() - values_a.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": ci_lower > 0 or ci_upper < 0,  # CI doesn't cross 0
    }


def permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sample permutation test for difference in means.

    Returns dict with: observed_delta, p_value.
    """
    rng = np.random.RandomState(seed)
    observed = values_b.mean() - values_a.mean()
    combined = np.concatenate([values_a, values_b])
    n_a = len(values_a)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_delta = combined[n_a:].mean() - combined[:n_a].mean()
        if abs(perm_delta) >= abs(observed):
            count_extreme += 1

    return {
        "observed_delta": float(observed),
        "p_value": float((count_extreme + 1) / (n_permutations + 1)),
    }


def cohens_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    pooled_std = np.sqrt((values_a.var() + values_b.var()) / 2)
    if pooled_std == 0:
        return 0.0
    return float((values_b.mean() - values_a.mean()) / pooled_std)


def compare_metrics(
    base_per_task: dict[str, float],
    ssd_per_task: dict[str, float],
    metric_name: str = "pass@1",
) -> dict[str, Any]:
    """Full statistical comparison between base and SSD per-task metrics.

    Args:
        base_per_task: {task_id: metric_value} for base model
        ssd_per_task: {task_id: metric_value} for SSD model

    Returns dict with bootstrap CI, permutation test p-value, and Cohen's d.
    """
    # Align on shared tasks
    shared_tasks = sorted(set(base_per_task.keys()) & set(ssd_per_task.keys()))
    if not shared_tasks:
        return {"error": "No shared tasks between base and SSD results"}

    base_values = np.array([base_per_task[t] for t in shared_tasks])
    ssd_values = np.array([ssd_per_task[t] for t in shared_tasks])

    base_ci = bootstrap_ci(base_values)
    ssd_ci = bootstrap_ci(ssd_values)
    delta_ci = bootstrap_delta_ci(base_values, ssd_values)
    perm = permutation_test(base_values, ssd_values)
    effect = cohens_d(base_values, ssd_values)

    return {
        "metric": metric_name,
        "n_tasks": len(shared_tasks),
        "base": base_ci,
        "ssd": ssd_ci,
        "delta": delta_ci,
        "permutation_test": perm,
        "cohens_d": effect,
    }
