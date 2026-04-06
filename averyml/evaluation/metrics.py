"""Metric computation utilities for code generation evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from averyml.evaluation.benchmarks.livecodebench_utils import (
    compute_metrics_from_results,
    estimate_pass_at_k,
)


def compute_pass_at_k_with_difficulty(
    results_by_task: dict[str, list[list[int]]],
    task_difficulty: dict[str, str],
    k_values: list[int],
) -> dict[str, Any]:
    """Compute pass@k overall and broken down by difficulty level.

    Args:
        results_by_task: task_id -> list of per-generation test results (list of 0/1).
        task_difficulty: task_id -> difficulty string (e.g., "easy", "medium", "hard").
        k_values: List of k values for pass@k.

    Returns:
        Dict with overall pass@k and per-difficulty pass@k.
    """
    metrics = {}

    # Overall
    overall = compute_metrics_from_results(results_by_task, k_list=k_values)
    for k in k_values:
        key = f"pass@{k}"
        if key in overall:
            metrics[key] = overall[key]

    # Per difficulty
    by_difficulty: dict[str, dict[str, list]] = defaultdict(dict)
    for task_id, results in results_by_task.items():
        diff = task_difficulty.get(task_id, "unknown")
        by_difficulty[diff][task_id] = results

    for diff, diff_results in sorted(by_difficulty.items()):
        diff_metrics = compute_metrics_from_results(diff_results, k_list=k_values)
        for k in k_values:
            key = f"pass@{k}"
            if key in diff_metrics:
                metrics[f"pass@{k}_{diff}"] = diff_metrics[key]

    return metrics


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """Format metrics as a readable table string."""
    lines = []
    lines.append(f"{'Metric':<25} {'Value':>10}")
    lines.append("-" * 37)

    # Overall metrics first
    for k in [1, 5, 10, 16, 20, 32]:
        key = f"pass@{k}"
        if key in metrics and isinstance(metrics[key], float):
            lines.append(f"{key:<25} {metrics[key]:>9.2%}")

    # Per-difficulty
    for k in [1, 5, 10, 16, 20, 32]:
        for diff in ["easy", "medium", "hard"]:
            key = f"pass@{k}_{diff}"
            if key in metrics and isinstance(metrics[key], float):
                lines.append(f"  {key:<23} {metrics[key]:>9.2%}")

    return "\n".join(lines)
