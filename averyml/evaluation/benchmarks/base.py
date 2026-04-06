"""Abstract base class for code generation benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Benchmark(ABC):
    """Base interface for all code generation benchmarks."""

    @abstractmethod
    def load_problems(self) -> list[dict]:
        """Load and return all benchmark problems."""
        ...

    @abstractmethod
    def evaluate_solution(self, problem: dict, code: str, timeout: float) -> dict:
        """Evaluate a single solution against test cases."""
        ...

    @abstractmethod
    def compute_metrics(self, results: dict[str, list[list[int]]], k_values: list[int]) -> dict[str, Any]:
        """Compute pass@k and per-difficulty metrics from evaluation results."""
        ...
