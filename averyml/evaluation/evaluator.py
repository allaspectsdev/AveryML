"""Evaluator orchestrator: generates solutions and evaluates them on benchmarks."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from averyml.config.evaluation import EvaluationConfig
from averyml.evaluation.benchmarks.livecodebench import LiveCodeBench
from averyml.evaluation.metrics import compute_pass_at_k_with_difficulty, format_metrics_table
from averyml.evaluation.results import ResultStore
from averyml.utils.registry import synthesis_backend_registry

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates Step 3 of SSD: load model, generate solutions, evaluate on benchmarks."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        """Execute the full evaluation pipeline."""
        benchmark = self._build_benchmark()
        problems = benchmark.load_problems()
        logger.info(f"Loaded {len(problems)} problems from {self.config.benchmark}")

        # Generate solutions
        all_outputs = self._generate_solutions(problems, benchmark)

        # Evaluate
        results_by_task, task_difficulty = self._evaluate_solutions(problems, all_outputs, benchmark)

        # Compute metrics
        metrics = compute_pass_at_k_with_difficulty(
            results_by_task, task_difficulty, self.config.k_values
        )

        # Log summary
        logger.info("\n" + format_metrics_table(metrics))

        # Save
        store = ResultStore(Path(self.config.output_path))
        result_path = store.save(metrics, self.config.model_dump())
        logger.info(f"Results saved to {result_path}")

        metrics["num_problems"] = len(problems)
        metrics["n_repeat"] = self.config.n_repeat
        return metrics

    def _build_benchmark(self) -> LiveCodeBench:
        """Instantiate the benchmark."""
        return LiveCodeBench(
            version=self.config.benchmark,
            max_workers=self.config.max_workers,
            timeout_per_test=self.config.timeout_per_test,
        )

    def _generate_solutions(self, problems: list[dict], benchmark: LiveCodeBench) -> list[list[str]]:
        """Generate n_repeat solutions per problem using the configured backend."""
        import time

        from tqdm import tqdm

        # Reuse synthesis backend infrastructure for generation
        backend_cls = synthesis_backend_registry.get(self.config.backend)
        if self.config.backend == "vllm":
            backend = backend_cls(tensor_parallel_size=self.config.tensor_parallel_size)
        else:
            backend = backend_cls()

        logger.info(f"Loading model: {self.config.model_id} (backend={self.config.backend})")
        backend.load_model(self.config.model_id)

        # all_outputs[problem_idx][repeat_idx] = model output text
        all_outputs: list[list[str]] = [[] for _ in problems]

        for i in tqdm(range(self.config.n_repeat), desc="Generation repeats"):
            seed = self.config.seeds[i % len(self.config.seeds)]
            prompts = [benchmark.format_prompt(p, backend.tokenizer) for p in problems]

            start = time.time()
            texts = backend.generate(prompts, self.config.decoding, self.config.max_tokens, seed)
            elapsed = time.time() - start
            logger.info(
                f"Repeat {i + 1}/{self.config.n_repeat}: "
                f"{len(texts)} solutions in {elapsed:.1f}s"
            )

            for j, text in enumerate(texts):
                all_outputs[j].append(text)

        backend.cleanup()
        return all_outputs

    def _evaluate_solutions(
        self,
        problems: list[dict],
        all_outputs: list[list[str]],
        benchmark: LiveCodeBench,
    ) -> tuple[dict[str, list[list[int]]], dict[str, str]]:
        """Evaluate all generated solutions and return results grouped by task_id."""
        from tqdm import tqdm

        from averyml.evaluation.benchmarks.livecodebench_utils import has_code

        results_by_task: dict[str, list[list[int]]] = defaultdict(list)
        task_difficulty: dict[str, str] = {}
        no_code_count = 0
        total_evals = 0

        logger.info(f"Evaluating {len(problems)} problems x {self.config.n_repeat} repeats...")

        for repeat_idx in tqdm(range(self.config.n_repeat), desc="Evaluation repeats"):
            for i, problem in enumerate(problems):
                result = benchmark.evaluate_solution(problem, all_outputs[i][repeat_idx])
                total_evals += 1

                if result.get("reason") == "No code block found.":
                    no_code_count += 1

                task_id = result["task_id"]
                task_difficulty[task_id] = problems[i].get("difficulty", "unknown")
                test_results = result.get("test_results", [])
                if test_results:
                    results_by_task[task_id].append(test_results)
                else:
                    num_tests = max(len(problems[i].get("test", [])), 1)
                    results_by_task[task_id].append([0] * num_tests)

        if no_code_count > 0:
            pct = no_code_count / total_evals * 100
            logger.warning(
                f"Code extraction failed for {no_code_count}/{total_evals} solutions ({pct:.1f}%). "
                f"The model may not be wrapping code in ```python blocks. "
                f"Consider adjusting the prompt template."
            )

        return dict(results_by_task), task_difficulty
