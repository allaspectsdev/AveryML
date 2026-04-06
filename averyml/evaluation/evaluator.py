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
        all_outputs = self._generate_solutions(problems)

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

    def _generate_solutions(self, problems: list[dict]) -> list[list[str]]:
        """Generate n_repeat solutions per problem using vLLM."""
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        logger.info(f"Loading model: {self.config.model_id}")
        llm = LLM(
            model=self.config.model_id,
            tensor_parallel_size=self.config.tensor_parallel_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        benchmark = LiveCodeBench(version=self.config.benchmark)
        stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        # all_outputs[problem_idx][repeat_idx] = model output text
        all_outputs: list[list[str]] = [[] for _ in problems]

        for i in range(self.config.n_repeat):
            seed = self.config.seeds[i % len(self.config.seeds)]

            prompts = [benchmark.format_prompt(p, tokenizer) for p in problems]

            sampling_params = SamplingParams(
                max_tokens=self.config.max_tokens,
                seed=seed,
                stop_token_ids=stop_token_ids,
                temperature=self.config.decoding.temperature,
                top_k=self.config.decoding.top_k,
                top_p=self.config.decoding.top_p,
                min_p=self.config.decoding.min_p,
            )

            logger.info(f"Generating solutions (repeat {i + 1}/{self.config.n_repeat})...")
            outputs = llm.generate(prompts, sampling_params)
            texts = [o.outputs[0].text for o in outputs]

            for j, text in enumerate(texts):
                all_outputs[j].append(text)

        return all_outputs

    def _evaluate_solutions(
        self,
        problems: list[dict],
        all_outputs: list[list[str]],
        benchmark: LiveCodeBench,
    ) -> tuple[dict[str, list[list[int]]], dict[str, str]]:
        """Evaluate all generated solutions and return results grouped by task_id."""
        results_by_task: dict[str, list[list[int]]] = defaultdict(list)
        task_difficulty: dict[str, str] = {}

        logger.info(f"Evaluating {len(problems)} problems x {self.config.n_repeat} repeats...")

        for repeat_idx in range(self.config.n_repeat):
            batch_results = []
            for i, problem in enumerate(problems):
                result = benchmark.evaluate_solution(problem, all_outputs[i][repeat_idx])
                batch_results.append(result)

            for i, result in enumerate(batch_results):
                task_id = result["task_id"]
                task_difficulty[task_id] = problems[i].get("difficulty", "unknown")
                test_results = result.get("test_results", [])
                if test_results:
                    results_by_task[task_id].append(test_results)
                else:
                    num_tests = max(len(problems[i].get("test", [])), 1)
                    results_by_task[task_id].append([0] * num_tests)

        return dict(results_by_task), task_difficulty
