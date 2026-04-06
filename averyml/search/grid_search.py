"""Grid search over (T_train, T_eval) temperature configurations.

Runs the full SSD pipeline (synthesis -> training -> evaluation) for each
cell in the temperature grid. This is central to reproducing the paper's
key finding about T_eff = T_train * T_eval (Figure 3, Section 3.4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from averyml.config.evaluation import EvaluationConfig
from averyml.config.search import SearchConfig
from averyml.config.synthesis import DecodingConfig, SynthesisConfig
from averyml.config.training import TrainingConfig
from averyml.evaluation.evaluator import Evaluator
from averyml.search.temperature import TemperaturePoint, build_grid, filter_diagonal_band
from averyml.search.tracker import SearchTracker
from averyml.synthesis.sampler import Sampler
from averyml.training.trainer import SSDTrainer

logger = logging.getLogger(__name__)


class GridSearch:
    """Runs the full SSD pipeline for each point in the temperature grid."""

    def __init__(self, config: SearchConfig, diagonal_only: bool = False):
        self.config = config
        self.diagonal_only = diagonal_only

    def run(self) -> pd.DataFrame:
        """Execute grid search. Returns DataFrame with per-cell results."""
        grid = build_grid(self.config)
        if self.diagonal_only:
            grid = filter_diagonal_band(grid)

        logger.info(f"Grid search: {len(grid)} cells to evaluate")

        output_path = Path(self.config.output_path)
        tracker = SearchTracker(output_path)
        remaining = tracker.get_remaining(grid)
        logger.info(f"Remaining: {len(remaining)} cells ({len(grid) - len(remaining)} already complete)")

        for i, point in enumerate(remaining):
            logger.info(f"\n{'='*60}")
            logger.info(f"Cell {i+1}/{len(remaining)}: {point}")
            logger.info(f"{'='*60}")

            try:
                metrics = self._run_cell(point, output_path)
                tracker.mark_complete(point, metrics)
            except Exception as e:
                logger.error(f"Cell failed: {point} - {e}")
                tracker.mark_complete(point, {"error": str(e)})

        return tracker.load_results()

    def _run_cell(self, point: TemperaturePoint, output_path: Path) -> dict:
        """Run the full SSD pipeline for a single grid cell."""
        cell_dir = output_path / f"t_train_{point.t_train}_t_eval_{point.t_eval}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Synthesis
        synth_config = SynthesisConfig(
            model_id=self.config.base_model_id,
            prompt_source=self.config.prompt_source,
            prompt_dataset=self.config.prompt_dataset,
            n_samples=self.config.n_samples,
            decoding=point.rho_train,
            output_path=str(cell_dir / "synthesis"),
        )
        dataset_path = Sampler(synth_config).run()

        # Step 2: Training
        train_config = TrainingConfig(
            model_id=self.config.base_model_id,
            dataset_path=str(dataset_path),
            output_dir=str(cell_dir / "checkpoints"),
            num_train_iterations=self.config.train_iterations,
            warmup_iterations=self.config.warmup_iterations,
        )
        checkpoint = SSDTrainer(train_config).run()

        # Step 3: Evaluation
        eval_config = EvaluationConfig(
            model_id=str(checkpoint),
            benchmark=self.config.benchmark,
            decoding=point.rho_eval,
            n_repeat=self.config.n_repeat,
            output_path=str(cell_dir / "results"),
        )
        metrics = Evaluator(eval_config).run()

        return metrics
