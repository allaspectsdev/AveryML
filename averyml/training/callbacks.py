"""Training callbacks for evaluation and experiment tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EvalCallback:
    """Runs evaluation at checkpoint save points during training."""

    def __init__(self, eval_config: Any):
        self.eval_config = eval_config

    def on_save(self, checkpoint_path: Path) -> dict:
        """Run evaluation on the saved checkpoint."""
        from averyml.evaluation.evaluator import Evaluator

        config = self.eval_config.model_copy(update={"model_id": str(checkpoint_path)})
        logger.info(f"Running eval callback on {checkpoint_path}")
        return Evaluator(config).run()
