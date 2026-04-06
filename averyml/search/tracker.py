"""Grid search progress tracking and resumption."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from averyml.search.temperature import TemperaturePoint
from averyml.utils.io import read_json, write_json

logger = logging.getLogger(__name__)


class SearchTracker:
    """Tracks grid search progress, enabling resumption of interrupted runs."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._progress_file = output_path / "search_progress.json"
        self._results_file = output_path / "search_results.json"
        self._completed: set[str] = set()
        self._results: list[dict[str, Any]] = []
        self._load()

    def _point_key(self, point: TemperaturePoint) -> str:
        return f"t_train={point.t_train}_t_eval={point.t_eval}"

    def _load(self):
        if self._progress_file.exists():
            data = read_json(self._progress_file)
            self._completed = set(data.get("completed", []))
            logger.info(f"Loaded {len(self._completed)} completed cells from {self._progress_file}")
        if self._results_file.exists():
            data = read_json(self._results_file)
            self._results = data.get("results", [])

    def _save(self):
        write_json({"completed": list(self._completed)}, self._progress_file)
        write_json({"results": self._results}, self._results_file)

    def is_complete(self, point: TemperaturePoint) -> bool:
        return self._point_key(point) in self._completed

    def mark_complete(self, point: TemperaturePoint, metrics: dict[str, Any]) -> None:
        key = self._point_key(point)
        self._completed.add(key)
        self._results.append({
            "t_train": point.t_train,
            "t_eval": point.t_eval,
            "t_eff": point.t_eff,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        })
        self._save()
        logger.info(f"Completed: {point}")

    def get_remaining(self, grid: list[TemperaturePoint]) -> list[TemperaturePoint]:
        return [p for p in grid if not self.is_complete(p)]

    def load_results(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame(self._results)
