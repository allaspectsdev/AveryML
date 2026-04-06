"""Result persistence and comparison for evaluation runs."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from averyml.utils.io import read_json, write_json


class ResultStore:
    """Save, load, list, and compare evaluation results."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)

    def save(self, results: dict[str, Any], config: dict, metadata: dict | None = None) -> Path:
        """Save evaluation results to a timestamped JSON file."""
        model_name = config.get("model_id", "unknown").replace("/", "_")
        result_dir = self.base_path / model_name
        result_dir.mkdir(parents=True, exist_ok=True)

        result_file = result_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
        data = {
            "results": {k: v for k, v in results.items() if k != "detail"},
            "config": config,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        write_json(data, result_file)
        return result_file

    def load(self, result_path: Path) -> dict[str, Any]:
        """Load results from a JSON file."""
        return read_json(result_path)

    def list_results(self) -> list[dict[str, Any]]:
        """List all saved results under the base path."""
        entries = []
        for json_file in sorted(self.base_path.rglob("results_*.json")):
            try:
                data = read_json(json_file)
                entries.append({
                    "path": str(json_file),
                    "model": data.get("config", {}).get("model_id", "?"),
                    "date": datetime.fromtimestamp(data.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M"),
                    "pass@1": data.get("results", {}).get("pass@1"),
                })
            except Exception:
                continue
        return entries

    def compare(self, result_paths: list[Path]) -> pd.DataFrame:
        """Compare multiple result files side by side."""
        rows = []
        for path in result_paths:
            data = self.load(path)
            row = {"file": path.name, "model": data.get("config", {}).get("model_id", "?")}
            results = data.get("results", {})
            for key, value in results.items():
                if isinstance(value, float):
                    row[key] = value
            rows.append(row)
        return pd.DataFrame(rows)
