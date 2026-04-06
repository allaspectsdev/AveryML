"""Dashboard state management and data helpers."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DashboardState:
    """Holds directory paths and shared state for the dashboard."""

    def __init__(self, results_dir: str, configs_dir: str, search_dir: str):
        self.results_dir = Path(results_dir)
        self.configs_dir = Path(configs_dir)
        self.search_dir = Path(search_dir)


class JobRunner:
    """Manages subprocess execution of pipeline steps."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.log_file: Path | None = None
        self.command: str = ""
        self.start_time: float = 0

    def launch(self, cmd: list[str]) -> str:
        if self.process is not None and self.process.poll() is None:
            return "A job is already running. Wait for it to finish."

        self.log_file = Path(tempfile.mktemp(suffix=".log"))
        self.command = " ".join(cmd)
        self.start_time = time.time()

        with open(self.log_file, "w") as f:
            self.process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, text=True,
            )
        return f"Launched: {self.command} (PID {self.process.pid})"

    def get_logs(self) -> str:
        if self.log_file is None or not self.log_file.exists():
            return ""
        try:
            return self.log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def get_status(self) -> tuple[str, str]:
        """Returns (status_text, css_class)."""
        if self.process is None:
            return "No jobs running", "idle"
        rc = self.process.poll()
        elapsed = time.time() - self.start_time
        if rc is None:
            return f"Running: {self.command} ({elapsed:.0f}s)", "running"
        elif rc == 0:
            return f"Completed: {self.command} ({elapsed:.0f}s)", "complete"
        else:
            return f"Failed: {self.command} (exit {rc})", "failed"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_all_results(state: DashboardState) -> list[dict[str, Any]]:
    """Load all result JSON files with full data."""
    results = []
    if not state.results_dir.exists():
        return results
    for json_file in sorted(state.results_dir.rglob("results_*.json"), reverse=True):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            data["_path"] = str(json_file)
            results.append(data)
        except Exception:
            continue
    return results


def results_to_table(results: list[dict]) -> pd.DataFrame:
    """Convert loaded results into a display table."""
    if not results:
        return pd.DataFrame(columns=["Model", "Benchmark", "Date", "pass@1", "pass@5", "pass@10"])
    rows = []
    for r in results:
        cfg = r.get("config", {})
        res = r.get("results", {})
        ts = r.get("timestamp", 0)
        rows.append({
            "Model": cfg.get("model_id", "?").split("/")[-1],
            "Benchmark": cfg.get("benchmark", "?"),
            "Date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?",
            "pass@1": f"{res.get('pass@1', 0):.1%}" if isinstance(res.get("pass@1"), float) else "-",
            "pass@5": f"{res.get('pass@5', 0):.1%}" if isinstance(res.get("pass@5"), float) else "-",
            "pass@10": f"{res.get('pass@10', 0):.1%}" if isinstance(res.get("pass@10"), float) else "-",
        })
    return pd.DataFrame(rows)


def load_search_results(state: DashboardState) -> pd.DataFrame | None:
    """Load grid search results DataFrame."""
    results_file = state.search_dir / "search_results.json"
    if not results_file.exists():
        return None
    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
        rows = data.get("results", [])
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None


def list_configs(state: DashboardState) -> dict[str, list[str]]:
    """List all YAML configs organized by category."""
    configs: dict[str, list[str]] = {}
    if not state.configs_dir.exists():
        return configs
    for yaml_file in sorted(state.configs_dir.rglob("*.yaml")):
        rel = yaml_file.relative_to(state.configs_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "other"
        configs.setdefault(category, []).append(str(rel))
    return configs


def get_config_class(category: str):
    """Return the Pydantic config class for a category."""
    from averyml.config import (
        EvaluationConfig,
        ExperimentConfig,
        SearchConfig,
        SynthesisConfig,
        TrainingConfig,
    )
    return {
        "synthesis": SynthesisConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "experiments": ExperimentConfig,
        "search": SearchConfig,
    }.get(category)


def validate_config(yaml_text: str, category: str) -> str:
    """Validate YAML text against the appropriate Pydantic model."""
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return f"YAML parse error: {e}"

    cls = get_config_class(category)
    if cls is None:
        return "Valid YAML (no schema validation for this category)"

    try:
        cls.model_validate(data or {})
        return f"Valid {cls.__name__}"
    except Exception as e:
        return f"Validation error: {e}"
