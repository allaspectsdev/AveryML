"""Tests for the dashboard module (no gradio required)."""

import json
from pathlib import Path

import pandas as pd

from averyml.dashboard.state import (
    DashboardState,
    list_configs,
    load_all_results,
    load_search_results,
    results_to_table,
    validate_config,
)


class TestDashboardState:
    def test_construction(self, tmp_path):
        state = DashboardState(str(tmp_path / "r"), str(tmp_path / "c"), str(tmp_path / "s"))
        assert state.results_dir == tmp_path / "r"


class TestLoadAllResults:
    def test_empty_dir(self, tmp_path):
        state = DashboardState(str(tmp_path), str(tmp_path), str(tmp_path))
        assert load_all_results(state) == []

    def test_nonexistent_dir(self, tmp_path):
        state = DashboardState(str(tmp_path / "nope"), str(tmp_path), str(tmp_path))
        assert load_all_results(state) == []

    def test_loads_results(self, tmp_path):
        result_dir = tmp_path / "results" / "test_model"
        result_dir.mkdir(parents=True)
        data = {"results": {"pass@1": 0.55}, "config": {"model_id": "test"}, "timestamp": 1700000000}
        (result_dir / "results_20240101_120000.json").write_text(json.dumps(data))

        state = DashboardState(str(tmp_path / "results"), str(tmp_path), str(tmp_path))
        results = load_all_results(state)
        assert len(results) == 1
        assert results[0]["results"]["pass@1"] == 0.55


class TestResultsToTable:
    def test_empty(self):
        df = results_to_table([])
        assert len(df) == 0
        assert "Model" in df.columns

    def test_formats_correctly(self):
        results = [{
            "config": {"model_id": "test/model", "benchmark": "lcb_v6"},
            "results": {"pass@1": 0.5512, "pass@5": 0.7},
            "timestamp": 1700000000,
            "_path": "/some/path",
        }]
        df = results_to_table(results)
        assert len(df) == 1
        assert df.iloc[0]["Model"] == "model"  # split("/")[-1]
        assert "55.1%" in df.iloc[0]["pass@1"]


class TestLoadSearchResults:
    def test_no_file(self, tmp_path):
        state = DashboardState(str(tmp_path), str(tmp_path), str(tmp_path))
        assert load_search_results(state) is None

    def test_loads_results(self, tmp_path):
        data = {"results": [
            {"t_train": 1.0, "t_eval": 0.8, "t_eff": 0.8, "pass@1": 0.42},
            {"t_train": 1.5, "t_eval": 0.8, "t_eff": 1.2, "pass@1": 0.55},
        ]}
        (tmp_path / "search_results.json").write_text(json.dumps(data))

        state = DashboardState(str(tmp_path), str(tmp_path), str(tmp_path))
        df = load_search_results(state)
        assert df is not None
        assert len(df) == 2
        assert "t_eff" in df.columns


class TestListConfigs:
    def test_lists_yaml_files(self, tmp_path):
        (tmp_path / "synthesis").mkdir()
        (tmp_path / "synthesis" / "default.yaml").write_text("model_id: test")
        (tmp_path / "training").mkdir()
        (tmp_path / "training" / "sft.yaml").write_text("model_id: test")

        state = DashboardState(str(tmp_path), str(tmp_path), str(tmp_path))
        configs = list_configs(state)
        assert "synthesis" in configs
        assert "training" in configs
        assert len(configs["synthesis"]) == 1


class TestValidateConfig:
    def test_valid_synthesis(self):
        yaml_text = "model_id: test/model\nbackend: vllm"
        result = validate_config(yaml_text, "synthesis")
        assert "Valid" in result

    def test_invalid_yaml(self):
        result = validate_config("{{bad yaml", "synthesis")
        assert "error" in result.lower()

    def test_unknown_category(self):
        result = validate_config("key: value", "unknown_category")
        assert "no schema" in result.lower()
