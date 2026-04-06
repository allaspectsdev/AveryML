"""Tests for ResultStore save/load/list/compare."""

import json
import time
from pathlib import Path

from averyml.evaluation.results import ResultStore


class TestResultStore:
    def test_save_and_load(self, tmp_path):
        store = ResultStore(tmp_path)
        results = {"pass@1": 0.55, "pass@5": 0.70}
        config = {"model_id": "test/model", "benchmark": "livecodebench_v6"}

        path = store.save(results, config)
        assert path.exists()

        loaded = store.load(path)
        assert loaded["results"]["pass@1"] == 0.55
        assert loaded["config"]["model_id"] == "test/model"

    def test_list_results(self, tmp_path):
        store = ResultStore(tmp_path)
        # Save two results
        store.save({"pass@1": 0.42}, {"model_id": "model_a"})
        time.sleep(0.01)
        store.save({"pass@1": 0.55}, {"model_id": "model_b"})

        entries = store.list_results()
        assert len(entries) == 2
        assert any("model_a" in e["model"] for e in entries)
        assert any("model_b" in e["model"] for e in entries)

    def test_list_empty(self, tmp_path):
        store = ResultStore(tmp_path / "nonexistent")
        entries = store.list_results()
        assert entries == []

    def test_compare(self, tmp_path):
        store = ResultStore(tmp_path)
        p1 = store.save({"pass@1": 0.42, "pass@5": 0.60}, {"model_id": "base"})
        p2 = store.save({"pass@1": 0.55, "pass@5": 0.70}, {"model_id": "ssd"})

        df = store.compare([p1, p2])
        assert len(df) == 2
        assert "pass@1" in df.columns
        assert df.iloc[0]["model"] in ("base", "ssd")
