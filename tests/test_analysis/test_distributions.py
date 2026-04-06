"""Tests for distribution analysis construction."""

from averyml.analysis.distributions import DistributionAnalyzer


class TestDistributionAnalyzer:
    def test_construction(self):
        analyzer = DistributionAnalyzer("base/model", "ssd/model")
        assert analyzer.base_model_id == "base/model"
        assert analyzer.ssd_model_id == "ssd/model"

    def test_load_prompts_default(self):
        analyzer = DistributionAnalyzer("base/model", "ssd/model")
        prompts = analyzer._load_prompts(None)
        assert len(prompts) > 0
        assert isinstance(prompts[0], str)

    def test_load_prompts_from_file(self, tmp_path):
        import json

        jsonl_path = tmp_path / "prompts.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"prompt_text": "def foo():"}) + "\n")
            f.write(json.dumps({"prompt_text": "def bar():"}) + "\n")

        analyzer = DistributionAnalyzer("base/model", "ssd/model")
        prompts = analyzer._load_prompts(jsonl_path)
        assert len(prompts) == 2
        assert prompts[0] == "def foo():"
