"""Tests for Evaluator construction and benchmark wiring."""

from averyml.config.evaluation import EvaluationConfig
from averyml.config.synthesis import DecodingConfig
from averyml.evaluation.evaluator import Evaluator


class TestEvaluatorConstruction:
    def test_builds_with_config(self):
        cfg = EvaluationConfig(model_id="test/model")
        evaluator = Evaluator(cfg)
        assert evaluator.config.model_id == "test/model"
        assert evaluator.config.benchmark == "livecodebench_v6"

    def test_build_benchmark(self):
        cfg = EvaluationConfig(model_id="test/model", benchmark="livecodebench_v6")
        evaluator = Evaluator(cfg)
        benchmark = evaluator._build_benchmark()
        assert benchmark is not None
        assert benchmark.version == "livecodebench_v6"

    def test_build_benchmark_v5(self):
        cfg = EvaluationConfig(model_id="test/model", benchmark="livecodebench_v5")
        evaluator = Evaluator(cfg)
        benchmark = evaluator._build_benchmark()
        assert benchmark.version == "livecodebench_v5"

    def test_decoding_config_applied(self):
        decoding = DecodingConfig(temperature=0.9, top_k=10)
        cfg = EvaluationConfig(model_id="test/model", decoding=decoding)
        assert cfg.decoding.temperature == 0.9
        assert cfg.decoding.top_k == 10
