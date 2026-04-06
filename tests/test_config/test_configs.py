"""Tests for the configuration system."""

import tempfile
from pathlib import Path

import pytest

from averyml.config.base import BaseConfig
from averyml.config.evaluation import EvaluationConfig
from averyml.config.experiment import ExperimentConfig
from averyml.config.search import SearchConfig
from averyml.config.synthesis import DecodingConfig, SynthesisConfig
from averyml.config.training import TrainingConfig


class TestDecodingConfig:
    def test_defaults(self):
        cfg = DecodingConfig()
        assert cfg.temperature == 0.6
        assert cfg.top_k == 20
        assert cfg.top_p == 0.95
        assert cfg.min_p == 0.0

    def test_custom_values(self):
        cfg = DecodingConfig(temperature=1.5, top_k=10, top_p=0.8)
        assert cfg.temperature == 1.5
        assert cfg.top_k == 10

    def test_validation_rejects_negative_temperature(self):
        with pytest.raises(Exception):
            DecodingConfig(temperature=-0.1)


class TestSynthesisConfig:
    def test_defaults(self):
        cfg = SynthesisConfig(model_id="test/model")
        assert cfg.n_samples == 1
        assert cfg.backend == "vllm"
        assert cfg.decoding.temperature == 0.6

    def test_yaml_roundtrip(self, tmp_path):
        cfg = SynthesisConfig(
            model_id="test/model",
            n_samples=3,
            decoding=DecodingConfig(temperature=1.5),
        )
        yaml_path = tmp_path / "test.yaml"
        cfg.to_yaml(yaml_path)
        loaded = SynthesisConfig.from_yaml(yaml_path)
        assert loaded.model_id == "test/model"
        assert loaded.n_samples == 3
        assert loaded.decoding.temperature == 1.5

    def test_merge_overrides(self):
        cfg = SynthesisConfig(model_id="test/model")
        merged = cfg.merge({"n_samples": 5, "backend": "hf"})
        assert merged.n_samples == 5
        assert merged.backend == "hf"
        assert merged.model_id == "test/model"


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig(model_id="test/model", dataset_path="./data.jsonl")
        assert cfg.learning_rate == 5e-6
        assert cfg.num_train_iterations == 2500
        assert cfg.warmup_iterations == 250
        assert cfg.bf16 is True
        assert cfg.lr_scheduler_type == "cosine"

    def test_thinking_config(self):
        cfg = TrainingConfig(
            model_id="test/model",
            dataset_path="./data.jsonl",
            num_train_iterations=300,
            warmup_iterations=50,
        )
        assert cfg.num_train_iterations == 300
        assert cfg.warmup_iterations == 50


class TestEvaluationConfig:
    def test_defaults(self):
        cfg = EvaluationConfig(model_id="test/model")
        assert cfg.benchmark == "livecodebench_v6"
        assert cfg.n_repeat == 20
        assert cfg.k_values == [1, 5, 10]

    def test_yaml_roundtrip(self, tmp_path):
        cfg = EvaluationConfig(
            model_id="test/model",
            benchmark="livecodebench_v5",
            decoding=DecodingConfig(temperature=0.9),
        )
        yaml_path = tmp_path / "eval.yaml"
        cfg.to_yaml(yaml_path)
        loaded = EvaluationConfig.from_yaml(yaml_path)
        assert loaded.benchmark == "livecodebench_v5"
        assert loaded.decoding.temperature == 0.9


class TestSearchConfig:
    def test_defaults(self):
        cfg = SearchConfig(base_model_id="test/model")
        assert len(cfg.t_train_values) == 5
        assert len(cfg.t_eval_values) == 7
        assert cfg.n_samples == 1


class TestExperimentConfig:
    def test_full_config(self, tmp_path):
        cfg = ExperimentConfig(
            name="test_experiment",
            synthesis=SynthesisConfig(model_id="test/model"),
            training=TrainingConfig(model_id="test/model", dataset_path="./data.jsonl"),
            evaluation=EvaluationConfig(model_id="test/model"),
        )
        yaml_path = tmp_path / "experiment.yaml"
        cfg.to_yaml(yaml_path)
        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.name == "test_experiment"
        assert loaded.synthesis.model_id == "test/model"
        assert loaded.training.learning_rate == 5e-6
