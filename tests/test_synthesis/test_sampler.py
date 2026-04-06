"""Tests for Sampler construction and config handling."""

import pytest

from averyml.config.synthesis import DecodingConfig, SynthesisConfig
from averyml.synthesis.sampler import Sampler


class TestSamplerConstruction:
    def test_builds_with_valid_config(self):
        cfg = SynthesisConfig(model_id="test/model")
        sampler = Sampler(cfg)
        assert sampler.config.model_id == "test/model"
        assert sampler.config.n_samples == 1
        assert sampler.config.backend == "vllm"

    def test_prompt_source_registry_lookup(self):
        cfg = SynthesisConfig(model_id="test/model", prompt_source="rstarcoder")
        sampler = Sampler(cfg)
        source = sampler._build_prompt_source()
        assert source is not None

    def test_custom_prompt_source_lookup(self):
        cfg = SynthesisConfig(model_id="test/model", prompt_source="custom", prompt_dataset="/tmp/test.jsonl")
        sampler = Sampler(cfg)
        source = sampler._build_prompt_source()
        assert source is not None

    def test_backend_registry_lookup(self):
        cfg = SynthesisConfig(model_id="test/model", backend="hf")
        sampler = Sampler(cfg)
        backend = sampler._build_backend()
        assert backend is not None

    def test_unknown_prompt_source_raises(self):
        cfg = SynthesisConfig(model_id="test/model", prompt_source="nonexistent")
        sampler = Sampler(cfg)
        with pytest.raises(KeyError):
            sampler._build_prompt_source()

    def test_unknown_backend_raises(self):
        cfg = SynthesisConfig(model_id="test/model", backend="nonexistent")
        sampler = Sampler(cfg)
        with pytest.raises(KeyError):
            sampler._build_backend()

    def test_decoding_config_passed_through(self):
        decoding = DecodingConfig(temperature=2.0, top_k=5)
        cfg = SynthesisConfig(model_id="test/model", decoding=decoding)
        assert cfg.decoding.temperature == 2.0
        assert cfg.decoding.top_k == 5
