"""Tests for synthesis backend registration and construction."""

from averyml.utils.registry import synthesis_backend_registry


class TestSynthesisBackendRegistry:
    def test_vllm_registered(self):
        cls = synthesis_backend_registry.get("vllm")
        assert cls is not None
        assert cls.__name__ == "VLLMSynthesisBackend"

    def test_hf_registered(self):
        cls = synthesis_backend_registry.get("hf")
        assert cls is not None
        assert cls.__name__ == "HFSynthesisBackend"

    def test_list_names(self):
        names = synthesis_backend_registry.list_names()
        assert "vllm" in names
        assert "hf" in names

    def test_unknown_raises(self):
        import pytest
        with pytest.raises(KeyError, match="Unknown synthesis backend"):
            synthesis_backend_registry.get("nonexistent")

    def test_hf_backend_construction(self):
        cls = synthesis_backend_registry.get("hf")
        backend = cls()
        assert hasattr(backend, "generate")
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "cleanup")

    def test_vllm_backend_construction(self):
        cls = synthesis_backend_registry.get("vllm")
        backend = cls(tensor_parallel_size=1)
        assert hasattr(backend, "generate")
        assert hasattr(backend, "load_model")
