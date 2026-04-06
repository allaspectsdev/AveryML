"""Tests for training backend registration."""

from averyml.utils.registry import training_backend_registry


class TestTrainingBackendRegistry:
    def test_hf_trainer_registered(self):
        cls = training_backend_registry.get("hf_trainer")
        assert cls is not None
        assert cls.__name__ == "HFTrainerBackend"

    def test_torchtune_registered(self):
        cls = training_backend_registry.get("torchtune")
        assert cls is not None
        assert cls.__name__ == "TorchtuneTrainerBackend"

    def test_list_names(self):
        names = training_backend_registry.list_names()
        assert "hf_trainer" in names
        assert "torchtune" in names

    def test_torchtune_raises_not_implemented(self):
        import pytest

        cls = training_backend_registry.get("torchtune")
        backend = cls()
        with pytest.raises(NotImplementedError, match="torchtune backend is not yet implemented"):
            backend.train(config=None, dataset=None)
