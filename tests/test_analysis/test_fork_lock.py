"""Tests for fork/lock detector construction."""

from averyml.analysis.fork_lock import ForkLockDetector


class TestForkLockDetector:
    def test_construction(self):
        detector = ForkLockDetector("base/model", "ssd/model")
        assert detector.base_model_id == "base/model"
        assert detector.ssd_model_id == "ssd/model"

    def test_load_prompts_default(self):
        detector = ForkLockDetector("base/model", "ssd/model")
        prompts = detector._load_prompts(None)
        assert len(prompts) > 0

    def test_load_prompts_from_file(self, tmp_path):
        import json

        jsonl_path = tmp_path / "prompts.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"prompt_text": "test prompt"}) + "\n")

        detector = ForkLockDetector("base/model", "ssd/model")
        prompts = detector._load_prompts(jsonl_path)
        assert len(prompts) == 1
