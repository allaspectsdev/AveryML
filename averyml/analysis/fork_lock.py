"""Fork/lock position detection and analysis.

Fork: position where the distribution is spread across multiple plausible
      tokens that lead to meaningfully different continuations.
Lock: position where the distribution is sharply peaked with a long
      distractor tail.

The paper shows SSD reshapes these differently: it suppresses distractor
tails at locks while preserving useful diversity at forks (Section 4.1-4.2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ForkLockDetector:
    """Detect and analyze fork/lock positions in token sequences."""

    def __init__(self, base_model_id: str, ssd_model_id: str):
        self.base_model_id = base_model_id
        self.ssd_model_id = ssd_model_id

    def run(self, prompts_path: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
        """Run fork/lock detection on both models."""
        prompts = self._load_prompts(prompts_path)
        results = {}

        for name, model_id in [("base", self.base_model_id), ("ssd", self.ssd_model_id)]:
            profiles = []
            for prompt in prompts:
                profile = self.compute_profile(model_id, prompt)
                profiles.append(profile)
            results[name] = profiles

        # Compare
        results["comparison"] = self._compare_profiles(results["base"], results["ssd"])

        if output_dir:
            from averyml.utils.io import write_json
            write_json(results, output_dir / "fork_lock_analysis.json")

        return results

    def compute_profile(self, model_id: str, prompt: str) -> dict[str, Any]:
        """Compute per-position fork/lock profile for a prompt."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].cpu()

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Per-position metrics
        entropy = -(probs * log_probs).sum(dim=-1).tolist()
        top1_prob = probs.max(dim=-1).values.tolist()

        # Classify positions
        fork_positions = [i for i, e in enumerate(entropy) if e > 2.0]
        lock_positions = [i for i, p in enumerate(top1_prob) if p > 0.9]

        return {
            "entropy": entropy,
            "top1_prob": top1_prob,
            "fork_positions": fork_positions,
            "lock_positions": lock_positions,
            "num_forks": len(fork_positions),
            "num_locks": len(lock_positions),
            "mean_entropy": float(np.mean(entropy)),
            "mean_top1_prob": float(np.mean(top1_prob)),
        }

    def _compare_profiles(self, base_profiles: list[dict], ssd_profiles: list[dict]) -> dict:
        """Compare fork/lock statistics between base and SSD models."""
        base_fork_entropy = []
        ssd_fork_entropy = []
        base_lock_top1 = []
        ssd_lock_top1 = []

        for bp, sp in zip(base_profiles, ssd_profiles):
            for pos in bp["fork_positions"]:
                if pos < len(bp["entropy"]) and pos < len(sp["entropy"]):
                    base_fork_entropy.append(bp["entropy"][pos])
                    ssd_fork_entropy.append(sp["entropy"][pos])
            for pos in bp["lock_positions"]:
                if pos < len(bp["top1_prob"]) and pos < len(sp["top1_prob"]):
                    base_lock_top1.append(bp["top1_prob"][pos])
                    ssd_lock_top1.append(sp["top1_prob"][pos])

        return {
            "fork_entropy_base_mean": float(np.mean(base_fork_entropy)) if base_fork_entropy else 0,
            "fork_entropy_ssd_mean": float(np.mean(ssd_fork_entropy)) if ssd_fork_entropy else 0,
            "lock_top1_base_mean": float(np.mean(base_lock_top1)) if base_lock_top1 else 0,
            "lock_top1_ssd_mean": float(np.mean(ssd_lock_top1)) if ssd_lock_top1 else 0,
        }

    def _load_prompts(self, prompts_path: Path | None) -> list[str]:
        if prompts_path:
            from averyml.utils.io import read_jsonl
            return [item["prompt_text"] for item in read_jsonl(prompts_path)]
        return ["def solve(arr):\n    # Sort the array\n    "]
