"""Token distribution analysis: compare base vs SSD model distributions.

Implements the analysis from Section 4 of the paper, showing that SSD
compresses distractor tails and makes T_eval more effective near the head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DistributionAnalyzer:
    """Compare token distributions between base and SSD models."""

    def __init__(self, base_model_id: str, ssd_model_id: str):
        self.base_model_id = base_model_id
        self.ssd_model_id = ssd_model_id

    def run(self, prompts_path: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
        """Run full distribution analysis."""
        prompts = self._load_prompts(prompts_path)
        results = {}

        base_logits = self._get_logits(self.base_model_id, prompts)
        ssd_logits = self._get_logits(self.ssd_model_id, prompts)

        results["cumulative_mass"] = self._compare_cumulative_mass(base_logits, ssd_logits)
        results["entropy"] = self._compare_entropy(base_logits, ssd_logits)
        results["top_k_overlap"] = self._compare_top_k_overlap(base_logits, ssd_logits)

        if output_dir:
            from averyml.utils.io import write_json
            write_json(results, output_dir / "distribution_analysis.json")

        return results

    def _load_prompts(self, prompts_path: Path | None) -> list[str]:
        if prompts_path:
            from averyml.utils.io import read_jsonl
            return [item["prompt_text"] for item in read_jsonl(prompts_path)]
        return ["def solve(arr):\n    # Sort the array\n    "]

    def _get_logits(self, model_id: str, prompts: list[str]) -> list[torch.Tensor]:
        """Get token logits from a model for given prompts."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

        all_logits = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            all_logits.append(outputs.logits[0].cpu())

        return all_logits

    def _compare_cumulative_mass(
        self, base_logits: list[torch.Tensor], ssd_logits: list[torch.Tensor]
    ) -> dict:
        """Compare how quickly cumulative probability mass rises (Figure 6a)."""
        results = {"base": [], "ssd": []}
        for base_l, ssd_l in zip(base_logits, ssd_logits):
            for name, logits in [("base", base_l), ("ssd", ssd_l)]:
                # Average over positions
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1).mean(dim=0)
                results[name].append(cumulative[:100].tolist())
        return results

    def _compare_entropy(
        self, base_logits: list[torch.Tensor], ssd_logits: list[torch.Tensor]
    ) -> dict:
        """Compare entropy distributions."""
        def compute_entropy(logits: torch.Tensor) -> float:
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            return entropy.mean().item()

        return {
            "base_mean_entropy": np.mean([compute_entropy(l) for l in base_logits]),
            "ssd_mean_entropy": np.mean([compute_entropy(l) for l in ssd_logits]),
        }

    def _compare_top_k_overlap(
        self, base_logits: list[torch.Tensor], ssd_logits: list[torch.Tensor], k: int = 20
    ) -> dict:
        """Measure how much the top-k support changes after SSD."""
        overlaps = []
        for base_l, ssd_l in zip(base_logits, ssd_logits):
            min_len = min(len(base_l), len(ssd_l))
            base_topk = torch.topk(base_l[:min_len], k, dim=-1).indices
            ssd_topk = torch.topk(ssd_l[:min_len], k, dim=-1).indices
            # Per-position overlap
            for pos in range(min_len):
                base_set = set(base_topk[pos].tolist())
                ssd_set = set(ssd_topk[pos].tolist())
                overlaps.append(len(base_set & ssd_set) / k)

        return {"mean_top_k_overlap": float(np.mean(overlaps)), "k": k}
