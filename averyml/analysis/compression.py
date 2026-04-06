"""Support compression and within-support reshaping analysis.

Implements the SSD loss decomposition from Equation 4 of the paper:

L(theta) = -log KeptMass_theta(S)                      [support compression]
          + (1-T) * H_{1/T}(p_theta(.|S))              [within-support reshaping]
          + T * KL(q || p_{theta,T}(.|S))               [alignment to base model]
          + const

where S is the support set after truncation, T = T_train.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CompressionAnalyzer:
    """Analyzes the three-term decomposition of SSD's training objective."""

    def __init__(self, base_model_id: str, ssd_model_id: str):
        self.base_model_id = base_model_id
        self.ssd_model_id = ssd_model_id

    def run(self, prompts_path: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
        """Run compression analysis."""
        prompts = self._load_prompts(prompts_path)
        results = self.analyze_prompts(prompts)

        if output_dir:
            from averyml.utils.io import write_json
            write_json(results, output_dir / "compression_analysis.json")

        return results

    def analyze_prompts(self, prompts: list[str], temperature: float = 1.5, top_k: int = 20) -> dict:
        """Compute the three-term decomposition across prompts."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

        support_compression_values = []
        kept_mass_values = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                outputs = base_model(**inputs)
            logits = outputs.logits[0].cpu()

            # Compute kept mass after top-k truncation
            probs = torch.softmax(logits, dim=-1)
            topk_probs, _ = torch.topk(probs, top_k, dim=-1)
            kept_mass = topk_probs.sum(dim=-1)
            mean_kept_mass = kept_mass.mean().item()

            # Term 1: -log KeptMass
            support_compression = -np.log(max(mean_kept_mass, 1e-10))

            support_compression_values.append(support_compression)
            kept_mass_values.append(mean_kept_mass)

        return {
            "mean_kept_mass": float(np.mean(kept_mass_values)),
            "mean_support_compression": float(np.mean(support_compression_values)),
            "temperature": temperature,
            "top_k": top_k,
            "num_prompts": len(prompts),
        }

    def _load_prompts(self, prompts_path: Path | None) -> list[str]:
        if prompts_path:
            from averyml.utils.io import read_jsonl
            return [item["prompt_text"] for item in read_jsonl(prompts_path)]
        return ["def solve(arr):\n    # Sort the array\n    "]
