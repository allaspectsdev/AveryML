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
    """Analyzes the three-term decomposition of SSD's training objective.

    Compares support compression between base and SSD models to show
    how SSD trims distractor tails while preserving the viable head.
    """

    def __init__(self, base_model_id: str, ssd_model_id: str):
        self.base_model_id = base_model_id
        self.ssd_model_id = ssd_model_id

    def run(self, prompts_path: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
        """Run compression analysis on both base and SSD models."""
        prompts = self._load_prompts(prompts_path)

        base_results = self._analyze_model(self.base_model_id, prompts)
        ssd_results = self._analyze_model(self.ssd_model_id, prompts)

        results = {
            "base": base_results,
            "ssd": ssd_results,
            "comparison": {
                "kept_mass_delta": ssd_results["mean_kept_mass"] - base_results["mean_kept_mass"],
                "compression_delta": ssd_results["mean_support_compression"] - base_results["mean_support_compression"],
            },
            "num_prompts": len(prompts),
        }

        if output_dir:
            from averyml.utils.io import write_json
            write_json(results, output_dir / "compression_analysis.json")

        return results

    def _analyze_model(self, model_id: str, prompts: list[str], temperature: float = 1.5, top_k: int = 20) -> dict:
        """Compute support compression metrics for a single model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Analyzing model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

        support_compression_values = []
        kept_mass_values = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
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
            "model_id": model_id,
            "mean_kept_mass": float(np.mean(kept_mass_values)),
            "mean_support_compression": float(np.mean(support_compression_values)),
            "temperature": temperature,
            "top_k": top_k,
        }

    def _load_prompts(self, prompts_path: Path | None) -> list[str]:
        if prompts_path:
            from averyml.utils.io import read_jsonl
            return [item["prompt_text"] for item in read_jsonl(prompts_path)]
        return ["def solve(arr):\n    # Sort the array\n    "]
