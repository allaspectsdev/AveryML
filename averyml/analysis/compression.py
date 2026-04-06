"""Support compression and within-support reshaping analysis.

Implements the full SSD loss decomposition from Equation 4 of the paper:

L(theta) = -log KeptMass_theta(S)                      [Term 1: support compression]
          + (1-T) * H_{1/T}(p_theta(.|S))              [Term 2: within-support reshaping]
          + T * KL(q || p_{theta,T}(.|S))               [Term 3: alignment to base model]
          + const

where S is the support set after truncation, T = T_train,
q is the renormalized distribution over S, and H_{1/T} is the Renyi entropy.
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

    Compares all three terms between base and SSD models to show
    how SSD trims distractor tails (Term 1), reshapes the surviving
    head (Term 2), and stays aligned with the base distribution (Term 3).
    """

    def __init__(self, base_model_id: str, ssd_model_id: str):
        self.base_model_id = base_model_id
        self.ssd_model_id = ssd_model_id

    def run(self, prompts_path: Path | None = None, output_dir: Path | None = None,
            temperature: float = 1.5, top_k: int = 20) -> dict[str, Any]:
        """Run full Eq.4 decomposition on both models."""
        prompts = self._load_prompts(prompts_path)

        base_results = self._analyze_model(self.base_model_id, prompts, temperature, top_k)
        ssd_results = self._analyze_model(self.ssd_model_id, prompts, temperature, top_k)

        # Cross-model Term 3: KL between SSD and base restricted to support
        alignment = self._compute_alignment(prompts, temperature, top_k)

        results = {
            "base": base_results,
            "ssd": ssd_results,
            "alignment": alignment,
            "comparison": {
                "kept_mass_delta": ssd_results["mean_kept_mass"] - base_results["mean_kept_mass"],
                "compression_delta": ssd_results["term1_mean"] - base_results["term1_mean"],
                "reshaping_delta": ssd_results["term2_mean"] - base_results["term2_mean"],
            },
            "temperature": temperature,
            "top_k": top_k,
            "num_prompts": len(prompts),
        }

        if output_dir:
            from averyml.utils.io import write_json
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(results, output_dir / "compression_analysis.json")

        return results

    def _analyze_model(self, model_id: str, prompts: list[str],
                       temperature: float, top_k: int) -> dict:
        """Compute Terms 1 and 2 for a single model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Analyzing model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

        term1_values = []  # support compression
        term2_values = []  # within-support reshaping
        kept_mass_values = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0].float().cpu()  # [seq_len, vocab]

            # Per-position analysis, averaged over sequence
            probs = torch.softmax(logits, dim=-1)

            # Support set S: top-k tokens at each position
            topk_vals, topk_ids = torch.topk(probs, top_k, dim=-1)

            # Term 1: -log KeptMass(S)
            kept_mass = topk_vals.sum(dim=-1)  # [seq_len]
            term1 = -torch.log(kept_mass.clamp(min=1e-10))
            term1_values.append(term1.mean().item())
            kept_mass_values.append(kept_mass.mean().item())

            # Term 2: (1-T) * H_{1/T}(p_theta(.|S))
            # Renyi entropy of order 1/T on the restricted+renormalized distribution
            p_restricted = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalized to S
            alpha = 1.0 / temperature
            if abs(alpha - 1.0) < 1e-6:
                # Shannon entropy (limit as alpha -> 1)
                renyi = -(p_restricted * torch.log(p_restricted.clamp(min=1e-10))).sum(dim=-1)
            else:
                renyi = torch.log((p_restricted ** alpha).sum(dim=-1).clamp(min=1e-10)) / (1.0 - alpha)
            term2 = (1.0 - temperature) * renyi
            term2_values.append(term2.mean().item())

        return {
            "model_id": model_id,
            "mean_kept_mass": float(np.mean(kept_mass_values)),
            "term1_mean": float(np.mean(term1_values)),
            "term2_mean": float(np.mean(term2_values)),
            "term1_values": term1_values,
            "term2_values": term2_values,
        }

    def _compute_alignment(self, prompts: list[str], temperature: float, top_k: int) -> dict:
        """Compute Term 3: T * KL(q || p_{theta,T}(.|S)) between SSD and base."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Computing alignment term (KL between SSD and base)...")

        base_tok = AutoTokenizer.from_pretrained(self.base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        ssd_tok = AutoTokenizer.from_pretrained(self.ssd_model_id)
        ssd_model = AutoModelForCausalLM.from_pretrained(
            self.ssd_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

        term3_values = []

        for prompt in prompts:
            # Get base logits
            base_inputs = base_tok(prompt, return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                base_out = base_model(**base_inputs)
            base_logits = base_out.logits[0].float().cpu()

            # Get SSD logits
            ssd_inputs = ssd_tok(prompt, return_tensors="pt").to(ssd_model.device)
            with torch.no_grad():
                ssd_out = ssd_model(**ssd_inputs)
            ssd_logits = ssd_out.logits[0].float().cpu()

            min_len = min(len(base_logits), len(ssd_logits))
            base_logits = base_logits[:min_len]
            ssd_logits = ssd_logits[:min_len]

            # Temperature-scaled base distribution
            base_probs_T = torch.softmax(base_logits / temperature, dim=-1)

            # Support set from base (top-k)
            _, topk_ids = torch.topk(torch.softmax(base_logits, dim=-1), top_k, dim=-1)

            # Gather probabilities on support
            base_on_S = torch.gather(base_probs_T, 1, topk_ids)
            base_on_S = base_on_S / base_on_S.sum(dim=-1, keepdim=True)

            ssd_probs = torch.softmax(ssd_logits, dim=-1)
            ssd_on_S = torch.gather(ssd_probs, 1, topk_ids)
            ssd_on_S = ssd_on_S / ssd_on_S.sum(dim=-1, keepdim=True).clamp(min=1e-10)

            # KL(q || p_T) where q=SSD, p_T=base tempered
            kl = (ssd_on_S * torch.log((ssd_on_S / base_on_S.clamp(min=1e-10)).clamp(min=1e-10))).sum(dim=-1)
            term3 = temperature * kl
            term3_values.append(term3.mean().item())

        return {
            "term3_mean": float(np.mean(term3_values)),
            "term3_values": term3_values,
        }

    def _load_prompts(self, prompts_path: Path | None) -> list[str]:
        if prompts_path:
            from averyml.utils.io import read_jsonl
            return [item["prompt_text"] for item in read_jsonl(prompts_path)]
        return ["def solve(arr):\n    # Sort the array\n    "]
