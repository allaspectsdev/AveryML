<p align="center">
  <h1 align="center">AveryML</h1>
  <p align="center">
    <strong>Make your LLM better at code by feeding it its own homework — no answers required.</strong>
  </p>
  <p align="center">
    <em>Inspired by Oleg</em>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2604.01193"><img src="https://img.shields.io/badge/paper-arXiv%3A2604.01193-b31b1b.svg" alt="Paper"></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/tests-47%20passed-brightgreen.svg" alt="Tests"></a>
  </p>
</p>

---

## What is this?

You know how sometimes the best way to learn is to just *do the thing*, even if you get it wrong? Turns out LLMs work the same way.

**AveryML** is a complete implementation of [Simple Self-Distillation (SSD)](https://arxiv.org/abs/2604.01193) — a technique from Apple Research that improves an LLM's code generation by training it on its own raw, unverified outputs. No teacher model. No reward model. No verifier. No RL. Just vibes and cross-entropy.

The result? **+12.9 percentage points** on pass@1 for Qwen3-30B-Instruct on LiveCodeBench v6 (42.4% -> 55.3%), with the biggest gains on the hardest problems.

### The trick in 30 seconds

```
1. SAMPLE  — Ask your model to solve ~10K coding problems at a high temperature
2. TRAIN   — Fine-tune the same model on those solutions (yes, including the wrong ones)
3. DECODE  — Deploy with a tuned evaluation temperature
```

That's it. The magic is in *how* this reshapes the model's token distributions: it sharpens "lock" positions (where precision matters) while preserving diversity at "fork" positions (where exploration matters). A single global temperature can't do both — but SSD bakes context-dependent sharpening into the weights.

---

## Why AveryML over the reference implementation?

Apple's [ml-ssd](https://github.com/apple/ml-ssd) repo is evaluation-only. AveryML is the full pipeline:

| | ml-ssd (Apple) | AveryML |
|---|:---:|:---:|
| Data synthesis | - | vLLM + HuggingFace backends |
| Training | - | HF Trainer SFT with long-context support |
| Evaluation | LCB v6 only | LCB v5 + v6, per-difficulty breakdowns |
| Temperature grid search | - | Full (T_train x T_eval) sweep with T_eff analysis |
| Distribution analysis | - | Fork/lock detection, Eq.4 decomposition |
| Config system | None | Pydantic + YAML with CLI overrides |
| CLI | Single script | 7 subcommands |
| Experiment tracking | - | W&B integration |

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/your-username/AveryML.git
cd AveryML
pip install -e ".[all]"

# See what you can do
averyml --help
```

### Run the full pipeline

```bash
averyml run-pipeline --config configs/experiments/full_pipeline_qwen3_4b.yaml
```

### Or step by step

```bash
# Step 1: Generate training data (sample from frozen model at T_train=1.5)
averyml synthesize --config configs/synthesis/default.yaml

# Step 2: Fine-tune (standard SFT, ~2500 steps)
averyml train --config configs/training/sft_instruct.yaml

# Step 3: Evaluate on LiveCodeBench v6
averyml evaluate --config configs/evaluation/lcb_v6.yaml
```

### Temperature grid search (the fun part)

The paper's key insight: performance is governed by **T_eff = T_train x T_eval**, with a sweet spot around ~1.2. Run the grid search to find your model's optimal band:

```bash
averyml search --config configs/search/temperature_grid.yaml --diagonal-only
```

---

## Project Structure

```
averyml/
  config/         Pydantic configs with YAML I/O
  synthesis/      Step 1 — Sample solutions from frozen base model
    backends/       vLLM (fast) or HuggingFace Transformers (accessible)
    prompts/        rSTARcoder seed dataset or custom JSONL
    filters.py      Minimal filtering only (empty + stub) — NO correctness checks
  training/       Step 2 — Standard SFT on raw outputs
    backends/       HuggingFace Trainer or torchtune
    data.py         Chat-templated tokenization with prompt masking
  evaluation/     Step 3 — Benchmark on LiveCodeBench
    benchmarks/     LCB v5 + v6, sandboxed code execution
    metrics.py      Unbiased pass@k estimator with per-difficulty breakdowns
  search/         Grid search over (T_train, T_eval) space
  analysis/       Token distribution, fork/lock, compression analysis
  cli.py          Typer CLI with 7 subcommands
configs/          Ready-to-use YAML configs for all pipeline stages
tests/            47 tests covering configs, filters, metrics, grid search
scripts/          Shell scripts for common workflows
```

---

## Key Concepts

### The Precision-Exploration Conflict

Code generation mixes two kinds of token positions:

- **Locks** — syntax/semantics leave almost no ambiguity (`if n ==`). Precision matters; the distractor tail is noise.
- **Forks** — multiple viable algorithms could follow (`def solve(arr):\n    `). Exploration matters; you *want* diversity.

A single global temperature can't serve both. Lower it and you lose exploration at forks. Raise it and distractors flood back in at locks.

**SSD resolves this by baking context-dependent reshaping into the weights** through support compression (trimming the tail) and within-support reshaping (redistributing mass among survivors). The result: locks get sharper, forks stay broad, and higher evaluation temperatures become newly effective.

### Minimal Filtering is a Feature

SSD uses **no correctness signal whatsoever**. The synthesized data isn't filtered by execution, test cases, or any quality metric. Even when ~62% of samples are gibberish (T_train=2.0 without truncation), the fine-tuned model still improves by +5.7pp. The signal comes from how high-temperature sampling reshapes token probabilities, not from training on correct code.

---

## Configuration

Every pipeline step is driven by a Pydantic config with YAML serialization. CLI flags override YAML values:

```bash
# Load from YAML, override temperature
averyml synthesize \
    --config configs/synthesis/default.yaml \
    --temperature 2.0 \
    --top-k 5
```

### Key hyperparameters (from the paper)

| Parameter | Default | Paper range | Notes |
|---|---|---|---|
| T_train | 1.5 | 0.5 - 2.0 | Higher = more diverse samples |
| T_eval | 0.6 | 0.4 - 1.5 | Tuned independently |
| top-k (train) | 10 | 5 - 20 | Truncation during synthesis |
| N (samples/prompt) | 1 | 1 | One sample per prompt suffices |
| Training steps | 2500 | 300 - 2500 | 2500 for instruct, 300 for thinking |
| Learning rate | 5e-6 | - | AdamW with cosine decay |

---

## Supported Models

SSD generalizes across model families, scales, and reasoning styles:

| Model | Type | Notes |
|---|---|---|
| Qwen3-4B-Instruct | Dense, instruct | Good for quick experiments |
| Qwen3-4B-Thinking | Dense, thinking | Use `sft_thinking.yaml` (300 steps) |
| Qwen3-30B-Instruct | MoE, instruct | Best absolute results in paper |
| Qwen3-30B-Thinking | MoE, thinking | Largest gains on hard problems |
| Llama-3.1-8B-Instruct | Dense, instruct | Cross-family generalization |

---

## Analysis Tools

Dig into *why* SSD works with built-in analysis:

```bash
# Compare token distributions between base and SSD models
averyml analyze \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --ssd-model ./checkpoints/sft_instruct/final_checkpoint \
    --analysis-type all
```

This produces:
- **Distribution comparison** — cumulative mass curves, entropy, top-k overlap (Figure 6)
- **Fork/lock detection** — per-position entropy and top-1 probability profiles (Figure 4)
- **Compression decomposition** — the three terms from Equation 4

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check averyml/

# Type check
mypy averyml/
```

---

## Citation

If you use AveryML in your research, please cite the original paper:

```bibtex
@misc{zhang2026ssd,
    title={Embarrassingly Simple Self-Distillation Improves Code Generation},
    author={Ruixiang Zhang and Richard He Bai and Huangjie Zheng and
            Navdeep Jaitly and Ronan Collobert and Yizhe Zhang},
    year={2026},
    eprint={2604.01193},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
}
```

---

## License

MIT
