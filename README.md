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
    <a href="#"><img src="https://img.shields.io/badge/tests-112%20passed-brightgreen.svg" alt="Tests"></a>
    <a href="#docker"><img src="https://img.shields.io/badge/docker-ready-2496ED.svg" alt="Docker"></a>
  </p>
</p>

---

## TL;DR

```bash
pip install -e ".[all]"
averyml run-pipeline --config configs/experiments/full_pipeline_qwen3_4b.yaml
```

Your model samples its own solutions, trains on them (wrong answers included), and comes out the other side writing **substantially better code**. No reward model. No verifier. No RL. One command.

---

## What is this?

You know how sometimes the best way to learn is to just *do the thing*, even if you get it wrong? Turns out LLMs work the same way.

**AveryML** is a production-grade implementation of [Simple Self-Distillation (SSD)](https://arxiv.org/abs/2604.01193) — a technique from Apple Research that improves an LLM's code generation by training it on its own raw, unverified outputs. No teacher model. No reward model. No verifier. No RL. Just vibes and cross-entropy.

The result? **+12.9 percentage points** on pass@1 for Qwen3-30B-Instruct on LiveCodeBench v6 (42.4% -> 55.3%), with the biggest gains on the hardest problems.

### The trick in 30 seconds

```
1. SAMPLE  — Ask your model to solve ~10K coding problems at a high temperature
2. TRAIN   — Fine-tune the same model on those solutions (yes, including the wrong ones)
3. DECODE  — Deploy with a tuned evaluation temperature
```

That's it. The magic is in *how* this reshapes the model's token distributions: it sharpens "lock" positions (where precision matters) while preserving diversity at "fork" positions (where exploration matters). A single global temperature can't do both — but SSD bakes context-dependent sharpening into the weights.

> **"Even when ~62% of synthesized samples are gibberish, the fine-tuned model still improves by +5.7pp."**
>
> The signal comes from how high-temperature sampling reshapes token probabilities, not from training on correct code. Yes, really.

---

## Why AveryML?

Apple's [ml-ssd](https://github.com/apple/ml-ssd) repo is evaluation-only — it benchmarks models but can't train them. AveryML is the complete, battle-tested pipeline:

| | ml-ssd (Apple) | AveryML |
|---|:---:|:---:|
| Data synthesis | - | vLLM + HuggingFace backends, progress bars, filtering stats |
| Training | - | HF Trainer SFT, flash attention fallback, checkpoint resume |
| Evaluation | LCB v6 only | LCB v5 + v6, per-difficulty breakdowns, code extraction fallback |
| Temperature grid search | - | Full (T_train x T_eval) sweep with T_eff analysis |
| Distribution analysis | - | Fork/lock detection, Eq.4 decomposition |
| Config system | None | Pydantic + YAML with live validation |
| CLI | Single script | 8 commands with clear error messages |
| Web dashboard | - | 6-tab Gradio UI with interactive Plotly charts |
| Docker | - | GPU-ready with docker compose |
| Experiment tracking | - | W&B integration |

---

## Quick Start

```bash
git clone https://github.com/allaspectsdev/AveryML.git
cd AveryML
pip install -e ".[all]"

averyml --help
```

### One command to rule them all

```bash
averyml run-pipeline --config configs/experiments/full_pipeline_qwen3_4b.yaml
```

This runs the entire SSD pipeline end-to-end: synthesize data, fine-tune, evaluate. Go get coffee.

### Or step by step (for the control freaks)

```bash
# Step 1: Generate training data — sample from frozen model at T_train=1.5
averyml synthesize --config configs/synthesis/default.yaml

# Step 2: Fine-tune — standard SFT, ~2500 steps, auto-resumes if interrupted
averyml train --config configs/training/sft_instruct.yaml

# Step 3: Evaluate — benchmark on LiveCodeBench v6 with pass@k metrics
averyml evaluate --config configs/evaluation/lcb_v6.yaml
```

### Temperature grid search (where it gets interesting)

The paper's key insight: performance is governed by **T_eff = T_train x T_eval**, with a quadratic peak near ~1.2. Find your model's sweet spot:

```bash
averyml search --config configs/search/temperature_grid.yaml --diagonal-only
```

This sweeps a grid of temperature combinations, trains a model for each, evaluates all of them, and gives you an interactive heatmap of which combinations work best. The `--diagonal-only` flag restricts to the productive T_eff band so you don't waste GPU hours on the corners.

---

## Web Dashboard

Why squint at JSON files when you can have interactive charts?

```bash
pip install averyml[dashboard]
averyml dashboard
```

Opens a 6-tab Gradio dashboard at `localhost:7860`:

| Tab | What you get |
|---|---|
| **Home** | Project status at a glance — recent results, running jobs, config inventory |
| **Pipeline** | Launch any step from the browser — config dropdowns, YAML preview, override sliders, live log streaming |
| **Results** | Compare runs side-by-side with interactive Plotly bar charts and per-difficulty (easy/medium/hard) breakdowns |
| **Temperature Search** | Interactive heatmap of the (T_train, T_eval) grid + T_eff scatter with quadratic fit and R^2 |
| **Data Explorer** | Browse synthesis datasets — pagination, response length stats, sample preview |
| **Config Editor** | Edit YAML in-browser with live Pydantic validation, auto-generated field docs, and save-with-backup |

The dashboard runs without GPU — it's a lightweight viewer that launches pipeline steps as subprocesses when needed.

---

## Built to Not Break

Real experiments crash. AveryML is built for that:

- **Flash attention fallback** — tries `flash_attention_2` first, falls back gracefully with a warning if the model doesn't support it. No more mysterious mid-training crashes.
- **Checkpoint resume** — set `resume_from_checkpoint: true` in your training config. If training dies at step 2400/2500, it picks up where it left off instead of starting over.
- **Dataset validation** — checks field names, empty datasets, tokenizer compatibility, and warns if >20% of samples get filtered. Catches data format mismatches early.
- **GPU count validation** — if your config says `tensor_parallel_size: 4` but you only have 1 GPU, you get a clear error before downloading a 30B model.
- **Code extraction fallback** — if the model outputs raw Python without markdown fences, AveryML tries it as code instead of scoring it as zero. Logs a count of extraction failures so you know what's happening.
- **Progress bars** — tqdm on synthesis rounds and evaluation repeats with per-round timing stats. No more "is it frozen or just thinking?"

---

## Project Structure

```
averyml/
  config/           Pydantic configs with YAML I/O and validation
  synthesis/        Step 1 — Sample solutions from frozen base model
    backends/         vLLM (fast) or HuggingFace Transformers (accessible)
    prompts/          rSTARcoder seed dataset or custom JSONL
    filters.py        Minimal filtering only (empty + stub) — NO correctness checks
  training/         Step 2 — Standard SFT on raw outputs
    backends/         HuggingFace Trainer (with flash attn fallback) or torchtune
    data.py           Chat-templated tokenization with prompt masking + validation
  evaluation/       Step 3 — Benchmark on LiveCodeBench
    benchmarks/       LCB v5 + v6, sandboxed code execution, code extraction fallback
    metrics.py        Unbiased pass@k estimator with per-difficulty breakdowns
  search/           Grid search over (T_train, T_eval) space with resume
  analysis/         Token distribution, fork/lock, compression analysis
  dashboard.py      6-tab Gradio web UI
  cli.py            Typer CLI — 8 commands
configs/            Ready-to-use YAML configs for all pipeline stages
tests/              101 tests covering configs, registries, filters, metrics, and more
scripts/            Shell scripts for common workflows
Dockerfile          GPU-ready container
docker-compose.yml  One-command deployment
```

---

## Key Concepts

### The Precision-Exploration Conflict

Code generation mixes two kinds of token positions:

- **Locks** — syntax/semantics leave almost no ambiguity (`if n ==`). The model knows the answer; the distractor tail is just noise. Precision matters.
- **Forks** — multiple viable algorithms could follow (`def solve(arr):\n    `). Merge sort? Quick sort? Binary search? Exploration matters; you *want* diversity here.

A single global temperature can't serve both. Lower it and you lose exploration at forks. Raise it and distractors flood back in at locks. Every fixed temperature is a compromise.

**SSD resolves this by baking context-dependent reshaping into the weights.** It does two things simultaneously:
1. **Support compression** — trims the diffuse tail of low-probability distractors
2. **Within-support reshaping** — redistributes mass among the surviving tokens

The result: locks get sharper, forks stay broad, and higher evaluation temperatures become newly effective. The model becomes both more precise *and* more explorable. That's the trick.

### Minimal Filtering is a Feature, Not a Bug

SSD uses **no correctness signal whatsoever**. The synthesized data isn't filtered by execution, test cases, or any quality metric. The paper shows this explicitly: at T_train=2.0 without truncation, ~62% of samples are multilingual gibberish — and the fine-tuned model *still* improves by +5.7pp.

The useful signal comes from how temperature-shifted sampling reshapes the token distribution, not from the semantic content of the outputs. This is why AveryML's `filters.py` only removes empty responses and single-line stubs — anything more would fight the mechanism.

---

## Configuration

Every pipeline step is driven by a Pydantic config with YAML serialization. CLI flags override YAML values:

```bash
# Load from YAML, override temperature and top-k
averyml synthesize \
    --config configs/synthesis/default.yaml \
    --temperature 2.0 \
    --top-k 5
```

Missing `--config`? You'll get a helpful error telling you exactly which fields are required — not a raw Python traceback.

### Key hyperparameters (from the paper)

| Parameter | Default | Paper range | What it does |
|---|---|---|---|
| T_train | 1.5 | 0.5 - 2.0 | Sampling temperature for synthesis. Higher = more diverse (and more wrong) samples |
| T_eval | 0.6 | 0.4 - 1.5 | Decoding temperature at evaluation. Tuned independently from T_train |
| top-k (train) | 10 | 5 - 20 | Truncation during synthesis. Adds a second improvement channel on top of temperature |
| N (samples/prompt) | 1 | 1 | Samples per prompt. One is enough — the magic is in temperature, not volume |
| Training steps | 2500 | 300 - 2500 | 2500 for instruct models, 300 for thinking models |
| Learning rate | 5e-6 | — | AdamW with cosine decay and warmup |

**The secret sauce:** T_eff = T_train x T_eval. The paper finds a quadratic peak near T_eff ~ 1.2 (R^2=0.75). The grid search finds this automatically for your model.

---

## Supported Models

SSD generalizes across model families, scales, and reasoning styles:

| Model | Type | Notes |
|---|---|---|
| Qwen3-4B-Instruct | Dense, instruct | Good starting point for quick experiments |
| Qwen3-4B-Thinking | Dense, thinking | Use `sft_thinking.yaml` (300 steps, 50 warmup) |
| Qwen3-30B-Instruct | MoE, instruct | Best absolute results in the paper (+12.9pp) |
| Qwen3-30B-Thinking | MoE, thinking | Largest gains on hard problems (+15.3pp) |
| Llama-3.1-8B-Instruct | Dense, instruct | Cross-family generalization proof |

In principle, SSD works with any instruction-tuned model that can generate code. The config system makes it easy to try your own.

---

## Docker

Run on cloud GPUs (Lambda, RunPod, vast.ai) without dependency headaches:

```bash
# Build the image
docker build -t averyml .

# Full pipeline with GPU
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    -e HF_TOKEN=$HF_TOKEN \
    averyml run-pipeline --config configs/experiments/full_pipeline_qwen3_4b.yaml

# Launch dashboard (accessible at http://localhost:7860)
docker run --gpus all -p 7860:7860 averyml dashboard --share

# Or use docker compose for the full setup
HF_TOKEN=hf_xxx docker compose run averyml synthesize --config configs/synthesis/default.yaml
```

The `docker-compose.yml` handles GPU passthrough, persistent volumes for data/checkpoints/model cache, and environment variables for HuggingFace and W&B tokens.

---

## Analysis Tools

Dig into *why* SSD works — reproduce the paper's analysis on your own models:

```bash
averyml analyze \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --ssd-model ./checkpoints/sft_instruct/final_checkpoint \
    --analysis-type all
```

This produces:
- **Distribution comparison** — cumulative mass curves showing SSD's faster probability concentration (Figure 6a)
- **Fork/lock detection** — per-position entropy and top-1 probability, showing where the model sharpens vs preserves diversity (Figure 4)
- **Compression decomposition** — the three terms from Equation 4: support compression, within-support reshaping, alignment

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full test suite (101 tests, ~0.5s)
pytest

# Lint
ruff check averyml/

# Type check
mypy averyml/
```

Tests cover: config serialization, registry population, filter logic, pass@k estimation, metric computation, temperature grid construction, result storage, dashboard data helpers, sandbox utilities, and more. GPU-dependent integration tests (model loading, generation, evaluation) require appropriate hardware.

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

MIT — do whatever you want with it.
