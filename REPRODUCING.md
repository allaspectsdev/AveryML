# Reproducing Paper Results

This guide walks through reproducing the results from Table 2 of [Embarrassingly Simple Self-Distillation Improves Code Generation](https://arxiv.org/abs/2604.01193) (Apple, 2026).

## Quick Start

AveryML includes preset configs that match the paper's exact hyperparameters:

```bash
# Reproduce Qwen3-4B-Instruct result (+7.5pp pass@1 on LCB v6)
averyml reproduce-paper --model qwen3_4b

# Reproduce the headline Qwen3-30B-Instruct result (+12.9pp pass@1)
averyml reproduce-paper --model qwen3_30b

# Reproduce Llama-3.1-8B-Instruct result (+3.5pp pass@1)
averyml reproduce-paper --model llama_8b

# Reproduce Qwen3-4B-Thinking result (+3.3pp pass@1)
averyml reproduce-paper --model qwen3_4b_thinking
```

Each command runs the full pipeline (synthesize -> train -> evaluate) with the exact hyperparameters from the paper.

## Preset Configs vs Paper

Each preset in `configs/presets/` maps to a specific row in Table 2:

| Preset | Paper Model | T_train | T_eval | top-k | Iterations | Expected pass@1 delta |
|---|---|---|---|---|---|---|
| `qwen3_4b_instruct` | Qwen3-4B-Instruct | 2.0 | 1.1 | 10 | 2500 | +7.5pp |
| `qwen3_4b_thinking` | Qwen3-4B-Thinking | 1.5 | 0.6 | 10 | 300 | +3.3pp |
| `qwen3_30b_instruct` | Qwen3-30B-Instruct | 2.0 | 1.1 | 10 | 2500 | +12.9pp |
| `llama_8b_instruct` | Llama-3.1-8B-Instruct | 1.5 | 0.6 | 10 | 2500 | +3.5pp |

## Step-by-Step: Reproducing Table 2 (Qwen3-4B-Instruct)

### Prerequisites

- 4x A100 80GB GPUs (or equivalent)
- ~10K competitive programming prompts (rSTARcoder seed subset)
- Python 3.10+, CUDA 12.4+

### Step 1: Evaluate the Base Model

First, establish the baseline:

```bash
averyml evaluate \
    --config configs/evaluation/lcb_v6.yaml \
    --model-id "Qwen/Qwen3-4B-Instruct-2507" \
    --temperature 0.6 \
    --n-repeat 20
```

Expected: ~34.8% pass@1 (paper reports 34.8%).

### Step 2: Run SSD

```bash
averyml reproduce-paper --model qwen3_4b
```

This runs:
1. **Synthesis**: Samples ~10K solutions at T_train=2.0, top-k=10
2. **Training**: Fine-tunes for 2500 steps with AdamW, cosine LR (5e-6)
3. **Evaluation**: Evaluates at T_eval=1.1 on LCB v6

Expected: ~42.4% pass@1 (+7.5pp over base).

### Step 3: Compare

```bash
averyml compare \
    results/Qwen_Qwen3-4B-Instruct-2507/results_*.json \
    results/qwen3_4b_instruct/results_*.json
```

This shows pass@k deltas with per-difficulty breakdowns.

## Reproducing the Temperature Sweep (Figure 3)

```bash
averyml search \
    --config configs/search/temperature_grid.yaml \
    --base-model-id "Qwen/Qwen3-4B-Instruct-2507"
```

This runs a 5x7 grid over (T_train, T_eval). Launch the dashboard to see the interactive heatmap:

```bash
averyml dashboard
```

## Hardware Requirements

| Model | Synthesis | Training | Evaluation |
|---|---|---|---|
| Qwen3-4B | 1x A100 | 1x A100 | 1x A100 |
| Qwen3-30B | 4x A100 | 8x A100 | 4x A100 |
| Llama-3.1-8B | 1x A100 | 1x A100 | 1x A100 |

Training time estimates (8x B200):
- 4B models: ~2 hours (2500 steps)
- 30B models: ~8 hours (2500 steps)
- Thinking variants: ~20 minutes (300 steps)

## Key Hyperparameters

If you're adapting SSD to your own model, the most important parameters are:

1. **T_train** (1.5-2.0): Higher = more diverse samples. The paper finds T_train=2.0 works best.
2. **T_eval** (0.8-1.2): Tuned independently. Use the grid search to find the best value.
3. **top-k** (5-10): Truncation during synthesis. Adds a second improvement channel.
4. **T_eff = T_train x T_eval** (~1.2): The effective temperature governs performance. Aim for the quadratic peak.

## Differences from the Paper

AveryML is a faithful reimplementation, but there are minor differences:

- **Training framework**: Paper uses Megatron-LM on 8x B200. AveryML uses HuggingFace Trainer. Results may differ by 0.5-1pp due to numerical differences.
- **Prompt source**: If you don't have the exact rSTARcoder seed subset, AveryML falls back to LiveCodeBench problems. This uses a different distribution and may affect results.
- **Evaluation**: We use the same LiveCodeBench v6 dataset and evaluation harness as the paper.

## Troubleshooting

**Q: My pass@1 is 2-3pp lower than the paper.**
A: Check that (1) T_train and T_eval match the preset, (2) you're using enough training iterations, (3) you haven't accidentally used the thinking config for an instruct model. AveryML warns about this automatically.

**Q: Synthesis is very slow.**
A: Use `--backend vllm` (default) with `--tensor-parallel-size` matching your GPU count. The HF backend is 10-50x slower.

**Q: Training OOMs.**
A: Enable `packing: true` and `gradient_checkpointing: true` in your training config. For 30B models, use `use_lora: true`.
