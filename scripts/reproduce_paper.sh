#!/bin/bash
# Reproduce key results from the SSD paper (Table 2) for Qwen3-4B-Instruct
set -euo pipefail

echo "=== AveryML: Reproducing SSD Paper Results ==="
echo ""

# Step 1: Synthesize training data
echo "--- Step 1: Data Synthesis ---"
averyml synthesize \
    --config configs/synthesis/default.yaml \
    --temperature 1.5 \
    --top-k 10

# Step 2: Fine-tune
echo "--- Step 2: Fine-tuning ---"
averyml train \
    --config configs/training/sft_instruct.yaml

# Step 3: Evaluate on LCB v6
echo "--- Step 3: Evaluate on LiveCodeBench v6 ---"
averyml evaluate \
    --config configs/evaluation/lcb_v6.yaml

# Step 4: Also evaluate on LCB v5
echo "--- Step 4: Evaluate on LiveCodeBench v5 ---"
averyml evaluate \
    --config configs/evaluation/lcb_v5.yaml

echo "=== Done ==="
