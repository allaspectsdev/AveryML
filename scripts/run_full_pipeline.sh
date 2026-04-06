#!/bin/bash
# Run the full SSD pipeline: synthesis -> training -> evaluation
set -euo pipefail

CONFIG="${1:-configs/experiments/full_pipeline_qwen3_4b.yaml}"

echo "=== AveryML: Full SSD Pipeline ==="
echo "Config: $CONFIG"
echo ""

averyml run-pipeline --config "$CONFIG"
