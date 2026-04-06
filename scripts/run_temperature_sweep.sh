#!/bin/bash
# Run the temperature grid search (reproduces Figure 3 from the paper)
set -euo pipefail

CONFIG="${1:-configs/search/temperature_grid.yaml}"
DIAGONAL_ONLY="${2:---diagonal-only}"

echo "=== AveryML: Temperature Grid Search ==="
echo "Config: $CONFIG"
echo ""

averyml search \
    --config "$CONFIG" \
    $DIAGONAL_ONLY
