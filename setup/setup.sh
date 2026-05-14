#!/usr/bin/env bash
# Run from repo root: bash setup/setup.sh
set -e

# --- vpint2 pairs config ---
MAX_REF_CLOUD=0.10
MAX_GAP_DAYS_CLOUDY_TARGET=5
CPUS=8

# --- steps ---

#python setup/allclear_download.py --metadata-only

python setup/vpint2_filter.py \
    --max-ref-cloud "$MAX_REF_CLOUD" \
    --max-gap-days "$MAX_GAP_DAYS_CLOUDY_TARGET"

python setup/allclear_download.py \
    --dataset-fpath setup/vpint2_samples.json \
    --skip-metadata \
    --cpus "$CPUS"

echo ""
echo "================================"
echo "  Setup complete."
echo "  Run: python benchmark.py --model-name <model> to start"
echo "================================"
echo ""
echo ""
echo ""

