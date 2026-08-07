#!/usr/bin/env bash
# Run from repo root: bash setup/setup.sh
set -e

# --- vpint2 pairs config ---
CPUS=8

VPINT2_ROI_LIST=setup/vpint2_candidates.txt
EMRDM_ROI_LIST=setup/emrdm_candidates.txt
INTERSECTION_ROI_LIST=setup/intersection_candidates.txt
VPINT2_PAIRS=setup/vpint2_pairs.json
EMRDM_PAIRS=setup/emrdm_pairs.json
INTERSECTION_SAMPLES=setup/intersection_samples.json
EMRDM_MAX_DAYS=2.0

# --- steps ---

python3 setup/allclear_download.py --metadata-only

# 1) Generate candidate ROI lists (metadata-only) and run full EMRDM filter
python3 setup/vpint2_filter.py \
    --roi-list-out "$VPINT2_ROI_LIST"

python3 setup/emrdm_filter.py \
    --roi-list-out "$EMRDM_ROI_LIST" \
    --max-days "$EMRDM_MAX_DAYS"

# 2) Intersect VPint2 candidates with EMRDM candidates, download only that set
comm -12 <(sort "$VPINT2_ROI_LIST") <(sort "$EMRDM_ROI_LIST") > "$INTERSECTION_ROI_LIST"
echo "[INFO] Intersection: $(wc -l < "$INTERSECTION_ROI_LIST") ROIs to download"

python3 setup/allclear_download.py \
    --roi-file "$INTERSECTION_ROI_LIST" \
    --skip-metadata \
    --cpus "$CPUS"

# 3) Run full VPint2 filter (uses downloaded masks)

python3 setup/vpint2_filter.py
python3 setup/emrdm_filter.py --max-days "$EMRDM_MAX_DAYS"

# 4) Intersect emrdm and vpint2 pairs to create the final sample set
python3 setup/intersection_samples.py \
    --emrdm-pairs-fpath "$EMRDM_PAIRS" \
    --vpint2-pairs-fpath "$VPINT2_PAIRS" \
    --out-fpath "$INTERSECTION_SAMPLES"

echo ""
echo "================================"
echo "  Setup complete."
echo "  Run: python3 benchmark.py --model-name <model> to start"
echo "================================"
echo ""
echo ""
echo ""

