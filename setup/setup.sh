#!/usr/bin/env bash
# Run from repo root: bash setup/setup.sh
set -e

CPUS=8

# filenames of ROI lists to download 
VPINT2_ROI_LIST=setup/vpint2_candidates.txt
EMRDM_ROI_LIST=setup/emrdm_candidates.txt
INTERSECTION_ROI_LIST=setup/intersection_candidates.txt

VPINT2_PAIRS=setup/vpint2_pairs.json
EMRDM_PAIRS=setup/emrdm_pairs.json
INTERSECTION_SAMPLES=setup/intersection_samples.json
EMRDM_MAX_DAYS=2.0

# --- steps ---

python3 setup/allclear_download.py --metadata-only

# 1) Using AllClear's test set metadata, generate a list(s) of samples meeting the input criteria of both VPint2 and EMRDM for download
python3 setup/vpint2_filter.py \
    --roi-list-out "$VPINT2_ROI_LIST"

python3 setup/emrdm_filter.py \
    --roi-list-out "$EMRDM_ROI_LIST" \
    --max-days "$EMRDM_MAX_DAYS"

# 2) Intersect both VPint2 and EMRDM ROI lists, and download only that set
comm -12 <(sort "$VPINT2_ROI_LIST") <(sort "$EMRDM_ROI_LIST") > "$INTERSECTION_ROI_LIST"
echo "[INFO] Intersection: $(wc -l < "$INTERSECTION_ROI_LIST") ROIs to download"

python3 setup/allclear_download.py \
    --roi-file "$INTERSECTION_ROI_LIST" \
    --skip-metadata \
    --cpus "$CPUS"

# 3) Run full filters (uses downloaded cloud/shadow masks from step 2)

python3 setup/vpint2_filter.py
python3 setup/emrdm_filter.py --max-days "$EMRDM_MAX_DAYS"

# 4) Intersect outputs from step 3 to create the final sample set
python3 setup/intersection_samples.py \
    --emrdm-pairs-fpath "$EMRDM_PAIRS" \
    --vpint2-pairs-fpath "$VPINT2_PAIRS" \
    --out-fpath "$INTERSECTION_SAMPLES"

echo ""
echo "================================"
echo "  Setup complete."
echo "  Run: python3 run.py --model-name <model> to start"
echo "================================"
echo ""
echo ""
echo ""

