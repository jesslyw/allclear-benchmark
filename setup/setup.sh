#!/usr/bin/env bash
# Run from repo root: bash setup/setup.sh
set -e

# --- vpint2 pairs config ---
CPUS=8

VPINT2_ROI_LIST=setup/vpint2_candidates.txt
EMRDM_ROI_LIST=setup/emrdm_candidates.txt
INTERSECTION_ROI_LIST=setup/intersection_candidates.txt
VPINT2_SAMPLES=setup/vpint2_samples.json
EMRDM_SAMPLES=setup/emrdm_samples.json
INTERSECTION_SAMPLES=setup/intersection_samples.json

# --- steps ---

python setup/allclear_download.py --metadata-only

# 1) Generate candidate ROI lists (metadata-only) and run full EMRDM filter
python setup/vpint2_filter.py \
    --screen \
    --candidates-out "$VPINT2_ROI_LIST"

python setup/emrdm_filter.py \
    --roi-list-out "$EMRDM_ROI_LIST"

python setup/emrdm_filter.py

# 2) Intersect VPint2 candidates with EMRDM candidates, download only that set
comm -12 <(sort "$VPINT2_ROI_LIST") <(sort "$EMRDM_ROI_LIST") > "$INTERSECTION_ROI_LIST"
echo "[INFO] Intersection: $(wc -l < "$INTERSECTION_ROI_LIST") ROIs to download"

python setup/allclear_download.py \
    --roi-file "$INTERSECTION_ROI_LIST" \
    --skip-metadata \
    --cpus "$CPUS"

# 3) Run full VPint2 filter (uses downloaded masks)

python setup/vpint2_filter.py

# 4) Intersect emrdm and vpint2 samples to create the final sample set
python setup/intersection_samples.py \
    --emrdm-samples-fpath "$EMRDM_SAMPLES" \
    --vpint2-samples-fpath "$VPINT2_SAMPLES" \
    --out-fpath "$INTERSECTION_SAMPLES"

echo ""
echo "================================"
echo "  Setup complete."
echo "  Run: python benchmark.py --model-name <model> to start"
echo "================================"
echo ""
echo ""
echo ""

