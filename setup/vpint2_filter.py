"""Filter AllClear test samples to VPint2-eligible candidates.

Two-phase filtering approach:
- Phase 1 (uses AllClear's test_tx3_s2-s1_100pct_1proi.json metadata + s2_metadata.csv): Screen ROI candidates using cloud/shadow percentages for data download via allclear-download.py
- Phase 2: using downloaded data, select VPint2 eligible reference+cloudy S2 pairs per ROI

Outputs:
- vpint2_pairs.json: references the selected reference+cloudy S2 pairs per ROI in test_tx3_s2-s1_100pct_1proi.json, plus pair metadata (cloud %, gap days)
- optional ROI list (--roi-list-out): ROI IDs from Phase 1, used for data download before Phase 2

Note: Full sample metadata is retrieved from the original test_tx3_s2-s1_100pct_1proi.json using vpint2_pairs.json as the filtering index.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio as rs
from tqdm import tqdm


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_REF_CLOUD = 0.10
MAX_TARGET_GAP_DAYS = 10
CSV_BOUNDS_PATH = "metadata/data/s2_metadata.csv"

# Mask bands (1-indexed, as read by rasterio) — see dataset.py channels["cld_shdw"].
CLOUD_BAND = 2
SHADOW_BAND = 5


def string_to_datetime(value):
    return datetime.strptime(value, TIME_FORMAT)


def frame_occlusion(s2_fpath):
    """Return how much of one S2 frame is covered by cloud or shadow.

    We read the matching cld_shdw mask and mark a pixel as occluded if cloud
    or shadow is present. The result is a fraction between 0 and 1. NaN values
    are treated as occluded pixels. If the mask file is missing locally,
    returns None.
    """
    mask_fpath = s2_fpath.replace("s2_toa", "cld_shdw")
    if not os.path.exists(mask_fpath):
        return None
    with rs.open(mask_fpath) as src:
        cloud = src.read(CLOUD_BAND)
        shadow = src.read(SHADOW_BAND)
    cloud = np.nan_to_num(cloud, nan=1.0)
    shadow = np.nan_to_num(shadow, nan=1.0)
    frac = float(((cloud + shadow) > 0).mean())
    return frac


def load_csv_bounds(csv_path):
    """Read cloud/shadow percentages from s2_metadata.csv.

    Returns {(roi, capture_date): (cloud_frac, shadow_frac)} and skips rows
    with missing or invalid values.
    """
    import csv as _csv
    bounds = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            c = row.get("cloud_percentage_30", "")
            s = row.get("shadow_percentage_30", "")
            if c in ("", "nan") or s in ("", "nan"):
                continue
            c = float(c)
            s = float(s)
            if c < 0 or s < 0:
                continue
            bounds[(row["roi"], row["capture_date"])] = (c / 100.0, s / 100.0)
    return bounds


def phase1_screen_vpint2(sample, csv_bounds):
    """Phase 1: List ROIs for download by checking if a clear reference exists before a cloudy frame.

    Returns "skip", "eligible", or "uncertain" based on s2_metadata.csv cloud/shadow percentages.
    """
    roi = sample["roi"][0]
    cloudy_accept_dts = []
    cloudy_straddle_dts = []
    ref_accept_dts = []
    ref_straddle_dts = []

    def has_before(ref_dts, cloudy_dts):
        return any(ref_dt < cloudy_dt for ref_dt in ref_dts for cloudy_dt in cloudy_dts)

    for ts, _ in sample.get("s2_toa", []):
        v = csv_bounds.get((roi, ts))
        if v is None:
            continue
        ts_dt = string_to_datetime(ts)
        floor = max(v[0], v[1])
        ceil = min(v[0] + v[1], 1.0)
        # cloudy frame should be partly cloudy (not fully clear/covered)
        if ceil <= 0 or floor >= 1:
            pass
        elif floor > 0 and ceil < 1:
            cloudy_accept_dts.append(ts_dt)
        else:
            cloudy_straddle_dts.append(ts_dt)

        # reference frame should be clear enough (<10% cloud)
        if ceil <= MAX_REF_CLOUD:
            ref_accept_dts.append(ts_dt)
        elif floor > MAX_REF_CLOUD:
            pass
        else:
            ref_straddle_dts.append(ts_dt)

    ordered_accept = has_before(ref_accept_dts, cloudy_accept_dts)
    ordered_possible = has_before(
        ref_accept_dts + ref_straddle_dts,
        cloudy_accept_dts + cloudy_straddle_dts,
    )

    if not ordered_possible:
        return "skip"
    if ordered_accept:
        return "eligible"
    return "uncertain"


def phase2_find_pair(sample, missing_masks):
    """Phase 2: Find the best reference+cloudy S2 pair by selection criteria.

    Selection priority: (1) minimize cloudy-target temporal gap ≤10 days, (2) minimize reference cloud%,
    (3) maximize cloudy cloud% (prefer more cloudy frames).
    Returns pairing metadata dict or None if no valid pair exists.
    """
    target_date = sample["target"][0][0]
    target_dt = string_to_datetime(target_date)

    inputs = sample["s2_toa"]

    best = None

    for cloudy_idx, cloudy_input in enumerate(inputs):
        cloudy_date, cloudy_fpath = cloudy_input[0], cloudy_input[1]
        cloudy_dt = string_to_datetime(cloudy_date)

        cloudy_cloud = frame_occlusion(cloudy_fpath)
        if cloudy_cloud is None:
            missing_masks.add(cloudy_fpath)
            continue

        # cloudy frame must be partly cloudy (not completely clear or covered)
        if cloudy_cloud <= 0 or cloudy_cloud >= 1:
            continue

        cloudy_target_delta = abs((target_dt - cloudy_dt).days)
        if cloudy_target_delta > MAX_TARGET_GAP_DAYS:
            continue

        refs = []

        for ref_idx, ref_input in enumerate(inputs):
            ref_date, ref_fpath = ref_input[0], ref_input[1]
            ref_dt = string_to_datetime(ref_date)

            if ref_dt >= cloudy_dt:
                continue

            ref_cloud = frame_occlusion(ref_fpath)
            if ref_cloud is None:
                missing_masks.add(ref_fpath)
                continue

            if ref_cloud <= MAX_REF_CLOUD:
                refs.append((ref_idx, ref_dt, ref_cloud))

        if not refs:
            continue

        ref_idx, ref_dt, ref_cloud = min(
            refs,
            key=lambda x: (
                x[2],
                abs((cloudy_dt - x[1]).days),
            ),
        )

        candidate = {
            "cloudy_index": cloudy_idx,
            "reference_index": ref_idx,
            "cloudy_occlusion_pct": round(cloudy_cloud, 4),
            "reference_occlusion_pct": round(ref_cloud, 4),
            "cloudy_reference_delta_days": abs((cloudy_dt - ref_dt).days),
            "cloudy_target_delta_days": cloudy_target_delta,
        }

        if best is None:
            best = candidate
            continue

        # Selection tuple: (cloudy_target_delta, reference_cloud%, -cloudy_cloud%)
        # Priority: (1) smallest target gap, (2) lowest reference cloud, (3) highest cloudy opacity
        if (
            candidate["cloudy_target_delta_days"],
            candidate["reference_occlusion_pct"],
            -candidate["cloudy_occlusion_pct"],
        ) < (
            best["cloudy_target_delta_days"],
            best["reference_occlusion_pct"],
            -best["cloudy_occlusion_pct"],
        ):
            best = candidate

    return best


def main():
    parser = argparse.ArgumentParser(
        description="Filter AllClear dataset into VPint2-compatible samples"
    )
    parser.add_argument(
        "--pairs-out",
        default="setup/vpint2_pairs.json",
        help="Output path for VPint2 pairs metadata (filtering index)",
    )

    parser.add_argument(
        "--roi-list-out",
        default=None,
        help=(
            "Phase 1: Pre-filter candidate ROIs to download and write them to a file (one per line). "
            "Use output with allclear-download.py --roi-file before running Phase 2."
        ),
    )

    args = parser.parse_args()

    metadata_json = "metadata/datasets/test_tx3_s2-s1_100pct_1proi.json"
    with open(metadata_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.roi_list_out:
        # Phase 1: List all ROI candidates to download. Eliminates all ROIs that do not meet VPint2's input criteria (clear s2 frame occuring before a cloudy s2 frame). Uses s2_metadata.csv as reference for cloud/shadow coverage
        csv_bounds = load_csv_bounds(CSV_BOUNDS_PATH)
        eligible = set()
        uncertain = set()
        for sample in dataset.values():
            if not sample.get("s2_toa") or not sample.get("target"):
                continue
            status = phase1_screen_vpint2(sample, csv_bounds)
            roi = sample["roi"][0]
            if status == "eligible":
                eligible.add(roi)
            elif status == "uncertain":
                uncertain.add(roi)
        candidates = sorted(eligible | uncertain)
        Path(args.roi_list_out).write_text(
            "\n".join(candidates) + "\n", encoding="utf-8")
        total = len(set(s["roi"][0] for s in dataset.values()))
        print(f"Saved {len(candidates)} candidate ROIs to {args.roi_list_out}")
        return

    # Phase 2: For each sample, find the best clear/cloudy S2 pair by closest cloudy to target temporal gap
    subset = {}
    missing_masks = set()

    for sample_id, sample in tqdm(dataset.items()):
        if not sample.get("s2_toa") or not sample.get("target"):
            continue

        pair = phase2_find_pair(sample, missing_masks)

        if pair is not None:
            subset[sample_id] = pair

    if missing_masks:
        print(
            f"WARNING: {len(missing_masks)} S2 frames had no local cld_shdw mask and were skipped.")

    Path(args.pairs_out).write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(subset)} VPint2-eligible pairs to {args.pairs_out}")


if __name__ == "__main__":
    main()
