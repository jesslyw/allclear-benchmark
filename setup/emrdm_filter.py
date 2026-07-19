"""Keep only the AllClear test samples that EMRDM can run on, using the SEN12MS-CR checkpoint.

Two-phase filtering approach:
- Phase 1 (uses AllClear's test_tx3_s2-s1_100pct_1proi.json metadata): identify ROIs with S1, S2, and target data for download via allclear-download.py
- Phase 2: using downloaded data, select EMRDM eligible S1-S2 pairs per ROI

Outputs:
- emrdm_pairs.json: references the selected S1-S2 pairs per ROI in test_tx3_s2-s1_100pct_1proi.json, plus pair metadata (cloudy %, delta days)
- optional ROI list (--roi-list-out): ROI IDs from Phase 1, used for data download before Phase 2

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

# Mask bands, see dataset.py channels["cld_shdw"].
CLOUD_BAND = 2
SHADOW_BAND = 5


def parse_dt(value):
    return datetime.strptime(value, TIME_FORMAT)


def frame_occlusion(s2_fpath):
    """Helper function: Receives the path of an S2 frame and returns its cloud/shadow fraction (0-1) or None if no mask found."""
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


def phase2_find_pair(sample):
    """Phase 2: for a sample, find the best S1-S2 pair by temporal proximity.
    Returns tuple: (pairing_metadata, has_tie, skipped_frame_count) or (None, False, skipped_count) if no valid pair.
    """
    s2_entries = sample.get("s2_toa", [])
    s1_entries = sample.get("s1", [])

    if not s1_entries:
        return None, False, 0

    s1_dts = [parse_dt(ts) for ts, _ in s1_entries]

    candidates = []
    skipped = 0

    for s2_idx, (s2_ts, s2_fpath) in enumerate(s2_entries):
        occlusion = frame_occlusion(s2_fpath)
        if occlusion is None:
            skipped += 1  # Skip s2 frames with missing cloud/shadow mask files
            continue

        s2_dt = parse_dt(s2_ts)

        # Find nearest S1 by time gap
        s1_gaps_seconds = [abs((s2_dt - s1_dt).total_seconds())
                           for s1_dt in s1_dts]
        delta_seconds = min(s1_gaps_seconds)
        best_s1_idx = s1_gaps_seconds.index(delta_seconds)
        delta_days = delta_seconds / 86400.0

        payload = {
            "s2_index": s2_idx,
            "s1_index": best_s1_idx,
            "s2_cloudy_fraction": round(occlusion, 4),
            "s2_s1_delta_days": round(delta_days, 6),
        }
        # Store candidate as: (time_gap_in_seconds, s2_index, metadata_payload)
        candidates.append((int(delta_seconds), s2_idx, payload))

    if not candidates:
        return None, False, skipped

    # Select pairing with smallest s1-s2 time gap; use smallest s2_idx as tiebreaker
    best = min(candidates, key=lambda c: (c[0], c[1]))
    best_delta_seconds = best[0]
    best_payload = best[2]
    tied_frames = [c for c in candidates if c[0] == best_delta_seconds]
    has_tie = len(tied_frames) > 1

    return best_payload, has_tie, skipped


def phase1_collect_rois(dataset):
    """Phase 1: returns ROI IDs with at least one S1 frame, one S2 frame, and target data.
    Use output with allclear-download.py --roi-file to download only these candidates before Phase 2 filtering.
    """
    rois = set()
    for sample in dataset.values():
        if not sample.get("s1") or not sample.get("s2_toa") or not sample.get("target"):
            continue
        roi_values = sample.get("roi", [])
        if roi_values:
            rois.add(roi_values[0])
    return sorted(rois)


def main():
    parser = argparse.ArgumentParser(
        description="Filter AllClear dataset into EMRDM-compatible samples"
    )
    parser.add_argument(
        "--pairs-out",
        default="setup/emrdm_pairs.json",
        help="Output path for EMRDM pairs metadata (filtering index)",
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
        # Phase 1: List of all ROIs with s1, s2, and target for download
        rois = phase1_collect_rois(dataset)
        Path(args.roi_list_out).write_text(
            "\n".join(rois) + "\n", encoding="utf-8")
        print(f"Saved {len(rois)} candidate ROIs to {args.roi_list_out}")
        return

 # Phase 2: For each downloaded ROI, find the best s1 s2 pair by closest temporal gap
    pairs = {}
    skipped_frames = 0
    for sample_id, sample in tqdm(dataset.items(), desc="Filtering"):
        if not sample.get("s2_toa") or not sample.get("target"):
            continue
        pair, _, skipped = phase2_find_pair(sample)
        skipped_frames += skipped
        if pair is not None:
            pairs[sample_id] = pair

    if skipped_frames:
        print(
            f"WARNING: {skipped_frames} S2 frames had no local cld_shdw mask and were skipped.")

    Path(args.pairs_out).write_text(
        json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(pairs)} EMRDM-eligible pairs to {args.pairs_out}")
    print(
        f"Eligible: {len(pairs)}/{len(dataset)} ({len(pairs)/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    main()
