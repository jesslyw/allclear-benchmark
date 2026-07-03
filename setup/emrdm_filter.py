"""Keep only the AllClear test samples that EMRDM can run on, using the SEN12MS-CR checkpoint.

A sample is kept only if it has both S1 and S2 data. AllClear provides multiple S2 timestamps per sample, so this script picks the S2 frame closest in time to the selected S1 frame.

Outputs:
- emrdm_pairs.json: pair choices plus per-sample metadata (acts like a pointer/index into the full dataset).
- emrdm_samples.json: the EMRDM-compatible subset, copied in the same structure as the original AllClear metadata.
- optional ROI list (--roi-list-out): unique ROI IDs from metadata-only eligibility checks.
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

_occlusion_cache = {}


def parse_dt(value):
    return datetime.strptime(value, TIME_FORMAT)


def frame_occlusion(s2_fpath):
    """Cloud/shadow fraction (0-1) of one S2 frame, or None if no mask."""
    if s2_fpath in _occlusion_cache:
        return _occlusion_cache[s2_fpath]
    mask_fpath = s2_fpath.replace("s2_toa", "cld_shdw")
    if not os.path.exists(mask_fpath):
        _occlusion_cache[s2_fpath] = None
        return None
    with rs.open(mask_fpath) as src:
        cloud = src.read(CLOUD_BAND)
        shadow = src.read(SHADOW_BAND)
    cloud = np.nan_to_num(cloud, nan=1.0)
    shadow = np.nan_to_num(shadow, nan=1.0)
    frac = float(((cloud + shadow) > 0).mean())
    _occlusion_cache[s2_fpath] = frac
    return frac


def find_emrdm_pair(sample):
    s2_entries = sample.get("s2_toa", [])
    s1_entries = sample.get("s1", [])

    if not s1_entries:
        return None, False

    s1_dts = [parse_dt(ts) for ts, _ in s1_entries]

    candidates = []

    for s2_idx, (s2_ts, s2_fpath) in enumerate(s2_entries):
        occlusion = frame_occlusion(s2_fpath)
        if occlusion is None:
            continue

        s2_dt = parse_dt(s2_ts)

        # nearest S1 in time
        s1_gaps_seconds = [abs((s2_dt - s1_dt).total_seconds())
                           for s1_dt in s1_dts]
        best_s1_idx = min(range(len(s1_gaps_seconds)),
                          key=lambda i: s1_gaps_seconds[i])
        delta_seconds = s1_gaps_seconds[best_s1_idx]
        delta_days = delta_seconds / 86400.0

        payload = {
            "s2_index": s2_idx,
            "s1_index": best_s1_idx,
            "s2_cloudy_fraction": round(occlusion, 4),
            "s2_s1_delta_days": round(delta_days, 6),
        }
        candidates.append((int(delta_seconds), s2_idx, payload))

    if not candidates:
        return None, False

    # smaller time gap wins; equal gaps use smallest s2_index as fallback
    def score(c):
        return (c[0], c[1])

    best = min(candidates, key=score)

    # tie: 2+ frames shared the smallest gap; index fallback decided
    min_gap = best[0]
    tie = sum(1 for c in candidates if c[0] == min_gap) > 1

    return best[2], tie


def collect_download_rois(dataset):
    """Pre-download: creates a list of emrdm eligable rois based on the emrdm filter criteria. This should be used with allclear-download.py and the --roi-file flag to download the required data, before running the full emrdm_filter.py script"""
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
        description="Filter AllClear metadata JSON to EMRDM-eligible samples"
    )
    parser.add_argument(
        "--metadata-json",
        default="metadata/datasets/test_tx3_s2-s1_100pct_1proi.json",
        help="Path to AllClear metadata JSON",
    )
    parser.add_argument(
        "--pairs-out",
        default="setup/emrdm_pairs.json",
        help="Output path for EMRDM pairs metadata",
    )
    parser.add_argument(
        "--data-out",
        default="setup/emrdm_samples.json",
        help="Output path for EMRDM samples JSON (AllClear format)",
    )
    parser.add_argument(
        "--roi-list-out",
        default=None,
        help=(
            "Write metadata-only candidate ROI IDs (one per line) and exit. "
            "Useful as a pre-download step."
        ),
    )
    args = parser.parse_args()

    with open(args.metadata_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.roi_list_out:
        rois = collect_download_rois(dataset)
        Path(args.roi_list_out).write_text(
            "\n".join(rois) + "\n", encoding="utf-8")
        print(f"Saved {len(rois)} candidate ROIs to {args.roi_list_out}")
        return

    pairs = {}
    for sample_id, sample in tqdm(dataset.items(), desc="Filtering"):
        if not sample.get("s2_toa") or not sample.get("target"):
            continue
        pair, _ = find_emrdm_pair(sample)
        if pair is not None:
            pairs[sample_id] = pair

    missing = sum(1 for v in _occlusion_cache.values() if v is None)
    if missing:
        print(
            f"WARNING: {missing} S2 frames had no local cld_shdw mask and were skipped.")

    Path(args.pairs_out).write_text(
        json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(pairs)} EMRDM-eligible pairs to {args.pairs_out}")

    samples_subset = {sid: dataset[sid] for sid in pairs}
    Path(args.data_out).write_text(
        json.dumps(samples_subset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(samples_subset)} samples to {args.data_out}")
    print(
        f"Eligible: {len(pairs)}/{len(dataset)} ({len(pairs)/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    main()
