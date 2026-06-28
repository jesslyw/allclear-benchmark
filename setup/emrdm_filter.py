"""Keep only the AllClear test samples that EMRDM can run on.

A sample is kept if it has an S1 image and an S2 frame that is 20-70%
cloud/shadow; we pair that frame with the closest S1 in time. Cloud/shadow %
is the fraction of pixels that are cloud OR shadow (mask bands 2 and 5, NaN
counts as covered), matching benchmark.py. Needs the mask files downloaded.

Outputs emrdm_pairs.json (chosen frames) and emrdm_samples.json (kept samples).
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


def load_csv_bounds(csv_path):
    """Cloud/shadow % per frame from s2_metadata.csv (for --screen)."""
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


def screen_emrdm(sample, csv_bounds, min_cloud=0.20, max_cloud=0.70):
    """CSV-only guess: "skip", "eligible", or "uncertain"."""
    if not sample.get("s1"):
        return "skip"
    roi = sample["roi"][0]
    has_accept = False
    has_straddle = False
    for ts, _ in sample.get("s2_toa", []):
        v = csv_bounds.get((roi, ts))
        if v is None:
            continue
        floor = max(v[0], v[1])
        ceil = min(v[0] + v[1], 1.0)
        if ceil < min_cloud or floor > max_cloud:
            continue
        if floor >= min_cloud and ceil <= max_cloud:
            has_accept = True
        else:
            has_straddle = True
    if has_accept:
        return "eligible"
    if has_straddle:
        return "uncertain"
    return "skip"


def mean_eligible_occlusion(dataset, min_cloud=0.20, max_cloud=0.70):
    """Mean cloud/shadow % over all eligible frames, or None."""
    occlusions = []
    for sample in dataset.values():
        for _, s2_fpath in sample.get("s2_toa", []):
            occ = frame_occlusion(s2_fpath)
            if occ is None:
                continue
            if min_cloud <= occ <= max_cloud:
                occlusions.append(occ)
    if not occlusions:
        return None
    return float(np.mean(occlusions))


def find_emrdm_pair(sample, min_cloud=0.20, max_cloud=0.70, target_occlusion=None):
    """Best (S2, S1) pair for a sample. Returns (pair, tie).

    Smallest S2-S1 time gap wins; ties go to the frame closest to
    target_occlusion (or most cloud/shadow if no target). pair is None if no
    frame fits; tie is True when 2+ frames shared the smallest gap.
    """
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
        if not (min_cloud <= occlusion <= max_cloud):
            continue

        s2_dt = parse_dt(s2_ts)

        # nearest S1 in time
        s1_gaps = [abs((s2_dt - s1_dt).days) for s1_dt in s1_dts]
        best_s1_idx = min(range(len(s1_gaps)), key=lambda i: s1_gaps[i])
        delta_days = s1_gaps[best_s1_idx]

        candidates.append({
            "s2_index": s2_idx,
            "s1_index": best_s1_idx,
            "s2_occlusion_pct": round(occlusion, 4),
            "s2_s1_delta_days": delta_days,
        })

    if not candidates:
        return None, False

    # smaller time gap wins, then closest to the mean (or most cloud/shadow)
    def score(c):
        if target_occlusion is None:
            return (c["s2_s1_delta_days"], -c["s2_occlusion_pct"])
        return (c["s2_s1_delta_days"],
                abs(c["s2_occlusion_pct"] - target_occlusion))

    best = min(candidates, key=score)

    # tie: 2+ frames shared the smallest gap, so the second criterion decided
    min_gap = best["s2_s1_delta_days"]
    tie = sum(1 for c in candidates if c["s2_s1_delta_days"] == min_gap) > 1

    return best, tie


def main():
    parser = argparse.ArgumentParser(
        description="Filter AllClear JSON to EMRDM-eligible samples"
    )
    parser.add_argument(
        "--metadata-json",
        default="metadata/datasets/test_tx3_s2-s1_100pct_1proi.json",
        help="Path to AllClear test metadata JSON",
    )
    parser.add_argument(
        "--pairs-out",
        default="setup/emrdm_pairs.json",
        help="Output path for EMRDM selection metadata",
    )
    parser.add_argument(
        "--data-out",
        default="setup/emrdm_samples.json",
        help="Output path for EMRDM samples JSON (AllClear format)",
    )
    parser.add_argument(
        "--min-cloud", type=float, default=0.20,
        help="Minimum cloud+shadow fraction for the S2 input (default: 0.20)",
    )
    parser.add_argument(
        "--max-cloud", type=float, default=0.70,
        help="Maximum cloud+shadow fraction for the S2 input (default: 0.70)",
    )
    parser.add_argument(
        "--screen", action="store_true",
        help="CSV-only pre-screen: write the candidate ROI download list using "
             "cloud/shadow numbers, without opening any mask files.",
    )
    parser.add_argument(
        "--csv", default="metadata/data/s2_metadata.csv",
        help="Path to s2_metadata.csv (only used with --screen)",
    )
    parser.add_argument(
        "--candidates-out", default="setup/emrdm_candidates.txt",
        help="Output path for candidate ROI IDs (only used with --screen)",
    )
    args = parser.parse_args()

    with open(args.metadata_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.screen:
        csv_bounds = load_csv_bounds(args.csv)
        eligible = set()
        uncertain = set()
        for sample in dataset.values():
            if not sample.get("s2_toa") or not sample.get("target"):
                continue
            status = screen_emrdm(sample, csv_bounds,
                                  args.min_cloud, args.max_cloud)
            roi = sample["roi"][0]
            if status == "eligible":
                eligible.add(roi)
            elif status == "uncertain":
                uncertain.add(roi)
        candidates = sorted(eligible | uncertain)
        Path(args.candidates_out).write_text(
            "\n".join(candidates) + "\n", encoding="utf-8")
        total = len(set(s["roi"][0] for s in dataset.values()))
        print(f"[screen] candidate ROIs: {len(candidates)} "
              f"(certain-eligible {len(eligible)}, "
              f"uncertain/straddle-only {len(uncertain - eligible)})")
        print(f"[screen] certainly-skip ROIs: {total - len(candidates)}")
        print(f"[screen] wrote candidate ROI list to {args.candidates_out}")
        return

    # tie-break target
    target = mean_eligible_occlusion(dataset, args.min_cloud, args.max_cloud)
    if target is not None:
        print(f"Mean cloud/shadow % of eligible frames: {target:.4f}")

    pairs = {}
    tie_count = 0
    for sample_id, sample in tqdm(dataset.items(), desc="Filtering"):
        if not sample.get("s2_toa") or not sample.get("target"):
            continue
        pair, tie = find_emrdm_pair(
            sample,
            min_cloud=args.min_cloud,
            max_cloud=args.max_cloud,
            target_occlusion=target,
        )
        if pair is not None:
            pairs[sample_id] = pair
            if tie:
                tie_count += 1

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
    if pairs:
        print(
            f"Tie-break decided the frame for {tie_count}/{len(pairs)} samples "
            f"({tie_count/len(pairs)*100:.1f}%)")


if __name__ == "__main__":
    main()
