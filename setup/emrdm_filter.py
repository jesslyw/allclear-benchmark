"""
Keep only the AllClear test samples that EMRDM can run on.

A sample is kept if:
  1. It has at least one S1 image.
  2. It has at least one S2 frame that is 20-70% cloud/shadow.
  3. We then pick the S2 frame closest in time to an S1 image.

How we measure cloud/shadow: each frame has a mask with a cloud band and a
shadow band. A pixel counts as covered if it is cloud OR shadow (band 2 or
band 5). We count those pixels and divide by all pixels. This is the same
rule benchmark.py uses when scoring, so our numbers match. Missing (NaN)
pixels count as covered. Needs the mask files to be downloaded first.

Outputs:
  emrdm_pairs.json   - the chosen frames per sample
  emrdm_samples.json - the kept samples in AllClear format

Usage:
  python setup/emrdm_filter.py \
      --metadata-json metadata/datasets/test_tx3_s2-s1_100pct_1proi.json
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

# Mask bands (1-indexed, as read by rasterio) — see dataset.py channels["cld_shdw"].
CLOUD_BAND = 2
SHADOW_BAND = 5

_occlusion_cache = {}


def parse_dt(value):
    return datetime.strptime(value, TIME_FORMAT)


def frame_occlusion(s2_fpath):
    """How much of one S2 frame is covered by cloud or shadow (0 to 1).

    Opens the frame's mask and counts pixels that are cloud OR shadow.
    NaN pixels count as covered. Returns None if the mask isn't downloaded.
    """
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
    """Read s2_metadata.csv and give back the cloud/shadow % for each frame.

    Used by --screen mode so we don't have to open the mask files. We can't
    get the exact covered % from the two numbers, but it has to be between
    max(cloud, shadow) and (cloud + shadow). Bad rows are skipped.
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


def screen_emrdm(sample, csv_bounds, min_cloud=0.20, max_cloud=0.70):
    """Guess if a sample could be EMRDM-eligible from the CSV alone.

      "skip"      - no chance (no S1, or no frame can land in the window)
      "eligible"  - at least one frame is surely inside the window
      "uncertain" - a frame might be inside, need the mask to be sure
    """
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
            continue  # this frame can't be in the window
        if floor >= min_cloud and ceil <= max_cloud:
            has_accept = True
        else:
            has_straddle = True
    if has_accept:
        return "eligible"
    if has_straddle:
        return "uncertain"
    return "skip"


def find_emrdm_pair(sample, min_cloud=0.20, max_cloud=0.70):
    """Pick the best (S2, S1) pair for a sample, or None if none fits.

    Look at every S2 frame in the 20-70% window, match it to the nearest S1
    in time, and keep the pair with the smallest time gap. Ties go to the
    frame with more cloud/shadow.
    """
    s2_entries = sample.get("s2_toa", [])
    s1_entries = sample.get("s1", [])

    if not s1_entries:
        return None

    s1_dts = [parse_dt(ts) for ts, _ in s1_entries]

    best = None

    for s2_idx, (s2_ts, s2_fpath) in enumerate(s2_entries):
        occlusion = frame_occlusion(s2_fpath)
        if occlusion is None:
            continue
        if not (min_cloud <= occlusion <= max_cloud):
            continue

        s2_dt = parse_dt(s2_ts)

        # nearest S1 in time to this S2 frame
        s1_gaps = [abs((s2_dt - s1_dt).days) for s1_dt in s1_dts]
        best_s1_idx = min(range(len(s1_gaps)), key=lambda i: s1_gaps[i])
        delta_days = s1_gaps[best_s1_idx]

        candidate = {
            "s2_index": s2_idx,
            "s1_index": best_s1_idx,
            "s2_occlusion_pct": round(occlusion, 4),
            "s2_s1_delta_days": delta_days,
        }

        if best is None:
            best = candidate
            continue

        # smaller time gap wins, then more cloud/shadow
        def score(c):
            return (c["s2_s1_delta_days"], -c["s2_occlusion_pct"])

        if score(candidate) < score(best):
            best = candidate

    return best


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

    pairs = {}
    for sample_id, sample in tqdm(dataset.items(), desc="Filtering"):
        if not sample.get("s2_toa") or not sample.get("target"):
            continue
        pair = find_emrdm_pair(
            sample,
            min_cloud=args.min_cloud,
            max_cloud=args.max_cloud,
        )
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
