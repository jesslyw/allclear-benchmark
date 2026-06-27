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


def string_to_datetime(value):
    return datetime.strptime(value, TIME_FORMAT)


def frame_occlusion(s2_fpath):
    """Fraction of pixels occluded by cloud OR shadow for one S2 frame.

    Reads the frame's cld_shdw mask and returns the union (cloud | shadow)
    fraction in [0, 1], matching benchmark.py's valid-pixel definition
    (`~((cloud + shadow) > 0)`); the two masks are unioned per pixel, not added
    as separate percentages. NaN pixels are treated as occluded. Returns None if
    the mask file is not available locally.
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

    We use this in --screen mode so we don't have to open the mask files.
    We can't know the real cloud+shadow union from the two numbers, but it
    has to be somewhere between max(cloud, shadow) and (cloud + shadow).
    Rows with missing or negative values are just skipped.
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


def screen_vpint2(sample, csv_bounds, max_ref_cloud=0.10):
    """Quick check if a sample could be VPint2-eligible, using only the CSV.

    We only look at the cloud side here: is there a cloudy frame (some clouds
    but not fully covered) and a clear-enough reference frame (union below
    max_ref_cloud)? We skip the date/order rules on purpose, so this keeps a
    few extra samples but never drops a real one.

    Returns "skip" / "eligible" / "uncertain".
    """
    roi = sample["roi"][0]
    cloudy_accept = cloudy_straddle = False
    ref_accept = ref_straddle = False
    for ts, _ in sample.get("s2_toa", []):
        v = csv_bounds.get((roi, ts))
        if v is None:
            continue
        floor = max(v[0], v[1])
        ceil = min(v[0] + v[1], 1.0)
        # cloudy frame: needs some clouds but not 100%
        if ceil <= 0 or floor >= 1:
            pass  # fully clear or fully covered, can't be the cloudy frame
        elif floor > 0 and ceil < 1:
            cloudy_accept = True
        else:
            cloudy_straddle = True
        # reference frame: clear enough (union <= max_ref_cloud)
        if ceil <= max_ref_cloud:
            ref_accept = True
        elif floor > max_ref_cloud:
            pass  # too cloudy to be a reference
        else:
            ref_straddle = True
    cloudy_ok = cloudy_accept or cloudy_straddle
    ref_ok = ref_accept or ref_straddle
    if not (cloudy_ok and ref_ok):
        return "skip"
    if cloudy_accept and ref_accept:
        return "eligible"
    return "uncertain"


def find_vpint2_pair(
    sample,
    max_ref_cloud=0.10,
    max_target_gap_days=5,
):
    target_date = sample["target"][0][0]
    target_dt = string_to_datetime(target_date)

    inputs = sample["s2_toa"]

    best = None

    for cloudy_idx, cloudy_input in enumerate(inputs):
        cloudy_date, cloudy_fpath = cloudy_input[0], cloudy_input[1]
        cloudy_dt = string_to_datetime(cloudy_date)

        # occlusion (cloud | shadow) from the frame's mask
        cloudy_cloud = frame_occlusion(cloudy_fpath)
        if cloudy_cloud is None:
            continue

        # Cloudy image must contain clouds/shadows, but not be fully covered.
        if cloudy_cloud <= 0 or cloudy_cloud >= 1:
            continue

        # delta days from target within range
        cloudy_target_delta = abs((target_dt - cloudy_dt).days)
        if cloudy_target_delta > max_target_gap_days:
            continue

        refs = []

        for ref_idx, ref_input in enumerate(inputs):
            ref_date, ref_fpath = ref_input[0], ref_input[1]
            ref_dt = string_to_datetime(ref_date)

            if ref_dt >= cloudy_dt:  # t before cloudy
                continue

            # occlusion (cloud | shadow) from the frame's mask
            ref_cloud = frame_occlusion(ref_fpath)
            if ref_cloud is None:
                continue

            # occlusion below threshold (simulating clear image)
            if ref_cloud <= max_ref_cloud:
                refs.append((ref_idx, ref_dt, ref_cloud))

        if not refs:
            continue

        ref_idx, ref_dt, ref_cloud = min(
            refs,
            key=lambda x: (
                x[2],  # pick reference image with lowest occlusion
                # if tie, pick reference closest in time to cloudy image
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

        current_score = (
            candidate["cloudy_target_delta_days"],
            candidate["reference_occlusion_pct"],
            -candidate["cloudy_occlusion_pct"],
        )

        best_score = (
            best["cloudy_target_delta_days"],
            best["reference_occlusion_pct"],
            -best["cloudy_occlusion_pct"],
        )

        # Keep the best candidate for this ROI:
        # prioritize smaller target-date gap, clearer reference image,
        # and higher occlusion in the cloudy image.
        if current_score < best_score:
            best = candidate

    return best


def main():
    parser = argparse.ArgumentParser(
        description="Filter AllClear JSON to VPint2-compatible samples"
    )

    parser.add_argument(
        "--metadata-json",
        default="metadata/datasets/test_tx3_s2-s1_100pct_1proi.json",
        help="Path to AllClear test metadata JSON",
    )

    parser.add_argument(
        "--pairs-out",
        default="setup/vpint2_pairs.json",
        help="Output path for VPint2 pairs metadata JSON",
    )

    parser.add_argument(
        "--data-out",
        default="setup/vpint2_samples.json",
        help="Output path for VPint2 samples JSON (AllClear format, eligible samples only)",
    )

    parser.add_argument(
        "--max-ref-cloud",
        type=float,
        default=0.10,
        help="Maximum cloud/shadow percentage for reference input",
    )

    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=5,
        help="Maximum temporal gap between cloudy input and target",
    )

    parser.add_argument(
        "--screen",
        action="store_true",
        help="CSV-only pre-screen: write the candidate ROI download list using "
             "cloud/shadow numbers, without opening any mask files.",
    )

    parser.add_argument(
        "--csv",
        default="metadata/data/s2_metadata.csv",
        help="Path to s2_metadata.csv (only used with --screen)",
    )

    parser.add_argument(
        "--candidates-out",
        default="setup/vpint2_candidates.txt",
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
            status = screen_vpint2(sample, csv_bounds, args.max_ref_cloud)
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

    subset = {}

    for sample_id, sample in tqdm(dataset.items()):
        if not sample.get("s2_toa") or not sample.get("target"):
            continue

        pair = find_vpint2_pair(
            sample,
            max_ref_cloud=args.max_ref_cloud,
            max_target_gap_days=args.max_gap_days,
        )

        if pair is not None:
            subset[sample_id] = pair

    missing = sum(1 for v in _occlusion_cache.values() if v is None)
    if missing:
        print(
            f"WARNING: {missing} S2 frames had no local cld_shdw mask and were skipped.")

    Path(args.pairs_out).write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(subset)} VPint2-compatible samples to {args.pairs_out}")

    dataset_subset = {sid: dataset[sid] for sid in subset}
    Path(args.data_out).write_text(
        json.dumps(dataset_subset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved filtered dataset to {args.data_out}")


if __name__ == "__main__":
    main()
