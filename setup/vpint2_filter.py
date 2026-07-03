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

_occlusion_cache = {}


def string_to_datetime(value):
    return datetime.strptime(value, TIME_FORMAT)


def frame_occlusion(s2_fpath):
    """Return how much of one S2 frame is covered by cloud or shadow.

    We read the matching cld_shdw mask and mark a pixel as occluded if cloud
    or shadow is present. The result is a fraction between 0 and 1. NaN values
    are treated as occluded pixels. If the mask file is missing locally,
    returns None.
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


def screen_vpint2(sample, csv_bounds):
    """CSV-only pre-download check.

    Labels a sample as "skip", "eligible", or "uncertain" based on whether a
    likely clear reference (<10% cloud/shadow) can occur before a cloudy frame.
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

        # reference frame should be likely clear enough
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


def find_vpint2_pair(sample):
    target_date = sample["target"][0][0]
    target_dt = string_to_datetime(target_date)

    inputs = sample["s2_toa"]

    best = None

    for cloudy_idx, cloudy_input in enumerate(inputs):
        cloudy_date, cloudy_fpath = cloudy_input[0], cloudy_input[1]
        cloudy_dt = string_to_datetime(cloudy_date)

        cloudy_cloud = frame_occlusion(cloudy_fpath)
        if cloudy_cloud is None:
            continue

        # cloudy frame must be partly cloudy
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
        "--screen",
        action="store_true",
        help="CSV-only pre-screen: write the candidate ROI download list using "
             "cloud/shadow numbers, without opening any mask files.",
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
        csv_bounds = load_csv_bounds(CSV_BOUNDS_PATH)
        eligible = set()
        uncertain = set()
        for sample in dataset.values():
            if not sample.get("s2_toa") or not sample.get("target"):
                continue
            status = screen_vpint2(sample, csv_bounds)
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

        pair = find_vpint2_pair(sample)

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
