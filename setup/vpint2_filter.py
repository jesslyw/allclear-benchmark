import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def string_to_datetime(value):
    return datetime.strptime(value, TIME_FORMAT)


def cloud_shadow_percentage(row):
    cloud = row["cloud_percentage_30"]
    shadow = row["shadow_percentage_30"]

    if cloud == '' or shadow == '':
        return None

    cloud = float(cloud)
    shadow = float(shadow)
    return min(cloud + shadow, 100.0) / 100.0


def load_cloud_metadata(csv_path):
    """
    Create lookup:
    (roi, capture_date) -> cloud/shadow percentage
    """
    cloud_lookup = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # if row["satellite"] != "s2_toa":
            #    continue

            key = (row["roi"], row["capture_date"])
            cloud_lookup[key] = cloud_shadow_percentage(row)

    return cloud_lookup


def get_cloud(cloud_lookup, roi, date):
    key = (roi, date)
    return cloud_lookup.get(key)


def find_vpint2_pair(
    sample,
    cloud_lookup,
    max_ref_cloud=0.10,
    max_target_gap_days=5,
):
    roi = sample["roi"][0]
    target_date = sample["target"][0][0]
    target_dt = string_to_datetime(target_date)

    inputs = sample["s2_toa"]

    best = None

    for cloudy_idx, cloudy_input in enumerate(inputs):
        cloudy_date = cloudy_input[0]
        cloudy_dt = string_to_datetime(cloudy_date)

        # has cloud/shadow percentage
        cloudy_cloud = get_cloud(cloud_lookup, roi, cloudy_date)
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
            ref_date = ref_input[0]
            ref_dt = string_to_datetime(ref_date)

            if ref_dt >= cloudy_dt:  # t before cloudy
                continue

            # has cloud/shadow percentage
            ref_cloud = get_cloud(cloud_lookup, roi, ref_date)
            if ref_cloud is None:
                continue

            # cloud percentage below threshold (simulating clear image)
            if ref_cloud <= max_ref_cloud:
                refs.append((ref_idx, ref_dt, ref_cloud))

        if not refs:
            continue

        ref_idx, ref_dt, ref_cloud = min(
            refs,
            key=lambda x: (
                x[2],  # pick reference image with lowest cloud percentage
                # if tie, pick reference closest in time to cloudy image
                abs((cloudy_dt - x[1]).days),
            ),
        )

        candidate = {
            "cloudy_index": cloudy_idx,
            "reference_index": ref_idx,
            "cloudy_cloud_shadow_pct": round(cloudy_cloud, 4),
            "reference_cloud_shadow_pct": round(ref_cloud, 4),
            "cloudy_reference_delta_days": abs((cloudy_dt - ref_dt).days),
            "cloudy_target_delta_days": cloudy_target_delta,
        }

        if best is None:
            best = candidate
            continue

        current_score = (
            candidate["cloudy_target_delta_days"],
            candidate["reference_cloud_shadow_pct"],
            -candidate["cloudy_cloud_shadow_pct"],
        )

        best_score = (
            best["cloudy_target_delta_days"],
            best["reference_cloud_shadow_pct"],
            -best["cloudy_cloud_shadow_pct"],
        )

        # Keep the best candidate for this ROI:
        # prioritize smaller target-date gap, clearer reference image,
        # and higher cloud coverage in the cloudy image.
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
        "--csv",
        default="metadata/data/s2_metadata.csv",
        help="Path to s2_metadata.csv (AllClear s2 metadata used to derive cloud/shadow percentages per image)",
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

    args = parser.parse_args()

    with open(args.metadata_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    cloud_lookup = load_cloud_metadata(args.csv)

    subset = {}

    for sample_id, sample in tqdm(dataset.items()):
        if not sample.get("s2_toa") or not sample.get("target"):
            continue

        pair = find_vpint2_pair(
            sample,
            cloud_lookup,
            max_ref_cloud=args.max_ref_cloud,
            max_target_gap_days=args.max_gap_days,
        )

        if pair is not None:
            subset[sample_id] = pair

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
