"""Create the final benchmark test set by keeping only samples present in both emrdm_pairs.json and vpint2_pairs.json (their intersection).
Fetches full metadata from the original AllClear test set using these indices.
Enriches with dominant land cover class from Dynamic World rasters (target frame).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

DEFAULT_OUT = Path(__file__).parent / "intersection_samples.json"


def get_dominant_lc_class(roi_id, target_date_str, data_root="data"):
    """Compute dominant land cover class from Dynamic World raster for target frame.

    Args:
        roi_id: ROI ID (e.g., 'roi12345')
        target_date_str: Target date string (e.g., '2022-03-18 00:13:39')
        data_root: Root directory for downloaded data

    Returns:
        Dominant class (1-9) or None if file not found or raster empty.
    """
    try:
        dt = datetime.strptime(target_date_str.split()[0], "%Y-%m-%d")
        dw_path = Path(data_root) / roi_id / f"{dt.year}_{dt.month}" / \
            "dw" / f"{roi_id}_dw_{dt.year}_{dt.month}_{dt.day}_median.tif"

        if not dw_path.exists():
            return None

        with rasterio.open(dw_path) as src:
            data = src.read(1).flatten()
            # Count non-zero classes (DW has classes 1-9, 0 is invalid)
            counts = np.bincount(data)
            if len(counts) > 1:
                # Return class with highest count (skipping 0)
                return int(np.argmax(counts[1:]) + 1)
        return None
    except (FileNotFoundError, ValueError, IndexError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create intersection of EMRDM and VPint2 eligible samples using filtering indices"
    )
    parser.add_argument(
        "--emrdm-pairs-fpath",
        required=True,
        help="Path to EMRDM pairs JSON (filtering index: which samples passed Phase 2)",
    )
    parser.add_argument(
        "--vpint2-pairs-fpath",
        required=True,
        help="Path to VPint2 pairs JSON (filtering index)",
    )
    parser.add_argument(
        "--metadata-json",
        default="metadata/datasets/test_tx3_s2-s1_100pct_1proi.json",
        help="Path to full AllClear metadata JSON (source of truth for all sample data)",
    )
    parser.add_argument(
        "--out-fpath",
        default=str(DEFAULT_OUT),
        help="Output path for intersection JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    emrdm_pairs_path = Path(args.emrdm_pairs_fpath)
    vpint2_pairs_path = Path(args.vpint2_pairs_fpath)
    metadata_path = Path(args.metadata_json)
    out_path = Path(args.out_fpath)

    if not emrdm_pairs_path.exists():
        raise FileNotFoundError(
            f"EMRDM pairs file not found: {emrdm_pairs_path}")
    if not vpint2_pairs_path.exists():
        raise FileNotFoundError(
            f"VPint2 pairs file not found: {vpint2_pairs_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load filtering indices (keys only)
    with open(emrdm_pairs_path) as f:
        emrdm_pairs = json.load(f)
    with open(vpint2_pairs_path) as f:
        vpint2_pairs = json.load(f)

    # Find intersection of sample IDs
    emrdm_ids = set(emrdm_pairs.keys())
    vpint2_ids = set(vpint2_pairs.keys())
    shared_ids = emrdm_ids & vpint2_ids

    # Load full metadata and extract shared samples
    with open(metadata_path) as f:
        full_metadata = json.load(f)

    intersection = {sid: full_metadata[sid]
                    for sid in shared_ids if sid in full_metadata}

    # Add dominant land cover class from target frame DW raster
    print("\nEnriching with land cover data...")
    enriched_count = 0
    for sid, sample in intersection.items():
        roi_id = sample["roi"][0]
        target_date = sample["target"][0][0]
        lc_class = get_dominant_lc_class(roi_id, target_date)
        if lc_class is not None:
            sample["dominant_lc_class"] = lc_class
            enriched_count += 1

    print(f"Enriched with land cover: {enriched_count}/{len(intersection)}")

    with open(out_path, "w") as f:
        json.dump(intersection, f, indent=2)

    print(f"EMRDM eligible      : {len(emrdm_ids)}")
    print(f"VPint2 eligible     : {len(vpint2_ids)}")
    print(f"Intersection samples: {len(intersection)}")
    print(f"EMRDM pairs source  : {emrdm_pairs_path}")
    print(f"VPint2 pairs source : {vpint2_pairs_path}")
    print(f"Metadata source     : {metadata_path}")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
