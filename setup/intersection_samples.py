"""Create the final benchmark test set by keeping only samples present in both emrdm_samples.json and vpint2_samples.json files (their intersection).
"""

import argparse
import json
from pathlib import Path

DEFAULT_OUT = Path(__file__).parent / "intersection_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create intersection of EMRDM and VPint2 sample JSON files"
    )
    parser.add_argument(
        "--emrdm-samples-fpath",
        required=True,
        help="Path to EMRDM samples JSON (source records preserved in output)",
    )
    parser.add_argument(
        "--vpint2-samples-fpath",
        required=True,
        help="Path to VPint2 samples JSON",
    )
    parser.add_argument(
        "--out-fpath",
        default=str(DEFAULT_OUT),
        help="Output path for intersection JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    emrdm_path = Path(args.emrdm_samples_fpath)
    vpint2_path = Path(args.vpint2_samples_fpath)
    out_path = Path(args.out_fpath)

    if not emrdm_path.exists():
        raise FileNotFoundError(f"EMRDM samples file not found: {emrdm_path}")
    if not vpint2_path.exists():
        raise FileNotFoundError(
            f"VPint2 samples file not found: {vpint2_path}")

    with open(emrdm_path) as f:
        emrdm = json.load(f)
    with open(vpint2_path) as f:
        vpint2 = json.load(f)

    shared = [k for k in emrdm if k in vpint2]
    out = {k: emrdm[k] for k in shared}

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"EMRDM samples       : {len(emrdm)}")
    print(f"VPint2 samples      : {len(vpint2)}")
    print(f"Intersection samples: {len(out)}")
    print(f"EMRDM source        : {emrdm_path}")
    print(f"VPint2 source       : {vpint2_path}")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
