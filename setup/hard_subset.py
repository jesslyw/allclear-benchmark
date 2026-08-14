"""Build a test set of scenes where no input frame is usable.

The main benchmark only has scenes with a clear frame available, because VPint2 needs
one. This picks the opposite: scenes where every S2 frame is almost fully clouded, so
there is nothing to copy. Cloud values come from AllClear's own metadata, so no images
need to be downloaded to build the list. Only scenes in emrdm_pairs.json are kept, which
also guarantees a usable S1 image. VPint2 cannot run on these.

Run: python3 setup/hard_subset.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

METADATA = "metadata/datasets/test_tx3_s2-s1_100pct_1proi.json"
S2_META = "metadata/data/s2_metadata.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emrdm-pairs-fpath", default="setup/emrdm_pairs.json")
    p.add_argument("--out-fpath", default="setup/hard_subset.json")
    p.add_argument("--min-cloud", type=float, default=90.0,
                   help="Every frame must be at least this cloudy (percent)")
    args = p.parse_args()

    full = json.loads(Path(METADATA).read_text())
    emrdm = json.loads(Path(args.emrdm_pairs_fpath).read_text())

    # cloud percentage per image, straight from AllClear's metadata
    meta = pd.read_csv(S2_META, usecols=["image_file_path", "cloud_percentage_30"])
    cloud = dict(zip(meta.image_file_path, meta.cloud_percentage_30))

    picked = {}
    for data_id, sample in full.items():
        if data_id not in emrdm:  # EMRDM needs a matching S1 image
            continue
        clouds = [cloud[path.removeprefix("data/")] for _, path in sample["s2_toa"]]
        if min(clouds) >= args.min_cloud:
            picked[data_id] = sample

    Path(args.out_fpath).write_text(json.dumps(picked, indent=1))
    print(f"Wrote {len(picked)} samples to {args.out_fpath}")

    rois = sorted({s["roi"][0] for s in picked.values()})
    Path("setup/hard_subset_rois.txt").write_text("\n".join(rois) + "\n")
    print(f"Wrote {len(rois)} ROI ids to setup/hard_subset_rois.txt (for download.py)")


if __name__ == "__main__":
    main()
