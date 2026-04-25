import argparse
import multiprocessing as mp
import json
from pathlib import Path

from download import download_metadata, load_roi_list, download_roi_worker


def rewrite_metadata_paths():
    OLD_PREFIX = "/scratch/allclear/dataset_v3/dataset_30k_v4"
    NEW_PREFIX = "data"
    datasets_dir = Path("./metadata/datasets")
    json_files = sorted(datasets_dir.glob("*.json"))

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            ds = json.load(f)

        changed = 0
        for _, sample in ds.items():
            for key, value in sample.items():
                if key == "roi":
                    continue
                # value is expected like: [[timestamp, path], [timestamp, path], ...]
                if not isinstance(value, list):
                    continue

                for pair in value:
                    if (
                        isinstance(pair, list)
                        and len(pair) >= 2
                        and isinstance(pair[1], str)
                        and pair[1].startswith(OLD_PREFIX)
                    ):
                        pair[1] = pair[1].replace(OLD_PREFIX, NEW_PREFIX, 1)
                        changed += 1

        if changed:
            with jf.open("w", encoding="utf-8") as f:
                json.dump(ds, f, indent=2)
            print(f"{jf.name}: updated {changed} paths")
        else:
            print(f"{jf.name}: no changes")


def download_selected_rois(roi_ids=None, cpus=8, skip_metadata=False):
    n_cores = max(1, cpus - 1)

    if skip_metadata:
        print("Skipping metadata download.")
    else:
        print("Downloading metadata files...")
        download_metadata()

    if roi_ids:
        print(f"\nROI-filter mode enabled: {', '.join(roi_ids)}")
    else:
        print("\nLoading ROI IDs from metadata...")
        roi_ids = load_roi_list()
        print(f"Found {len(roi_ids)} unique ROI IDs")

    chunk_size = len(roi_ids) // n_cores + 1
    roi_chunks = [roi_ids[i:i + chunk_size]
                  for i in range(0, len(roi_ids), chunk_size)]

    print(f"\nDownloading ROIs using {n_cores} processes...")
    with mp.Pool(n_cores) as pool:
        pool.map(download_roi_worker, roi_chunks)

    print("\nDownload completed!")


def main():
    parser = argparse.ArgumentParser(description="AllClear downloader wrapper")
    parser.add_argument("--cpus", type=int, default=8,
                        help="Number of CPU cores to use (default: 8)")
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip metadata download step",
    )
    parser.add_argument(
        "--roi-id",
        type=str,
        nargs="+",
        default=None,
        help="Download one or more ROIs (e.g., --roi-id roi801784 roi123456)",
    )
    args = parser.parse_args()
    download_selected_rois(
        roi_ids=args.roi_id,
        cpus=args.cpus,
        skip_metadata=args.skip_metadata,
    )


if __name__ == "__main__":
    main()
