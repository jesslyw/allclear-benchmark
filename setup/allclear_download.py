# Wrapper around download.py (AllClear) that adds download options
# (metadata-only, custom dataset, specific ROI IDs) and rewrites the
# hardcoded server paths in the metadata JSONs to point at the local data/ folder.

import argparse
import multiprocessing as mp
import json
import shutil
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


def _iter_sample_paths(sample):
    for key, value in sample.items():
        if key == "roi" or not isinstance(value, list):
            continue
        for pair in value:
            if (
                isinstance(pair, list)
                and len(pair) >= 2
                and isinstance(pair[1], str)
            ):
                yield Path(pair[1])


def _required_roi_ids_from_dataset(dataset_fpaths):
    missing_or_incomplete = set()
    complete = set()

    for dataset_fpath in dataset_fpaths:
        with Path(dataset_fpath).open("r", encoding="utf-8") as f:
            dataset = json.load(f)

        for sample in dataset.values():
            roi_id = sample["roi"][0]
            paths = list(_iter_sample_paths(sample))
            if paths and all(path.exists() for path in paths):
                complete.add(roi_id)
            else:
                missing_or_incomplete.add(roi_id)

    return sorted(missing_or_incomplete - complete)


def _remove_incomplete_roi_folders(roi_ids):
    data_dir = Path("data")
    for roi_id in roi_ids:
        roi_dir = data_dir / roi_id
        if roi_dir.exists():
            shutil.rmtree(roi_dir)


def download_selected_rois(
    roi_ids=None,
    cpus=8,
    skip_metadata=False,
    dataset_fpaths=None,
    repair_incomplete=False,
):
    n_cores = max(1, cpus - 1)

    if skip_metadata:
        print("Skipping metadata download.")
    else:
        print("Downloading metadata files...")
        download_metadata()

    if dataset_fpaths:
        print("\nChecking required files from dataset metadata...")
        roi_ids = _required_roi_ids_from_dataset(dataset_fpaths)
        print(f"Found {len(roi_ids)} missing/incomplete ROI IDs")
        if repair_incomplete:
            print("Removing incomplete ROI folders before redownload...")
            _remove_incomplete_roi_folders(roi_ids)
    elif roi_ids:
        print(f"\nROI-filter mode enabled: {', '.join(roi_ids)}")
    else:
        print("\nLoading ROI IDs from metadata...")
        roi_ids = load_roi_list()
        print(f"Found {len(roi_ids)} unique ROI IDs")

    if not roi_ids:
        print("\nNothing to download.")
        return

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
    parser.add_argument(
        "--dataset-fpath",
        type=str,
        nargs="+",
        default=None,
        help="Download only ROIs with missing files in these dataset JSONs",
    )
    parser.add_argument(
        "--repair-incomplete",
        action="store_true",
        help="Remove incomplete ROI folders before redownloading them",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download and extract metadata only, skip ROI data download",
    )
    args = parser.parse_args()
    if args.metadata_only:
        from download import download_metadata
        download_metadata()
        rewrite_metadata_paths()
        return
    download_selected_rois(
        roi_ids=args.roi_id,
        cpus=args.cpus,
        skip_metadata=args.skip_metadata,
        dataset_fpaths=args.dataset_fpath,
        repair_incomplete=args.repair_incomplete,
    )


if __name__ == "__main__":
    main()
