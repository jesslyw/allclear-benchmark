# allclear-benchmark

Benchmark for evaluating cloud removal models on the [AllClear](https://github.com/Zhou-Hangyu/allclear) dataset.
Implements VPint2 as a baseline, evaluated on a filtered subset of the AllClear test set.

## Requirements

Python 3.10. Using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Setup

Filters the AllClear test set to a subset compatible with VPint2's input requirements. See docs/methodology.md for the filtering criteria.

Run from the repo root:

```bash
bash setup/setup.sh
```

This will:

1. Download AllClear metadata (dataset JSONs + cloud/shadow CSVs)
2. Filter the test set to VPint2-eligible samples → `setup/vpint2_pairs.json`, `setup/vpint2_dataset.json`
3. Download the image data (TIFFs) for the eligible ROIs

Thresholds used for filtering can be configured at the top of `setup/setup.sh`.

## Run benchmark

```bash
python benchmark.py --model-name VPint2 --batch-size 1
```

## Visualise predictions

```bash
python visualise.py --roi <roi>
```

Output saved to `pred_vs_target/<roi_id>.png`.

Example output: ![Example visualisation](pred_vs_target/roi793494_2022-08-04_2022-08-11.png)

## Visualise all ROIs on a map

A small helper script is provided to generate a GeoJSON file from the filtered subset, which can be visualised using a map tool like https://geojson.io/next.

```bash
python setup/make_geojson.py
```

## Attribution

- Dataset and data loading (`dataset.py`, `download.py`) based on [AllClear](https://github.com/Zhou-Hangyu/allclear) (MIT License)
- VPint2 model: [VPint2](https://github.com/ADA-research/VPint2) (submodule under `models/VPint2`)
