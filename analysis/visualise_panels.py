"""
Builds one qualitative panel image per selected sample: 5 model predictions
side by side (LeastCloudy, Mosaicing, VPint2, UnCRtainTS, EMRDM), so each ROI
gets its own small comparison image instead of one giant combined grid.

No need to rerun any model for this — we build it straight from the saved
outputs/AllClear/intersection_samples/*_predictions.pt and *_metadata.csv
files. Row i of a model's predictions.pt is always the same sample as row i
of its metadata.csv (benchmark.py writes both together, per batch), so we
just look up the row and grab the image.

Note: every model's saved prediction is a single reconstructed image (shape
13, 1, H, W), even though models differ in how many input S2 frames they
consume internally (LeastCloudy uses 1 frame, others use multi-temporal
stacks) — that's an input-side detail that doesn't show up in the output
shape, since the task is always "produce one cloud-free image."

Predictions only, no ground-truth/cloudy-input column, since those aren't
saved locally (see .agents/task1-artifact-plots-plan.md, Resolved Decisions #4).

Reads which samples to render from outputs/thesis/tables/qualitative_selection.csv
(made by the notebook). Run this after the notebook:
    .venv/bin/python3 analysis/visualise_panels.py

It also builds the Section 4.5 figure, which is a different shape: one scene where no input
frame is usable, LeastCloudy above EMRDM, each as input / prediction / target. That one cannot
come from the prediction tensors, because they hold no target. It is assembled instead from
benchmark.py's own visualisation PNGs, so its RGB stretch is benchmark.py's rather than the one
above -- the two rows are comparable with each other, not with the panels. Produce the inputs on
the host with, for each of the two models:

    python3 benchmark.py --model-name <model> \
      --dataset-fpath setup/hard_subset.json --selected-rois roi29225 \
      --aux-sensors s1 --batch-size 1 --draw-vis 1 --max-vis-samples 1 \
      --experiment-output-path outputs_hard_vis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "outputs" / "AllClear" / "intersection_samples"
SELECTION_CSV = ROOT / "outputs" / "thesis" / "tables" / "qualitative_selection.csv"
OUT_DIR = ROOT / "outputs" / "thesis" / "visualisations"

# Section 4.5 figure: assembled from benchmark.py's visualisation PNGs, not the tensors.
# Section 4.3 scene: benchmark.py renders raw reflectance, which prints near-black. The panels
# above divide by _RGB_VMAX for the same reason, so the same gain is applied here to make the two
# figures comparable on the page. Only the display is scaled; the metrics come from the tensors.
INDEX_GAP_VIS = (ROOT / "outputs_panel_vis" / "AllClear" / "intersection_samples"
                 / "UnCRtainTS_vis" / "roi533648_2022-03-29_2022-04-13.png")

HARD_VIS = ROOT / "outputs_hard_vis" / "AllClear" / "hard_subset"
HARD_SCENE = "roi29225_2022-10-17_2022-11-01.png"
# chosen from the hard-set metrics: input frame 100% clouded, EMRDM's MAE mid-range for the set
# so the scene is typical, and the largest gap to LeastCloudy among those candidates
_LABEL_H, _TITLE_H = 46, 44

MODELS = ["LeastCloudy", "Mosaicing", "VPint2", "UnCRtainTS", "EMRDM"]
_R, _G, _B = 3, 2, 1  # same RGB bands visualise.py uses

# S2 TOA reflectance RGB bands are legitimately low (95th percentile ~0.28,
# checked across LeastCloudy's predictions) — a plain clip(0, 1) renders
# everything near-black. Same fixed vmax applied to every image, so
# brightness differences reflect real scene differences, not a per-image
# auto-stretch.
_RGB_VMAX = 0.3


def to_rgb(band_stack: torch.Tensor):
    """One sample's prediction (13, 1, H, W) -> a plain RGB image (H, W, 3), stretched to [0, 1]."""
    rgb = band_stack[[_R, _G, _B], 0].permute(1, 2, 0)
    return (rgb / _RGB_VMAX).clip(0, 1).numpy()


def load_model_index(model: str):
    metadata = pd.read_csv(DATA_ROOT / f"{model}_metadata.csv")
    predictions = torch.load(DATA_ROOT / f"{model}_predictions.pt", map_location="cpu", weights_only=False)
    row_of = {data_id: i for i, data_id in enumerate(metadata["data_id"])}
    return row_of, predictions


def render_panel(data_id: str, model_indices: dict) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(2.5 * len(MODELS), 2.8), dpi=150)
    for ax, model in zip(axes, MODELS):
        row_of, predictions = model_indices[model]
        ax.axis("off")
        ax.set_title(model, fontsize=9)
        if data_id not in row_of:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.imshow(to_rgb(predictions[row_of[data_id]]))

    fig.suptitle(data_id, fontsize=10)
    fig.tight_layout()
    # PDF only -- that's what gets embedded in the LaTeX thesis.
    fig.savefig(OUT_DIR / f"{data_id}_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def render_index_gap_scene() -> bool:
    """Section 4.3: brighten benchmark.py's own panel to the stretch used for the appendix panels."""
    if not INDEX_GAP_VIS.exists():
        return False
    import numpy as np

    im = np.asarray(Image.open(INDEX_GAP_VIS).convert("RGB")).astype(np.float32)
    page = im.mean(axis=2) > 240                      # leave the white background alone
    out = np.clip(im / _RGB_VMAX, 0, 255)
    out[page] = im[page]
    Image.fromarray(out.astype("uint8")).save(
        OUT_DIR / "roi533648_uncrtaints_inputs_target.png")
    return True


def render_hard_scene() -> bool:
    """Section 4.5: two rows, LeastCloudy above EMRDM, each input / prediction / target."""
    if not (HARD_VIS / "EMRDM_vis" / HARD_SCENE).exists():
        return False

    def columns(image, n):
        w = image.width // n
        return [image.crop((i * w, _TITLE_H, (i + 1) * w, image.height)) for i in range(n)]

    # LeastCloudy and UnCRtainTS write 3 inputs + prediction + target; EMRDM writes 1 input +
    # prediction + target. Keep one input, the prediction and the target from each.
    least = columns(Image.open(HARD_VIS / "LeastCloudy_vis" / HARD_SCENE), 5)
    uncr = columns(Image.open(HARD_VIS / "UnCRtainTS_vis" / HARD_SCENE), 5)
    emrdm = columns(Image.open(HARD_VIS / "EMRDM_vis" / HARD_SCENE), 3)
    rows = [("LeastCloudy", [least[0], least[3], least[4]]),
            ("UnCRtainTS", [uncr[0], uncr[3], uncr[4]]),
            ("EMRDM", emrdm)]

    w, h = rows[0][1][0].size
    canvas = Image.new("RGB", (w * 3, (h + _LABEL_H) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except OSError:
        font = ImageFont.load_default()
    for r, (name, panels) in enumerate(rows):
        top = r * (h + _LABEL_H)
        draw.text((10, top + 8), name, fill="black", font=font)
        for c, panel in enumerate(panels):
            canvas.paste(panel, (c * w, top + _LABEL_H))

    canvas.save(OUT_DIR / "roi29225_hard_scene.png")
    return True


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(SELECTION_CSV)
    data_ids = selection["data_id"].tolist()

    model_indices = {model: load_model_index(model) for model in MODELS}

    for data_id in data_ids:
        render_panel(data_id, model_indices)

    print(f"Wrote {len(data_ids)} panel images (one per sample, 5 models each) to {OUT_DIR}")

    if render_index_gap_scene():
        print(f"Wrote the Section 4.3 index-gap scene to {OUT_DIR}")
    else:
        print(f"Skipped the Section 4.3 scene: {INDEX_GAP_VIS} not present")

    if render_hard_scene():
        print(f"Wrote the Section 4.5 hard-scene figure to {OUT_DIR}")
    else:
        print(f"Skipped the Section 4.5 figure: {HARD_VIS} not populated (see the docstring)")


if __name__ == "__main__":
    main()
