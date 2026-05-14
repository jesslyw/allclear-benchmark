"""
Visualise VPint2 prediction vs ground truth for one sample.
Saves <data_id>.png inside pred_vs_target/.
Run: python visualise.py --roi <roi>
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import model_wrappers as wrappers
from dataset import AllClearDataset

_R, _G, _B = 3, 2, 1


def to_rgb(t):
    return np.clip(t[[_R, _G, _B], 0].permute(1, 2, 0).cpu().numpy(), 0, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-fpath", default="setup/vpint2_dataset.json")
    p.add_argument("--vpint2-pairs-fpath", default="setup/vpint2_pairs.json")
    p.add_argument("--roi", required=True, help="e.g. roi1008997")
    return p.parse_args()


def main():
    args = parse_args()
    args.device = "cpu"
    args.aux_sensors = []
    args.aux_data = ["cld_shdw", "dw"]
    args.main_sensor = "s2_toa"
    args.target_mode = "s2p"
    args.tx = 3
    args.batch_size = 1
    args.num_workers = 0
    args.model_name = "VPint2"

    dataset_json = json.loads(Path(args.dataset_fpath).read_text())
    dataset = AllClearDataset(
        dataset=dataset_json,
        selected_rois=[args.roi],
        main_sensor=args.main_sensor,
        aux_sensors=args.aux_sensors,
        aux_data=args.aux_data,
        tx=args.tx,
        target_mode=args.target_mode,
    )
    model = wrappers.VPint2(args)

    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        try:
            prepped = model.preprocess(batch)
        except FileNotFoundError:
            continue
        if prepped.get("skip_batch"):
            continue
        pred = model.forward(prepped)["output"]
        if pred is None:
            continue

        data_id = batch["data_id"][0]
        cloudy_t = prepped["vpint_batch"][0]["t_cloudy"]
        cloudy = batch["input_images"][0, :13, cloudy_t].unsqueeze(1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(axes,
                                  [cloudy, pred[0], batch["target"][0]],
                                  ["Cloudy input", "VPint2 prediction", "Ground truth"]):
            ax.imshow(to_rgb(img))
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(data_id, fontsize=9)
        plt.tight_layout()
        out = Path("pred_vs_target") / f"{data_id}.png"
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
        return

    print("No eligible sample found.")


if __name__ == "__main__":
    main()
