"""
Benchmark runner for AllClear-style datasets.

Based on: https://github.com/Zhou-Hangyu/allclear (allclear/benchmark.py)
License: MIT
"""

import argparse
import json
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import model_wrappers as wrappers
from dataset import AllClearDataset
from metrics import compute_batch_metrics


@dataclass
class MetricTotals:
    mae_sum: float = 0.0
    rmse_sum: float = 0.0
    psnr_sum: float = 0.0
    sam_sum: float = 0.0
    ssim_sum: float = 0.0
    ndvi_mae_sum: float = 0.0
    nbr_mae_sum: float = 0.0
    count: int = 0


def _to_bcthw(x: torch.Tensor, target_c: int) -> torch.Tensor:
    """Normalize tensor layout to (B, C, T, H, W)."""
    if x.dim() == 4:  # (B, C, H, W)
        return x.unsqueeze(2)
    if x.dim() != 5:
        raise ValueError(f"Expected 4D/5D tensor, got shape {tuple(x.shape)}")

    if x.shape[1] == target_c:  # already (B, C, T, H, W)
        return x
    if x.shape[2] == target_c:  # likely (B, T, C, H, W)
        return x.permute(0, 2, 1, 3, 4)

    raise ValueError(
        f"Cannot infer output layout from shape {tuple(x.shape)} with target C={target_c}"
    )


def _valid_mask(batch: dict, target: torch.Tensor) -> torch.Tensor:
    """Build valid-pixel mask from target cloud/shadow mask. Returns (B, 1, T, H, W), True = valid."""
    if "target_cld_shdw" not in batch:
        # assume all pixels valid
        return torch.ones(
            (target.shape[0], 1, target.shape[2],
             target.shape[3], target.shape[4]),
            dtype=torch.bool, device=target.device,
        )
    # normalise mask to shape of target
    cld = _to_bcthw(batch["target_cld_shdw"], target_c=2)
    return (~((cld[:, 0] + cld[:, 1]) > 0)).unsqueeze(1)


class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = self._setup_model()
        self.data_loader = self._setup_data_loader()

    def _setup_model(self):
        name = self.args.model_name.lower()
        if name == "vpint2":
            return wrappers.VPint2(self.args)
        raise ValueError(
            f"Unknown model '{self.args.model_name}'. Available: VPint2")

    def _setup_data_loader(self):
        with open(self.args.dataset_fpath, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)
        selected_rois = (
            self.args.selected_rois
            if self.args.selected_rois and "all" not in self.args.selected_rois
            else "all"
        )
        dataset = AllClearDataset(
            dataset=dataset_json,
            selected_rois=selected_rois,
            main_sensor=self.args.main_sensor,
            aux_sensors=self.args.aux_sensors,
            aux_data=self.args.aux_data,
            tx=self.args.tx,
            target_mode=self.args.target_mode,
        )
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def run(self):
        totals = MetricTotals()

        # wrapper to guard against missing files
        def _safe_iter(loader):
            it = iter(loader)
            while True:
                try:
                    yield next(it)
                except StopIteration:
                    return
                except FileNotFoundError as e:
                    print(f"[SKIP] Missing file: {e}")

        for batch in tqdm(_safe_iter(self.data_loader), total=len(self.data_loader), desc="Benchmark"):
            with torch.no_grad():
                prepped = self.model.preprocess(batch)
                if prepped.get("skip_batch"):
                    # not in vpint2_pairs.json
                    continue
                pred = self.model.forward(prepped)["output"]
                # no model output
                if pred is None:
                    continue

                target = batch["target"].to(self.device)
                target_c = target.shape[1]
                # align tensors to (B C T H W)
                pred = _to_bcthw(pred.to(self.device), target_c=target_c)
                target = _to_bcthw(target, target_c=target_c)

                mae, rmse, psnr, sam, ssim, ndvi_mae, nbr_mae, inc = compute_batch_metrics(
                    pred, target, _valid_mask(batch, target))
                totals.mae_sum += mae
                totals.rmse_sum += rmse
                totals.psnr_sum += psnr
                totals.sam_sum += sam
                totals.ssim_sum += ssim
                totals.ndvi_mae_sum += ndvi_mae
                totals.nbr_mae_sum += nbr_mae
                totals.count += inc

        if totals.count == 0:
            raise RuntimeError("No valid batches were evaluated.")

        results = {
            "MAE": totals.mae_sum / totals.count,
            "RMSE": totals.rmse_sum / totals.count,
            "PSNR": totals.psnr_sum / totals.count,
            "SAM": totals.sam_sum / totals.count,
            "SSIM": totals.ssim_sum / totals.count,
            "NDVI_MAE": totals.ndvi_mae_sum / totals.count,
            "NBR_MAE": totals.nbr_mae_sum / totals.count,
            "num_batches": totals.count,
        }
        print(json.dumps(results, indent=2))
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal AllClear benchmark runner")
    parser.add_argument("--dataset-fpath", type=str,
                        required=True, help="Path to dataset metadata JSON")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Wrapper class name, e.g., VPint2")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the model on, eg. cpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for data loading")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of workers for DataLoader")
    parser.add_argument("--main-sensor", type=str,
                        default="s2_toa", help="Main sensor for the dataset")
    parser.add_argument("--aux-sensors", type=str, nargs="*", default=[])
    parser.add_argument("--aux-data", type=str, nargs="+",
                        default=["cld_shdw", "dw"])
    # TODO: if s2s, t alignment needed
    parser.add_argument("--target-mode", type=str,
                        choices=["s2p", "s2s"], default="s2p")
    parser.add_argument("--tx", type=int, default=3,
                        help="Number of images in a sample for the dataset")
    parser.add_argument("--selected-rois", type=str, nargs="+",
                        default=None, help="Selected ROIs for benchmarking")
    # model-specific data paths
    parser.add_argument("--vpint2-pairs-fpath", type=str,
                        default=None, help="Path to VPint2 pairs JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = BenchmarkRunner(args)
    runner.run()
