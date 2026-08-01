"""
Benchmark runner for AllClear

Adapted from: https://github.com/Zhou-Hangyu/allclear (allclear/benchmark.py)
License: MIT
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import model_wrappers as wrappers
from dataset import AllClearDataset
from metrics import compute_batch_metrics


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
        if name == "emrdm":
            return wrappers.EMRDM(self.args)
        if name == "leastcloudy":
            return wrappers.LeastCloudy(self.args)
        if name == "mosaicing":
            return wrappers.Mosaicing(self.args)
        if name == "uncrtaints":
            return wrappers.UnCRtainTS(self.args)
        raise ValueError(
            f"Unknown model '{self.args.model_name}'. Available: VPint2, EMRDM, LeastCloudy, Mosaicing, UnCRtainTS")

    def _setup_data_loader(self):
        with open(self.args.dataset_fpath, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)
        batch_size = self.args.batch_size
        if self.args.model_name.lower() == "emrdm" and batch_size != 1:
            print(
                f"[INFO] EMRDM pointer-based evaluation uses batch-size=1 to keep prediction/target alignment safe. "
                f"Overriding --batch-size {batch_size} -> 1."
            )
            batch_size = 1
            self.args.batch_size = 1
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
            s1_preprocess_mode=self.args.s1_preprocess_mode,
            max_diff=self.args.s1_max_diff,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers)

    def run(self):
        metric_values = {
            "MAE": [],
            "RMSE": [],
            "PSNR": [],
            "SAM": [],
            "SSIM": [],
            "NDVI_MAE": [],
            "NBR_MAE": [],
        }

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
                    # for emrdm/vpint2: wrapper marked this batch as unusable (sample(s) not in filtered test set or had no usable matched S1 input)
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

                batch_metrics = compute_batch_metrics(
                    pred, target, _valid_mask(batch, target)
                )
                if batch_metrics["MAE"].numel() == 0:
                    continue
                for key, values in batch_metrics.items():
                    metric_values[key].append(values.detach().cpu())

        if not metric_values["MAE"]:
            raise RuntimeError("No valid batches were evaluated.")

        merged = {
            key: torch.cat(values, dim=0)
            for key, values in metric_values.items()
        }
        num_samples = int(torch.isfinite(merged["MAE"]).sum().item())

        results = {
            "MAE": torch.nanmean(merged["MAE"]).item(),
            "RMSE": torch.nanmean(merged["RMSE"]).item(),
            "PSNR": torch.nanmean(merged["PSNR"]).item(),
            "SAM": torch.nanmean(merged["SAM"]).item(),
            "SSIM": torch.nanmean(merged["SSIM"]).item(),
            "NDVI_MAE": torch.nanmean(merged["NDVI_MAE"]).item(),
            "NBR_MAE": torch.nanmean(merged["NBR_MAE"]).item(),
            "num_samples": num_samples,
        }
        print(json.dumps(results, indent=2))
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal AllClear benchmark runner")
    parser.add_argument("--dataset-fpath", type=str, default="setup/vpint2_samples.json",
                        help="Path to dataset metadata JSON")
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
    parser.add_argument("--s1-preprocess-mode", type=str, default="default",
                        choices=["default", "uncrtaints"],
                        help="S1 normalization: 'uncrtaints' for UnCRtainTS model")
    parser.add_argument("--s1-max-diff", type=int, default=2,
                        help="Max days between S1 and S2 acquisition for temporal alignment (default: 2)")
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
                        default="setup/vpint2_pairs.json", help="Path to VPint2 pairs JSON")
    parser.add_argument("--emrdm-config-fpath", type=str,
                        default=None, help="Path to EMRDM config YAML")
    parser.add_argument("--emrdm-ckpt-fpath", type=str,
                        default=None, help="Path to EMRDM checkpoint (.ckpt)")
    parser.add_argument("--emrdm-pairs-fpath", type=str,
                        default=None, help="Path to emrdm_pairs.json (pre-selected S2 frame indices)")
    parser.add_argument("--emrdm-no-s1", action="store_true",
                        help="Zero out S1 channels (ablation: measure benefit of SAR conditioning)")
    parser.add_argument("--uncrtaints-base-path", type=str,
                        default="models/UnCRtainTS", help="Path to UnCRtainTS repo root")
    parser.add_argument("--uncrtaints-weight-folder", type=str,
                        default=None, help="UnCRtainTS weight folder name")
    parser.add_argument("--uncrtaints-experiment-name", type=str,
                        default=None, help="UnCRtainTS experiment name")
    parser.add_argument("--uncrtaints-resume-at", type=int,
                        default=0, help="UnCRtainTS checkpoint epoch (0 = latest)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = BenchmarkRunner(args)
    runner.run()
