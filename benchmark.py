"""
Benchmark runner for AllClear

Adapted from: https://github.com/Zhou-Hangyu/allclear (allclear/benchmark.py)
License: MIT
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import model_wrappers as wrappers
from dataset import AllClearDataset
from metrics import compute_batch_metrics


LULC_METRICS = ["MAE", "RMSE", "PSNR", "SAM", "SSIM", "NDVI_MAE", "NBR_MAE"]


def _plot_lulc_metrics(metrics_data: dict[int, dict[str, float]], save_dir: Path, model_name: str) -> None:
    """Plot class-wise LULC metrics using Dynamic World colors."""
    dw_colors = [
        "#000000",  # placeholder for class -1
        "#419bdf",  # 0
        "#397d49",  # 1
        "#88b053",  # 2
        "#7a87c6",  # 3
        "#e49635",  # 4
        "#dfc35a",  # 5
        "#c4281b",  # 6
        "#a59b8f",  # 7
        "#b39fe1",  # 8
    ]
    metric_order = LULC_METRICS
    vmax_by_metric = {
        "MAE": 0.1,
        "RMSE": 0.1,
        "PSNR": 40,
        "SAM": 15,
        "SSIM": 1,
        "NDVI_MAE": 1,
        "NBR_MAE": 1,
    }
    class_indices = sorted(metrics_data.keys())

    fig, axes = plt.subplots(
        1, len(metric_order), figsize=(4 * len(metric_order), 5), sharex=True, dpi=200
    )
    if len(metric_order) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metric_order):
        values = [metrics_data[c].get(metric, float("nan"))
                  for c in class_indices]
        ax.bar(class_indices, values, color=[
               dw_colors[c + 1] for c in class_indices])
        ax.set_title(metric)
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_xticks(class_indices)
        ax.grid(True)
        y_max = vmax_by_metric.get(metric)
        if y_max is not None:
            ax.set_ylim(0, y_max)

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_dir / f"{model_name}_lulc_metrics.png", bbox_inches="tight")
    fig.savefig(
        save_dir / f"{model_name}_lulc_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def _rgb_from_chw(frame_chw: torch.Tensor) -> torch.Tensor:
    """Convert (C,H,W) tensor in [0,1] to RGB (H,W,3)."""
    if frame_chw.shape[0] >= 4:
        bands = [3, 2, 1]  # S2 RGB
    elif frame_chw.shape[0] >= 3:
        bands = [2, 1, 0]
    else:
        # Fallback: repeat the first channel if we don't have enough channels.
        first = frame_chw[0:1]
        frame_chw = torch.cat([first, first, first], dim=0)
        bands = [0, 1, 2]
    return frame_chw[bands].permute(1, 2, 0).clamp(0, 1)


def _batch_item(value, index):
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _get_model_input_views(batch: dict, prepped: dict, batch_index: int, model_name: str) -> list[tuple[str, torch.Tensor]]:
    """Return model-aware list of input frames to visualize as (title, CHW tensor)."""
    inputs_s2 = batch["input_images"][batch_index, :13]  # (C,T,H,W)
    n_t = inputs_s2.shape[1]
    model_name = model_name.lower()

    if model_name == "vpint2" and "vpint_batch" in prepped:
        data_id = _batch_item(batch["data_id"], batch_index)
        matched = None
        for item in prepped["vpint_batch"]:
            if item.get("data_id") == data_id:
                matched = item
                break
        if matched is not None:
            t_ref = int(matched["t_ref"])
            t_cloudy = int(matched["t_cloudy"])
            views = []
            if 0 <= t_ref < n_t:
                views.append((f"Input ref t={t_ref}", inputs_s2[:, t_ref]))
            if 0 <= t_cloudy < n_t:
                views.append(
                    (f"Input cloudy t={t_cloudy}", inputs_s2[:, t_cloudy]))
            if views:
                return views

    if model_name == "emrdm" and "t_cloudy" in prepped:
        t_cloudy = int(prepped["t_cloudy"][batch_index].item())
        if 0 <= t_cloudy < n_t:
            return [(f"Input cloudy t={t_cloudy}", inputs_s2[:, t_cloudy])]

    return [(f"Input t={t}", inputs_s2[:, t]) for t in range(n_t)]


def _save_batch_visualizations(
    batch: dict,
    prepped: dict,
    pred: torch.Tensor,
    target: torch.Tensor,
    out_dir: Path,
    model_name: str,
    vis_count: int,
    max_vis_samples: int,
) -> int:
    """Save qualitative panels and return updated visualization count."""
    vis_dir = out_dir / f"{model_name}_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    b_size = pred.shape[0]
    for b in range(b_size):
        if max_vis_samples > 0 and vis_count >= max_vis_samples:
            return vis_count

        data_id = str(_batch_item(batch["data_id"], b))
        input_views = _get_model_input_views(batch, prepped, b, model_name)

        pred_t = 0
        tgt_t = 0
        pred_frame = pred[b, :, pred_t].detach().cpu()
        target_frame = target[b, :, tgt_t].detach().cpu()
        panels = input_views + \
            [("Prediction", pred_frame), ("Target", target_frame)]

        n_panels = len(panels)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(3 * n_panels, 3), dpi=160)
        if n_panels == 1:
            axes = [axes]

        for ax, (title, frame) in zip(axes, panels):
            rgb = _rgb_from_chw(frame).numpy()
            ax.imshow(rgb)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(data_id, fontsize=9)
        fig.tight_layout()

        safe_id = data_id.replace("/", "_").replace(" ", "_")
        fig.savefig(vis_dir / f"{safe_id}.png", bbox_inches="tight")
        plt.close(fig)
        vis_count += 1

    return vis_count


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
    cld = _to_bcthw(batch["target_cld_shdw"].to(target.device), target_c=2)
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
        if self.args.model_name.lower() == "vpint2" and batch_size != 1:
            print(
                f"[INFO] VPint2 pair-based evaluation uses batch-size=1 to keep prediction/target alignment safe. "
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
        lulc_metric_values = {
            c: {
                "MAE": [],
                "RMSE": [],
                "PSNR": [],
                "SAM": [],
                "SSIM": [],
                "NDVI_MAE": [],
                "NBR_MAE": [],
            }
            for c in range(9)
        }
        predictions = []
        metadata_rows = []
        vis_count = 0

        dataset_name = Path(self.args.dataset_fpath).stem
        output_dir = Path(self.args.experiment_output_path) / \
            "AllClear" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

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
                # Preserve raw inputs for visualization before wrappers reshape batch tensors.
                vis_batch = {
                    "input_images": batch["input_images"],
                    "data_id": batch.get("data_id"),
                }
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
                predictions.append(pred.detach().cpu())

                if self.args.draw_vis == 1:
                    vis_count = _save_batch_visualizations(
                        batch=vis_batch,
                        prepped=prepped,
                        pred=pred,
                        target=target,
                        out_dir=output_dir,
                        model_name=self.args.model_name,
                        vis_count=vis_count,
                        max_vis_samples=self.args.max_vis_samples,
                    )

                valid_mask = _valid_mask(batch, target)
                batch_metrics = compute_batch_metrics(pred, target, valid_mask)
                if batch_metrics["MAE"].numel() == 0:
                    continue
                for key, values in batch_metrics.items():
                    metric_values[key].append(values.detach().cpu())

                # metadata rows (one row per evaluated frame)
                data_ids = batch["data_id"]
                n_frames = int(batch_metrics["MAE"].numel())
                cld_stats = batch.get("input_cld_shdw")
                avg_cloud = None
                avg_shadow = None
                consistent_cloud = None
                consistent_shadow = None
                if cld_stats is not None:
                    cld_stats = _to_bcthw(
                        cld_stats.to(self.device), target_c=2)
                    # B,C,T,H,W -> B,C
                    avg_cld_shdw = torch.mean(cld_stats, dim=[2, 3, 4]).cpu()
                    # consistent cloud/shadow across all input frames
                    consistent = (torch.sum(cld_stats, dim=2)
                                  == cld_stats.shape[2]).float()
                    consistent_cld_shdw = torch.mean(
                        consistent, dim=[2, 3]).cpu()
                    avg_cloud = avg_cld_shdw[:, 0].tolist()
                    avg_shadow = avg_cld_shdw[:, 1].tolist()
                    consistent_cloud = consistent_cld_shdw[:, 0].tolist()
                    consistent_shadow = consistent_cld_shdw[:, 1].tolist()

                values_by_metric = {k: v.detach().cpu().tolist()
                                    for k, v in batch_metrics.items()}
                if len(data_ids) == n_frames:
                    for i, data_id in enumerate(data_ids):
                        row = {
                            "data_id": data_id,
                            "mae": values_by_metric["MAE"][i],
                            "rmse": values_by_metric["RMSE"][i],
                            "psnr": values_by_metric["PSNR"][i],
                            "sam": values_by_metric["SAM"][i],
                            "ssim": values_by_metric["SSIM"][i],
                            "ndvi_mae": values_by_metric["NDVI_MAE"][i],
                            "nbr_mae": values_by_metric["NBR_MAE"][i],
                        }
                        if avg_cloud is not None:
                            row["avg_cld_percent"] = avg_cloud[i]
                            row["avg_shdw_percent"] = avg_shadow[i]
                            row["consistent_cld_percent"] = consistent_cloud[i]
                            row["consistent_shdw_percent"] = consistent_shadow[i]
                        metadata_rows.append(row)
                else:
                    # fallback for T>1: keep rows aligned with frame-wise metrics
                    idx = 0
                    for i, data_id in enumerate(data_ids):
                        for t in range(target.shape[2]):
                            if idx >= n_frames:
                                break
                            row = {
                                "data_id": f"{data_id}_t{t}",
                                "mae": values_by_metric["MAE"][idx],
                                "rmse": values_by_metric["RMSE"][idx],
                                "psnr": values_by_metric["PSNR"][idx],
                                "sam": values_by_metric["SAM"][idx],
                                "ssim": values_by_metric["SSIM"][idx],
                                "ndvi_mae": values_by_metric["NDVI_MAE"][idx],
                                "nbr_mae": values_by_metric["NBR_MAE"][idx],
                            }
                            if avg_cloud is not None:
                                row["avg_cld_percent"] = avg_cloud[i]
                                row["avg_shdw_percent"] = avg_shadow[i]
                                row["consistent_cld_percent"] = consistent_cloud[i]
                                row["consistent_shdw_percent"] = consistent_shadow[i]
                            metadata_rows.append(row)
                            idx += 1

                # class-wise LULC metrics (artifacts 3/4)
                if "target_dw" in batch:
                    target_dw = _to_bcthw(
                        batch["target_dw"].to(self.device), target_c=1)
                    class_map = target_dw[:, 0:1, ...]
                    for class_id in range(9):
                        class_valid = valid_mask & (class_map == class_id)
                        class_metrics = compute_batch_metrics(
                            pred, target, class_valid)
                        if class_metrics["MAE"].numel() == 0:
                            continue
                        for metric_name in LULC_METRICS:
                            lulc_metric_values[class_id][metric_name].append(
                                class_metrics[metric_name].detach().cpu()
                            )

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

        # artifact 1: predictions tensor
        if predictions:
            torch.save(torch.cat(predictions, dim=0), output_dir /
                       f"{self.args.model_name}_predictions.pt")

        # artifact 2: per-sample metadata CSV
        if metadata_rows:
            metadata_fields = [
                "data_id",
                "avg_cld_percent",
                "avg_shdw_percent",
                "consistent_cld_percent",
                "consistent_shdw_percent",
                "mae",
                "rmse",
                "psnr",
                "sam",
                "ssim",
                "ndvi_mae",
                "nbr_mae",
            ]
            with open(output_dir / f"{self.args.model_name}_metadata.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=metadata_fields)
                writer.writeheader()
                for row in metadata_rows:
                    writer.writerow({k: row.get(k, float("nan"))
                                    for k in metadata_fields})

        # artifacts 3/4: class-wise LULC metrics CSV + plot
        final_lulc_metrics = {}
        for class_id in range(9):
            class_result = {}
            for metric_name in LULC_METRICS:
                pieces = lulc_metric_values[class_id][metric_name]
                if not pieces:
                    class_result[metric_name] = float("nan")
                else:
                    class_result[metric_name] = torch.nanmean(
                        torch.cat(pieces, dim=0)).item()
            final_lulc_metrics[class_id] = class_result

        with open(output_dir / f"{self.args.model_name}_lulc_metrics.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = ["class", *LULC_METRICS]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for class_id in range(9):
                writer.writerow(
                    {"class": class_id, **final_lulc_metrics[class_id]})

        _plot_lulc_metrics(final_lulc_metrics, output_dir,
                           self.args.model_name)

        # artifact 5: aggregated metrics CSV
        with open(output_dir / f"{self.args.model_name}_aggregated_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)

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
    parser.add_argument("--experiment-output-path", type=str,
                        default="outputs", help="Base output path for benchmark artifacts")
    parser.add_argument("--draw-vis", type=int, default=0,
                        help="0: do not save qualitative panels, 1: save panels")
    parser.add_argument("--max-vis-samples", type=int, default=0,
                        help="Maximum number of samples to visualize (0 means all valid samples)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = BenchmarkRunner(args)
    runner.run()
