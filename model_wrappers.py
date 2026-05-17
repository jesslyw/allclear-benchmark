from abc import ABC, abstractmethod
import json
import sys
from pathlib import Path
import numpy as np
import torch
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    # Source:
    # https://github.com/Zhou-Hangyu/allclear
    # File: baseline_wrappers.py
    # Commit:  3db4f86
    # License: MIT
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def preprocess(self, inputs):
        """

        Args:
            inputs:

        Returns:

        """
        pass

    @abstractmethod
    def forward(self, inputs):
        pass


class VPint2(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        vpint_root = Path(__file__).resolve().parent / "models" / "VPint2"
        if not vpint_root.exists():
            raise FileNotFoundError(
                f"VPint2 submodule not found at {vpint_root}. "
                "Add it under models/VPint2 first."
            )
        sys.path.append(str(vpint_root))
        from VPint.WP_MRP import WP_SMRP
        self.WP_SMRP = WP_SMRP
        self.clip_val = 1.0  # dataset.py normalizes optical channels to [0, 1]
        self.prioritise_identity = True
        # loads vpint2_subset.json
        eligibility_fpath = getattr(args, "vpint2_pairs_fpath", None)
        if not eligibility_fpath:
            raise ValueError(
                "VPint2 requires --vpint2-pairs-fpath. "
                "Create it with vpint2_filter.py first."
            )
        with open(eligibility_fpath, "r", encoding="utf-8") as f:
            self.eligibility = json.load(f)

    def get_model_config(self):
        return None

    @staticmethod
    def _batch_value(value, index):
        if isinstance(value, (list, tuple)):
            return value[index]
        return value

    def preprocess(self, inputs):
        # DataLoader adds B:
        # inputs["input_images"]:   (B, C, T, H, W)
        # inputs["target"]:         (B, C, 1, H, W) for s2p
        # inputs["input_cld_shdw"]: (B, 2, T, H, W)
        x = inputs["input_images"][:, :13]   # S2 bands only
        m = inputs["input_cld_shdw"]

        batch_items = []
        # B = batch, vpint2 runs with size=1 (how many samples per run)
        B, C, T, H, W = x.shape

        for b in range(B):
            data_id = self._batch_value(inputs["data_id"], b)
            choice = self.eligibility.get(data_id)
            if choice is None:  # not found in subset
                continue

            t_ref = int(choice["reference_index"])
            t_cloudy = int(choice["cloudy_index"])
            # index within range of batch (always 3)
            if not (0 <= t_ref < T and 0 <= t_cloudy < T):
                continue

            masks = m[b].sum(dim=0) > 0
            cloudy = x[b, :, t_cloudy].permute(1, 2, 0).cpu().numpy()
            feature = x[b, :, t_ref].permute(1, 2, 0).cpu().numpy()
            mask_2d = masks[t_cloudy].cpu().numpy()

            # VPint2 expects cloudy pixels as NaN so they can later be filled
            cloudy = cloudy.copy()
            cloudy[mask_2d, :] = np.nan

            batch_items.append({
                "cloudy": cloudy,
                "feature": feature,
                "mask": mask_2d,
                "t_ref": t_ref,
                "t_cloudy": t_cloudy,
                "data_id": data_id,
            })

        # forward() will consume this
        return {"vpint_batch": batch_items, "skip_batch": len(batch_items) == 0}

    def forward(self, inputs):
        batch_items = inputs["vpint_batch"]
        if not batch_items:
            return {"output": None}

        output_batch = []

        for item in batch_items:
            # (H, W, C), with NaNs on masked pixels
            cloudy = item["cloudy"]
            feature = item["feature"]    # (H, W, C)
            h, w, c = cloudy.shape
            pred = np.zeros((h, w, c), dtype=np.float32)

            for b in range(c):
                mrp = self.WP_SMRP(cloudy[:, :, b], feature[:, :, b])
                pred[:, :, b] = mrp.run(
                    method="exact",
                    clip_val=self.clip_val,
                    prioritise_identity=self.prioritise_identity,
                )

            pred_t = torch.from_numpy(pred).permute(
                2, 0, 1).unsqueeze(1)  # (C, 1, H, W)
            output_batch.append(pred_t)

        output = torch.stack(output_batch, dim=0).to(
            self.device)  # (B, C, 1, H, W)
        return {"output": output}


class UnCRtainTS(BaseModel):
    # Source:
    # https://github.com/Zhou-Hangyu/allclear
    # File: baseline_wrappers.py
    # Commit: 3db4f86
    # License: MIT
    def __init__(self, args):
        super().__init__(args)
        uncrtaints_root = Path(__file__).resolve().parent / "models" / "UnCRtainTS"
        if not uncrtaints_root.exists():
            raise FileNotFoundError(
                f"UnCRtainTS submodule not found at {uncrtaints_root}. "
                "Add it with: git submodule add https://github.com/PatrickTUM/UnCRtainTS models/UnCRtainTS"
            )
        sys.path.insert(0, str(uncrtaints_root))
        sys.path.insert(0, str(uncrtaints_root / "model"))
        from model.src.model_utils import get_model, load_checkpoint
        from model.src.utils import str2list
        from model.parse_args import create_parser

        base_path = Path(getattr(args, "uncrtaints_base_path", str(uncrtaints_root)))
        weight_folder = args.uncrtaints_weight_folder
        experiment_name = args.uncrtaints_experiment_name
        self.experiment_name = experiment_name
        resume_at = getattr(args, "uncrtaints_resume_at", 0)

        conf_path = base_path / weight_folder / experiment_name / "conf.json"
        with open(conf_path, "r") as f:
            model_config = json.load(f)

        parser = create_parser(mode="test")
        no_overwrite = ["pid", "device", "resume_at", "trained_checkp", "res_dir",
                        "weight_folder", "root1", "root2", "root3", "max_samples_count",
                        "batch_size", "display_step", "plot_every", "export_every",
                        "input_t", "region", "min_cov", "max_cov", "f"]
        conf_dict = {k: v for k, v in model_config.items() if k not in no_overwrite}
        conf_dict["resume_at"] = resume_at
        conf_dict["weight_folder"] = str(base_path / weight_folder)
        conf_dict["device"] = str(self.device)
        import argparse as _argparse
        t_args = _argparse.Namespace(**conf_dict)
        config, _ = parser.parse_known_args(namespace=t_args)
        config = str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])

        self.model = get_model(config).to(self.device)
        ckpt_n = f"_epoch_{resume_at}" if resume_at > 0 else ""
        load_checkpoint(config, str(base_path / weight_folder), self.model, f"model{ckpt_n}")
        self.model.eval()

        self.num_input_dims = 13 if "noSAR_1" in experiment_name else 15
        self.S2_BANDS = 13

    def get_model_config(self):
        return None

    def preprocess(self, inputs):
        inputs["input_images"] = inputs["input_images"].to(self.device)
        inputs["input_cld_shdw"] = inputs["input_cld_shdw"].to(self.device)
        inputs["input_images"] = inputs["input_images"].permute(0, 2, 1, 3, 4)[:, :, :self.num_input_dims]
        inputs["input_cld_shdw"] = torch.clip(inputs["input_cld_shdw"].sum(dim=1), 0, 1)
        # Store under private key to avoid corrupting batch["target"] shape (used by benchmark.py for target_c)
        inputs["_uc_target"] = inputs["target"].to(self.device).permute(0, 2, 1, 3, 4)[:, :, :self.S2_BANDS]
        # diagonal_1 was trained with S1 channels first; move them from [13:15] to front
        if "diagonal_1" in self.experiment_name:
            inputs["input_images"] = torch.cat(
                [inputs["input_images"][:, :, -2:], inputs["input_images"][:, :, :-2]], dim=2)
        return inputs

    def forward(self, inputs):
        input_imgs = inputs["input_images"]           # (B, T, C, H, W)
        target_imgs = inputs["_uc_target"]            # (B, 1, 13, H, W)
        masks = inputs["input_cld_shdw"]              # (B, T, H, W)
        dates = inputs["time_differences"].to(self.device)  # (B, T)
        model_inputs = {"A": input_imgs, "B": target_imgs, "dates": dates, "masks": masks}

        with torch.no_grad():
            self.model.set_input(model_inputs)
            self.model.forward()
            self.model.rescale()
            out = self.model.fake_B[:, :, :self.S2_BANDS]  # (B, T, 13, H, W)
        return {"output": out}


class LeastCloudy(BaseModel):
    # Source: https://github.com/Zhou-Hangyu/allclear (baseline_wrappers.py, MIT License)
    def get_model_config(self):
        return None

    def preprocess(self, inputs):
        return inputs

    def forward(self, inputs):
        x = inputs["input_images"][:, :13].to(self.device)  # (B, 13, T, H, W)
        m = inputs["input_cld_shdw"].to(self.device)        # (B, 2,  T, H, W)
        cloudiness = m.sum(dim=(1, 3, 4))                   # (B, T)
        t_best = cloudiness.argmin(dim=1)                   # (B,)
        B, C, T, H, W = x.shape
        idx = t_best.view(B, 1, 1, 1, 1).expand(B, C, 1, H, W)
        return {"output": x.gather(2, idx)}                 # (B, 13, 1, H, W)


class Mosaicing(BaseModel):
    # Source: https://github.com/Zhou-Hangyu/allclear (baseline_wrappers.py, MIT License)
    def get_model_config(self):
        return None

    def preprocess(self, inputs):
        return inputs

    def forward(self, inputs):
        x = inputs["input_images"][:, :13].to(self.device)  # (B, 13, T, H, W)
        m = inputs["input_cld_shdw"].to(self.device)        # (B, 2,  T, H, W)
        clear = (1 - m.sum(dim=1, keepdim=True).clamp(0, 1))  # (B, 1, T, H, W)
        clear = clear.expand_as(x)
        sum_pixels = (x * clear).sum(dim=2, keepdim=True)   # (B, 13, 1, H, W)
        sum_views = clear.sum(dim=2, keepdim=True).clamp(min=1)
        return {"output": sum_pixels / sum_views}            # (B, 13, 1, H, W)


class EMRDM(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        emrdm_root = Path(__file__).resolve().parent / "models" / "EMRDM"
        if not emrdm_root.exists():
            raise FileNotFoundError(
                f"EMRDM submodule not found at {emrdm_root}. "
                "Add it with: git submodule add https://github.com/Ly403/EMRDM models/EMRDM"
            )
        sys.path.insert(0, str(emrdm_root))
        from omegaconf import OmegaConf
        from sgm.util import instantiate_from_config
        config = OmegaConf.load(args.emrdm_config_fpath)
        self.model = instantiate_from_config(config.model)
        ckpt = torch.load(args.emrdm_ckpt_fpath, map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval().to(self.device)
        self.model.sampler.device = str(self.device)

    def get_model_config(self):
        return None

    @staticmethod
    def _gather_t(tensor, indices):
        """Index (B, C, T, H, W) per sample along T using a (B,) index tensor."""
        B, C, T, H, W = tensor.shape
        idx = indices.view(B, 1, 1, 1, 1).expand(B, C, 1, H, W)
        return tensor.gather(2, idx).squeeze(2)  # (B, C, H, W)

    def preprocess(self, inputs):
        x = inputs["input_images"].to(self.device)
        s2 = x[:, :13]   # (B, 13, T, H, W)
        s1 = x[:, 13:15]  # (B, 2,  T, H, W)
        m = inputs["input_cld_shdw"].to(self.device)  # (B, 2, T, H, W)

        cloud_frac = m.sum(dim=1).mean(dim=(-1, -2))  # (B, T)
        t_ref = cloud_frac.argmin(dim=1)              # (B,)
        t_cloudy = cloud_frac.argmax(dim=1)           # (B,)

        ref_s2 = self._gather_t(s2, t_ref)       # (B, 13, H, W)
        cloudy_s2 = self._gather_t(s2, t_cloudy)  # (B, 13, H, W)
        s1_frame = self._gather_t(s1, t_cloudy)   # (B, 2,  H, W)

        def scale(x): return x * 2.0 - 1.0

        S1S2 = torch.cat([scale(s1_frame), scale(cloudy_s2)], dim=1)  # (B, 15, H, W): S1 first, then cloudy S2 (matches SEN12MS-CR training order)

        # sentinel.yaml uses "S2" as mean_key and "S1S2" as conditioner input_key
        return {"emrdm_batch": {"S1S2": S1S2, "S2": scale(ref_s2)},
                "t_cloudy": t_cloudy}

    def forward(self, inputs):
        batch = inputs["emrdm_batch"]
        with torch.no_grad():
            c, uc = self.model.conditioner.get_unconditional_conditioning(
                batch, force_uc_zero_embeddings=["S1S2"]
            )
            B = batch["S2"].shape[0]
            z_mu = self.model.encode_first_stage(batch["S2"])
            samples, _ = self.model.sample(
                c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=B
            )
            out = self.model.decode_first_stage(samples)
        out_01 = ((out + 1.0) / 2.0).clamp(0.0, 1.0)
        return {"output": out_01.unsqueeze(2)}  # (B, 13, 1, H, W)
