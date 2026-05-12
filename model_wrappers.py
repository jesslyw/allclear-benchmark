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
