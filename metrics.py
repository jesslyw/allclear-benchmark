import torch

_S2_B4 = 3   # Red    (0-indexed from the 13 S2-TOA bands)
_S2_B8 = 7   # NIR
_S2_B12 = 12  # SWIR-2


def _ndvi(x: torch.Tensor) -> torch.Tensor:
    """x: (B, C, T, H, W) → NDVI (B, 1, T, H, W)"""
    nir = x[:, _S2_B8:_S2_B8 + 1]
    red = x[:, _S2_B4:_S2_B4 + 1]
    return (nir - red) / (nir + red).clamp_min(1e-12)


def _nbr(x: torch.Tensor) -> torch.Tensor:
    """x: (B, C, T, H, W) → NBR (B, 1, T, H, W)"""
    nir = x[:, _S2_B8:_S2_B8 + 1]
    swir = x[:, _S2_B12:_S2_B12 + 1]
    return (nir - swir) / (nir + swir).clamp_min(1e-12)


def compute_batch_metrics(
    output: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[float, float, float, float, float, float, float, int]:
    """
    output/target: (B, C, T, H, W), values expected in [0, 1]
    valid_mask:    (B, 1, T, H, W), True = valid target pixel

    Returns: mae, rmse, psnr, sam, ssim, ndvi_mae, nbr_mae, count
    """
    diff = output - target
    valid = valid_mask.expand_as(diff)

    if int(valid.sum().item()) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    abs_err = diff.abs()[valid]
    sq_err = (diff * diff)[valid]

    mae = abs_err.mean()
    rmse = sq_err.mean().sqrt()
    psnr = -10.0 * torch.log10(sq_err.mean().clamp_min(1e-12))

    # SAM: spectral angle mapper, averaged over valid pixels
    pix_valid = valid_mask.squeeze(1)  # (B, T, H, W)
    out_vec = output.permute(0, 2, 3, 4, 1)[pix_valid]
    tgt_vec = target.permute(0, 2, 3, 4, 1)[pix_valid]

    dot = (out_vec * tgt_vec).sum(dim=-1)
    cos = dot / (torch.linalg.norm(out_vec, dim=-1) * torch.linalg.norm(tgt_vec, dim=-1)).clamp_min(1e-12)
    sam = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))).mean()

    # Global SSIM over valid pixels, averaged over channels
    ssims = []
    for c in range(output.shape[1]):
        x = output[:, c:c + 1][valid_mask]
        y = target[:, c:c + 1][valid_mask]
        if x.numel() < 2:
            continue
        mu_x, mu_y = x.mean(), y.mean()
        cov_xy = ((x - mu_x) * (y - mu_y)).mean()
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssims.append(
            ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) /
            ((mu_x ** 2 + mu_y ** 2 + c1) * (x.var(unbiased=False) + y.var(unbiased=False) + c2))
        )
    ssim = torch.stack(ssims).mean() if ssims else torch.tensor(0.0, device=output.device)

    ndvi_mae = (_ndvi(output) - _ndvi(target)).abs().squeeze(1)[pix_valid].mean()
    nbr_mae = (_nbr(output) - _nbr(target)).abs().squeeze(1)[pix_valid].mean()

    return mae.item(), rmse.item(), psnr.item(), sam.item(), ssim.item(), ndvi_mae.item(), nbr_mae.item(), 1
