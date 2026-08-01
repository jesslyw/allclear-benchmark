import torch

_S2_B4 = 3   # Red    (0-indexed from the 13 S2-TOA bands)
_S2_B8 = 7   # NIR
_S2_B12 = 12  # SWIR-2


"""
Metric utilities used by benchmark.py.

Design notes:
- Input tensors are expected in (B, C, T, H, W).
- We flatten to (N, C, H, W) with N = B*T to compute one score per frame.
- valid_mask has shape (B, 1, T, H, W), i.e., one validity flag per pixel
    shared across all channels.
"""


def _ndvi(x: torch.Tensor) -> torch.Tensor:
    """x: (N, C, H, W) -> NDVI (N, 1, H, W)."""
    nir = x[:, _S2_B8:_S2_B8 + 1]
    red = x[:, _S2_B4:_S2_B4 + 1]
    return (nir - red) / (nir + red).clamp_min(1e-12)


def _nbr(x: torch.Tensor) -> torch.Tensor:
    """x: (N, C, H, W) -> NBR (N, 1, H, W)."""
    nir = x[:, _S2_B8:_S2_B8 + 1]
    swir = x[:, _S2_B12:_S2_B12 + 1]
    return (nir - swir) / (nir + swir).clamp_min(1e-12)


def _flatten_batch_time(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) -> (B*T, C, H, W)."""
    return x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], x.shape[3], x.shape[4])


def _flatten_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """(B, 1, T, H, W) -> (B*T, 1, H, W)."""
    return valid_mask.permute(0, 2, 1, 3, 4).reshape(-1, 1, valid_mask.shape[3], valid_mask.shape[4]).float()


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Element-wise num/den with NaN where denominator is zero."""
    out = torch.full_like(num, float("nan"), dtype=torch.float32)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def _mae_per_sample(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_exp = mask.expand_as(output)
    mae_masked = (output - target).abs() * mask_exp
    num = mae_masked.sum(dim=[1, 2, 3])
    den = mask_exp.sum(dim=[1, 2, 3])
    return _safe_div(num, den)


def _rmse_per_sample(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_exp = mask.expand_as(output)
    mse_masked = ((output - target) ** 2) * mask_exp
    num = mse_masked.sum(dim=[1, 2, 3])
    den = mask_exp.sum(dim=[1, 2, 3])
    mse = _safe_div(num, den)
    return torch.sqrt(mse)


def _psnr_per_sample(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, max_pixel: float = 1.0) -> torch.Tensor:
    rmse = _rmse_per_sample(output, target, mask)
    return 20 * torch.log10(output.new_tensor(max_pixel) / rmse.clamp_min(1e-12))


def _sam_per_sample(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    norm_out = torch.nn.functional.normalize(output, p=2, dim=1)
    norm_tar = torch.nn.functional.normalize(target, p=2, dim=1)
    dot_product = (norm_out * norm_tar).sum(dim=1).clamp(-1, 1)  # (N, H, W)
    angles = torch.rad2deg(torch.acos(dot_product))
    mask_2d = mask[:, 0] > 0.5
    angles_masked = torch.where(mask_2d, angles, torch.tensor(
        float("nan"), device=output.device))
    return torch.nanmean(angles_masked, dim=[1, 2])


def _ssim_per_sample(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    c, device = output.shape[1], output.device
    l = 1.0
    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2

    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    window = (g.view(1, 1, -1) * g.view(1, -1, 1)
              ).view(1, 1, window_size, window_size)
    window = window.repeat(c, 1, 1, 1)

    mask = mask > 0.5
    mu_x = torch.nn.functional.conv2d(
        output * mask, window, padding=window_size // 2, groups=c)
    mu_y = torch.nn.functional.conv2d(
        target * mask, window, padding=window_size // 2, groups=c)
    sigma_x = torch.nn.functional.conv2d(
        output ** 2 * mask, window, padding=window_size // 2, groups=c)
    sigma_y = torch.nn.functional.conv2d(
        target ** 2 * mask, window, padding=window_size // 2, groups=c)
    sigma_xy = torch.nn.functional.conv2d(
        output * target * mask, window, padding=window_size // 2, groups=c)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x -= mu_x_sq
    sigma_y -= mu_y_sq
    sigma_xy -= mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator

    mask_exp = mask.expand_as(ssim_map)
    ssim_map_masked = torch.where(
        mask_exp, ssim_map, torch.tensor(float("nan"), device=device))
    return torch.nanmean(ssim_map_masked, dim=[1, 2, 3])


def _index_mae_per_sample(index_out: torch.Tensor, index_tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # index_*: (N, 1, H, W), mask: (N, 1, H, W)
    mask_2d = mask[:, 0] > 0.5
    err = (index_out - index_tgt).abs()[:, 0]
    num = torch.where(mask_2d, err, torch.zeros_like(err)).sum(dim=[1, 2])
    den = mask_2d.sum(dim=[1, 2]).float()
    return _safe_div(num, den)


def compute_batch_metrics(
    output: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    output/target: (B, C, T, H, W), values expected in [0, 1]
    valid_mask:    (B, 1, T, H, W), True = valid target pixel

    Returns dict of per-frame metric vectors (N = B*T).
    """
    # Flatten batch and time so each frame gets one metric value.
    out = _flatten_batch_time(output)
    tgt = _flatten_batch_time(target)
    mask = _flatten_valid_mask(valid_mask)

    if int(mask.sum().item()) == 0:
        empty = torch.empty(0, device=output.device)
        return {
            "MAE": empty,
            "RMSE": empty,
            "PSNR": empty,
            "SAM": empty,
            "SSIM": empty,
            "NDVI_MAE": empty,
            "NBR_MAE": empty,
        }

    return {
        "MAE": _mae_per_sample(out, tgt, mask),
        "RMSE": _rmse_per_sample(out, tgt, mask),
        "PSNR": _psnr_per_sample(out, tgt, mask),
        "SAM": _sam_per_sample(out, tgt, mask),
        "SSIM": _ssim_per_sample(out, tgt, mask),
        "NDVI_MAE": _index_mae_per_sample(_ndvi(out), _ndvi(tgt), mask),
        "NBR_MAE": _index_mae_per_sample(_nbr(out), _nbr(tgt), mask),
    }
