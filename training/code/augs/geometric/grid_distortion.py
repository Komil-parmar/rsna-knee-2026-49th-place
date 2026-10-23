"""NEW op (2026-08-23). Smooth low-order warp from a 3x3 displacement grid —
gradient-nonlinearity distortion (subtler and smoother than `elastic`)."""
import torch
import torch.nn.functional as F

from augs.base import PixelAugmentation
from augs.registry import register_pixel


def grid_distortion(x, lam, gen):
    B, S, C, H, W = x.shape
    disp = (torch.rand(B, 2, 3, 3, generator=gen) - 0.5).to(x.device)
    disp = F.interpolate(disp, size=(H, W), mode="bicubic",
                         align_corners=False) * 0.05 * lam
    # pixel-center grid: the identity mapping under align_corners=False
    # (FIX 2026-08-23: legacy linspace(-1,1) grid resampled even at lam=0)
    ys = (torch.arange(H, device=x.device, dtype=x.dtype) * 2 + 1) / H - 1
    xs = (torch.arange(W, device=x.device, dtype=x.dtype) * 2 + 1) / W - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([xx, yy], -1).unsqueeze(0)
    grid = base + disp.permute(0, 2, 3, 1)
    flat = x.reshape(B, S * C, H, W)
    out = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)
    return out.reshape(B, S, C, H, W)


@register_pixel
class GridDistortion(PixelAugmentation):
    """3x3-node smooth warp (amp 0.05·λ) — gradient-nonlinearity distortion."""
    name = "grid_distortion"
    category = "geometric"

    def apply(self, x, lam, gen):
        return grid_distortion(x, lam, gen)
