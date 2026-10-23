"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Low-res random displacement field (6×6, 0.04·λ) — soft-tissue deformation.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def elastic(x, lam, gen):
    """Mild elastic deformation via low-res random displacement field."""
    B, S, C, H, W = x.shape
    disp = (torch.rand(B, 2, 6, 6, generator=gen) - 0.5).to(x.device)
    disp = F.interpolate(disp, size=(H, W), mode="bicubic",
                         align_corners=False) * 0.04 * lam
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
class Elastic(PixelAugmentation):
    """Low-res random displacement field (6×6, 0.04·λ) — soft-tissue deformation."""
    name = "elastic"
    category = "geometric"

    def apply(self, x, lam, gen):
        return elastic(x, lam, gen)
