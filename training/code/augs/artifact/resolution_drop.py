"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Down/up-sample by (1−0.5·λ) — acquisition-resolution loss (low-field look).
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def resolution_drop(x, lam, gen):
    """Acquisition-resolution loss: downsample then upsample (low-field look)."""
    B, S, C, H, W = x.shape
    f = 1.0 - 0.5 * lam
    h = max(32, int(H * f))
    flat = x.reshape(-1, 1, H, W)
    out = F.interpolate(F.interpolate(flat, size=(h, h), mode="area"),
                        size=(H, W), mode="bilinear", align_corners=False)
    return out.reshape(B, S, C, H, W)


@register_pixel
class ResolutionDrop(PixelAugmentation):
    """Down/up-sample by (1−0.5·λ) — acquisition-resolution loss (low-field look)."""
    name = "resolution_drop"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return resolution_drop(x, lam, gen)
