"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). One square hole (8–28%·λ of H), same position across slices — local occlusion.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def cutout(x, lam, gen):
    """One random square hole per study (same position across slices)."""
    B, S, C, H, W = x.shape
    sz = int(H * (0.08 + 0.20 * lam))
    out = x.clone()
    for b in range(B):
        cy = int(_rand(gen, 0, 1) * (H - sz))
        cx = int(_rand(gen, 0, 1) * (W - sz))
        out[b, :, :, cy:cy + sz, cx:cx + sz] = 0.0
    return out


@register_pixel
class Cutout(PixelAugmentation):
    """One square hole (8–28%·λ of H), same position across slices — local occlusion."""
    name = "cutout"
    category = "structure"

    def apply(self, x, lam, gen):
        return cutout(x, lam, gen)
