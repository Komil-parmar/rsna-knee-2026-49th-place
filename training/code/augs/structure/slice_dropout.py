"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Zero 1–2 random slices — missing/corrupt slice robustness (occlusion along depth).
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def slice_dropout(x, lam, gen):
    """Zero up to 2 random slices per study (occlusion along depth)."""
    B, S, C, H, W = x.shape
    n = 1 + int(lam > 0.5)
    out = x.clone()
    for b in range(B):
        idx = torch.randperm(C, generator=gen)[:n]
        out[b, :, idx] = 0.0
    return out


@register_pixel
class SliceDropout(PixelAugmentation):
    """Zero 1–2 random slices — missing/corrupt slice robustness (occlusion along depth)."""
    name = "slice_dropout"
    category = "structure"

    def apply(self, x, lam, gen):
        return slice_dropout(x, lam, gen)
