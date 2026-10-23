"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Roll the slice axis by ±2·λ positions — slice-window placement variation (slot pipeline).
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def slice_shift(x, lam, gen):
    """Shift the 9-slice window inside each slot by up to +-2 positions."""
    k = int(round(float(_rand(gen, -1, 1)) * 2 * lam))
    if k == 0:
        return x
    return torch.roll(x, shifts=k, dims=2)


@register_pixel
class SliceShift(PixelAugmentation):
    """Roll the slice axis by ±2·λ positions — slice-window placement variation (slot pipeline)."""
    name = "slice_shift"
    category = "structure"

    def apply(self, x, lam, gen):
        return slice_shift(x, lam, gen)
