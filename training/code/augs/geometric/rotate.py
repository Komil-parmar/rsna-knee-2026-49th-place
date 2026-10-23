"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). In-plane rotation up to ±20°·λ — patient positioning variation.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def rotate(x, lam, gen):
    ang = float(_rand(gen, -1, 1)) * 20.0 * lam          # up to +-20 deg
    return _affine(x, angle=ang)


@register_pixel
class Rotate(PixelAugmentation):
    """In-plane rotation up to ±20°·λ — patient positioning variation."""
    name = "rotate"
    category = "geometric"

    def apply(self, x, lam, gen):
        return rotate(x, lam, gen)
