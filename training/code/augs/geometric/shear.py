"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Shear up to 12°·λ — mild oblique-acquisition geometry.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def shear(x, lam, gen):
    return _affine(x, shear=float(_rand(gen, -1, 1)) * 12.0 * lam)


@register_pixel
class Shear(PixelAugmentation):
    """Shear up to 12°·λ — mild oblique-acquisition geometry."""
    name = "shear"
    category = "geometric"

    def apply(self, x, lam, gen):
        return shear(x, lam, gen)
