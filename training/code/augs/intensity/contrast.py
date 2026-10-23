"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Contrast 1±0.5·λ around the image mean.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def contrast(x, lam, gen):
    c = 1.0 + float(_rand(gen, -1, 1)) * 0.5 * lam
    m = x.mean(dim=(-2, -1), keepdim=True)
    return ((x - m) * c + m).clamp(0, 1)


@register_pixel
class Contrast(PixelAugmentation):
    """Contrast 1±0.5·λ around the image mean."""
    name = "contrast"
    category = "intensity"

    def apply(self, x, lam, gen):
        return contrast(x, lam, gen)
