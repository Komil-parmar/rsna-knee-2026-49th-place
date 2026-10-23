"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Additive ±0.25·λ — global intensity offset.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def brightness(x, lam, gen):
    return (x + float(_rand(gen, -1, 1)) * 0.25 * lam).clamp(0, 1)


@register_pixel
class Brightness(PixelAugmentation):
    """Additive ±0.25·λ — global intensity offset."""
    name = "brightness"
    category = "intensity"

    def apply(self, x, lam, gen):
        return brightness(x, lam, gen)
