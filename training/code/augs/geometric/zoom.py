"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Scale 1±0.2·λ — magnification / knee-size and FOV variation.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def zoom(x, lam, gen):
    s = 1.0 + float(_rand(gen, -1, 1)) * 0.20 * lam      # 0.8-1.2x
    return _affine(x, scale=s)


@register_pixel
class Zoom(PixelAugmentation):
    """Scale 1±0.2·λ — magnification / knee-size and FOV variation."""
    name = "zoom"
    category = "geometric"

    def apply(self, x, lam, gen):
        return zoom(x, lam, gen)
