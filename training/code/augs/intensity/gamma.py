"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Gamma 1±0.5·λ — display/scanner tone-curve variation.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def gamma(x, lam, gen):
    g = 1.0 + float(_rand(gen, -1, 1)) * 0.5 * lam       # 0.5-1.5
    return x.clamp(0, 1) ** max(g, 0.1)


@register_pixel
class Gamma(PixelAugmentation):
    """Gamma 1±0.5·λ — display/scanner tone-curve variation."""
    name = "gamma"
    category = "intensity"

    def apply(self, x, lam, gen):
        return gamma(x, lam, gen)
