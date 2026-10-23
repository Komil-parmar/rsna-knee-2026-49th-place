"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). In-plane shift up to ±15%·λ — off-centre positioning / FOV placement.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def translate(x, lam, gen):
    tx = float(_rand(gen, -1, 1)) * 0.15 * lam
    ty = float(_rand(gen, -1, 1)) * 0.15 * lam
    return _affine(x, tx=tx, ty=ty)


@register_pixel
class Translate(PixelAugmentation):
    """In-plane shift up to ±15%·λ — off-centre positioning / FOV placement."""
    name = "translate"
    category = "geometric"

    def apply(self, x, lam, gen):
        return translate(x, lam, gen)
