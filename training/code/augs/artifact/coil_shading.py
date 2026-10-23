"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Linear intensity gradient in a random direction — surface-coil falloff.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def coil_shading(x, lam, gen):
    """Linear intensity gradient across the image (surface-coil falloff)."""
    B, S, C, H, W = x.shape
    ang = float(_rand(gen, 0, 2 * math.pi))
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                            torch.linspace(-1, 1, W, device=x.device),
                            indexing="ij")
    g = (math.cos(ang) * xx + math.sin(ang) * yy)
    field = 1.0 + g * 0.35 * lam
    return (x * field.to(x.dtype)).clamp(0, 1)


@register_pixel
class CoilShading(PixelAugmentation):
    """Linear intensity gradient in a random direction — surface-coil falloff."""
    name = "coil_shading"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return coil_shading(x, lam, gen)
