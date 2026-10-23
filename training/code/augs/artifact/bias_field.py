"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). B1 inhomogeneity: smooth low-order multiplicative field (TorchIO-style), ±30%·λ.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def bias_field(x, lam, gen):
    """B1 inhomogeneity: smooth low-order multiplicative field (TorchIO-style)."""
    B, S, C, H, W = x.shape
    field = (torch.rand(B, 1, 3, 3, generator=gen) - 0.5).to(x.device)
    field = F.interpolate(field, size=(H, W), mode="bicubic",
                          align_corners=False)
    field = 1.0 + field * 0.6 * lam
    return (x * field.unsqueeze(1)).clamp(0, 1)


@register_pixel
class BiasField(PixelAugmentation):
    """B1 inhomogeneity: smooth low-order multiplicative field (TorchIO-style), ±30%·λ."""
    name = "bias_field"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return bias_field(x, lam, gen)
