"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Separable blur σ=0.3+1.2·λ — smoothing-filter / mild resolution variation.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def gaussian_blur(x, lam, gen):
    B, S, C, H, W = x.shape
    sig = 0.3 + 1.2 * lam
    r = 3
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k1 = torch.exp(-t ** 2 / (2 * sig ** 2))
    k1 = (k1 / k1.sum()).view(1, 1, 1, -1)
    flat = x.reshape(-1, 1, H, W)
    out = F.conv2d(F.pad(flat, (r, r, 0, 0), mode="replicate"), k1)
    out = F.conv2d(F.pad(out, (0, 0, r, r), mode="replicate"),
                   k1.transpose(-1, -2))
    return out.reshape(B, S, C, H, W)


@register_pixel
class GaussianBlur(PixelAugmentation):
    """Separable blur σ=0.3+1.2·λ — smoothing-filter / mild resolution variation."""
    name = "gaussian_blur"
    category = "intensity"

    def apply(self, x, lam, gen):
        return gaussian_blur(x, lam, gen)
