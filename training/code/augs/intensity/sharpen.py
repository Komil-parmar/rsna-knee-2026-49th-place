"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Unsharp-mask blend (0.8·λ) — edge-enhancement filter variation.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def sharpen(x, lam, gen):
    B, S, C, H, W = x.shape
    k = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                     dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    flat = x.reshape(-1, 1, H, W)
    sh = F.conv2d(F.pad(flat, (1, 1, 1, 1), mode="replicate"), k)
    out = flat + lam * 0.8 * (sh - flat)
    return out.clamp(0, 1).reshape(B, S, C, H, W)


@register_pixel
class Sharpen(PixelAugmentation):
    """Unsharp-mask blend (0.8·λ) — edge-enhancement filter variation."""
    name = "sharpen"
    category = "intensity"

    def apply(self, x, lam, gen):
        return sharpen(x, lam, gen)
