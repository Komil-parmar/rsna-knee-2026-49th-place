"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). k-space truncation (keep (1−0.45·λ) of frequencies) — Gibbs ringing at edges.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def gibbs_ringing(x, lam, gen):
    """Truncate k-space (crop high frequencies) -> ringing at edges."""
    B, S, C, H, W = x.shape
    keep = int(H * (1.0 - 0.45 * lam)) // 2
    flat = x.reshape(-1, H, W)
    k = torch.fft.fftshift(torch.fft.fft2(flat), dim=(-2, -1))
    mask = torch.zeros(H, W, device=x.device)
    cy, cx = H // 2, W // 2
    mask[cy - keep:cy + keep, cx - keep:cx + keep] = 1.0
    out = torch.fft.ifft2(torch.fft.ifftshift(k * mask, dim=(-2, -1))).real
    return out.clamp(0, 1).reshape(B, S, C, H, W)


@register_pixel
class GibbsRinging(PixelAugmentation):
    """k-space truncation (keep (1−0.45·λ) of frequencies) — Gibbs ringing at edges."""
    name = "gibbs_ringing"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return gibbs_ringing(x, lam, gen)
