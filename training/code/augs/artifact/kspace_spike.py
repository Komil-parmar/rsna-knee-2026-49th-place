"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). One bright off-centre k-space point — herringbone/spike artifact.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def kspace_spike(x, lam, gen):
    """Herringbone/spike artifact: one bright off-center k-space point."""
    B, S, C, H, W = x.shape
    flat = x.reshape(-1, H, W)
    k = torch.fft.fftshift(torch.fft.fft2(flat), dim=(-2, -1))
    dy = int(float(_rand(gen, 0.05, 0.3)) * H)
    dx = int(float(_rand(gen, 0.05, 0.3)) * W)
    amp = k.abs().amax(dim=(-2, -1), keepdim=False) * 0.02 * lam
    k[:, H // 2 + dy, W // 2 + dx] += amp.to(k.dtype)
    out = torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).real
    return out.clamp(0, 1).reshape(B, S, C, H, W)


@register_pixel
class KspaceSpike(PixelAugmentation):
    """One bright off-centre k-space point — herringbone/spike artifact."""
    name = "kspace_spike"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return kspace_spike(x, lam, gen)
