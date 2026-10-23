"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Magnitude-image noise |x+n₁+i·n₂|, σ=0.06·λ — the physically correct MRI noise model.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def rician_noise(x, lam, gen):
    """MRI magnitude noise: |x + n1 + i n2| with Gaussian n."""
    sig = 0.06 * lam
    n1 = (torch.randn(x.shape, generator=gen) * sig).to(x.device)
    n2 = (torch.randn(x.shape, generator=gen) * sig).to(x.device)
    return torch.sqrt((x + n1) ** 2 + n2 ** 2).clamp(0, 1)


@register_pixel
class RicianNoise(PixelAugmentation):
    """Magnitude-image noise |x+n₁+i·n₂|, σ=0.06·λ — the physically correct MRI noise model."""
    name = "rician_noise"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return rician_noise(x, lam, gen)
