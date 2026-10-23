"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Blend a translated copy (shift 10·λ px, α=0.25·λ) — patient-motion phase ghost.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def motion_ghost(x, lam, gen):
    """Ghosting: blend a translated copy (patient-motion phase ghost)."""
    shift = max(1, int(10 * lam))
    ghost = torch.roll(x, shifts=shift, dims=-2)
    a = 0.25 * lam
    return ((1 - a) * x + a * ghost).clamp(0, 1)


@register_pixel
class MotionGhost(PixelAugmentation):
    """Blend a translated copy (shift 10·λ px, α=0.25·λ) — patient-motion phase ghost."""
    name = "motion_ghost"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return motion_ghost(x, lam, gen)
