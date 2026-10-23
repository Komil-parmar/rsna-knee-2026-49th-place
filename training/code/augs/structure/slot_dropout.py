"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Blank one non-primary slot — missing-series robustness for the slot pipeline (never slot 0).
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def slot_dropout(x, lam, gen):
    """Blank one non-primary slot (simulates missing series; model's slot
    gating must cope). Never drops slot 0 (SAG fluid-FS, the workhorse)."""
    B = x.shape[0]
    out = x.clone()
    for b in range(B):
        s = 1 + int(_rand(gen, 0, 1) * (x.shape[1] - 1))
        out[b, s] = 0.0
    return out


@register_pixel
class SlotDropout(PixelAugmentation):
    """Blank one non-primary slot — missing-series robustness for the slot pipeline (never slot 0)."""
    name = "slot_dropout"
    category = "structure"

    def apply(self, x, lam, gen):
        return slot_dropout(x, lam, gen)
