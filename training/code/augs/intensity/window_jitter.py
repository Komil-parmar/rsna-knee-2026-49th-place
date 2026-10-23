"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Jitter the 1–99 pct intensity window used at prep — re-windowing robustness.
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def window_jitter(x, lam, gen):
    """Re-window: jitter the intensity clip range (train prep used 1-99pct)."""
    lo = float(_rand(gen, 0, 1)) * 0.08 * lam
    hi = 1.0 - float(_rand(gen, 0, 1)) * 0.08 * lam
    return ((x - lo) / max(hi - lo, 1e-3)).clamp(0, 1)


@register_pixel
class WindowJitter(PixelAugmentation):
    """Jitter the 1–99 pct intensity window used at prep — re-windowing robustness."""
    name = "window_jitter"
    category = "intensity"

    def apply(self, x, lam, gen):
        return window_jitter(x, lam, gen)
