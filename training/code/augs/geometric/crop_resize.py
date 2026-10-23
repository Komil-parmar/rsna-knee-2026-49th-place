"""Auto-migrated from src/train/mri_augs.py (verbatim body; see
scripts/build_augs_pkg.py). Tighter crop (−20%·λ) resized back — FOV jitter, matches crop-geometry sensitivity (I-027).
"""
import math

import torch
import torch.nn.functional as F

from augs._shared import _rand, _affine
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def crop_resize(x, lam, gen):
    """Random tighter crop then resize back (FOV jitter, up to -20%)."""
    frac = 1.0 - 0.20 * lam * float(_rand(gen, 0, 1))
    return _affine(x, scale=1.0 / max(frac, 1e-3))


@register_pixel
class CropResize(PixelAugmentation):
    """Tighter crop (−20%·λ) resized back — FOV jitter, matches crop-geometry sensitivity (I-027)."""
    name = "crop_resize"
    category = "geometric"

    def apply(self, x, lam, gen):
        return crop_resize(x, lam, gen)
