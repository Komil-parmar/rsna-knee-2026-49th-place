"""NEW op (2026-08-23). Random monotone piecewise-linear intensity remap
through (0,0)-(k1,y1)-(k2,y2)-(1,1) — protocol/vendor tone-curve domain shift
beyond what gamma/contrast express."""
import torch

from augs._shared import _rand
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def histogram_remap(x, lam, gen):
    k1 = float(_rand(gen, 0.2, 0.5))
    k2 = float(_rand(gen, 0.5, 0.8))
    y1 = min(max(k1 + float(_rand(gen, -1, 1)) * 0.25 * lam, 0.02), 0.95)
    y2 = min(max(k2 + float(_rand(gen, -1, 1)) * 0.25 * lam, y1 + 0.02), 0.98)
    xc = x.clamp(0, 1)
    seg1 = xc / k1 * y1
    seg2 = y1 + (xc - k1) / (k2 - k1) * (y2 - y1)
    seg3 = y2 + (xc - k2) / (1 - k2) * (1 - y2)
    out = torch.where(xc < k1, seg1, torch.where(xc < k2, seg2, seg3))
    return out.clamp(0, 1)


@register_pixel
class HistogramRemap(PixelAugmentation):
    """Monotone 2-knot piecewise-linear intensity curve (knot shift ±0.25·λ)."""
    name = "histogram_remap"
    category = "intensity"

    def apply(self, x, lam, gen):
        return histogram_remap(x, lam, gen)
