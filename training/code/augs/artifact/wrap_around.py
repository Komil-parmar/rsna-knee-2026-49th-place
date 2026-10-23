"""NEW op (2026-08-23). Wrap-around (aliasing): a strip from one image edge
folds onto the opposite edge — FOV smaller than the anatomy in the
phase-encode direction."""
import torch

from augs._shared import _rand
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def wrap_around(x, lam, gen):
    B, S, C, H, W = x.shape
    w = max(4, int(W * (0.05 + 0.15 * lam) * float(_rand(gen, 0.5, 1.0))))
    a = 0.6 * lam                       # 0 at lam=0 (identity contract)
    out = x.clone()
    if float(_rand(gen, 0, 1)) < 0.5:
        strip = x[..., :, :w]
        out[..., :, W - w:] = ((1 - a) * out[..., :, W - w:] + a * strip).clamp(0, 1)
    else:
        strip = x[..., :, W - w:]
        out[..., :, :w] = ((1 - a) * out[..., :, :w] + a * strip).clamp(0, 1)
    return out


@register_pixel
class WrapAround(PixelAugmentation):
    """Fold an edge strip (5-20%·λ of W) onto the opposite edge (α up to 0.6)."""
    name = "wrap_around"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return wrap_around(x, lam, gen)
