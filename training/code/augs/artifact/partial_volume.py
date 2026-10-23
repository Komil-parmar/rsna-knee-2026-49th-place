"""NEW op (2026-08-23). Partial-volume mixing along the slice axis: each
slice blends toward the mean of its neighbours — simulates thicker slices /
larger slice gaps than the acquisition protocol."""
import torch

from augs.base import PixelAugmentation
from augs.registry import register_pixel


def partial_volume(x, lam, gen):
    a = 0.6 * lam
    xp = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
    xn = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
    return ((1 - a) * x + a * 0.5 * (xp + xn)).clamp(0, 1)


@register_pixel
class PartialVolume(PixelAugmentation):
    """Neighbour-slice blending (α=0.6·λ) — thick-slice / slice-gap simulation."""
    name = "partial_volume"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return partial_volume(x, lam, gen)
