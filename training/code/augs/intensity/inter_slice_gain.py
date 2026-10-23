"""NEW op (2026-08-23). Independent per-slice gain jitter — scanner
auto-gain / normalization inconsistency BETWEEN slices of one series (breaks
the assumption that a triplet's channels share one intensity scale)."""
import torch

from augs.base import PixelAugmentation
from augs.registry import register_pixel


def inter_slice_gain(x, lam, gen):
    B, S, C, H, W = x.shape
    g = 1.0 + ((torch.rand((B, S, C, 1, 1), generator=gen) - 0.5)
               * 2 * 0.15 * lam).to(x.device)
    return (x * g).clamp(0, 1)


@register_pixel
class InterSliceGain(PixelAugmentation):
    """Per-slice gain 1±0.15·λ within each token/slot."""
    name = "inter_slice_gain"
    category = "intensity"

    def apply(self, x, lam, gen):
        return inter_slice_gain(x, lam, gen)
