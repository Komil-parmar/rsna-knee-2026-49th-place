"""NEW op (2026-08-23). Zipper artifact: a narrow stripe of alternating
noise across the image — RF interference leaking into the receiver."""
import torch

from augs._shared import _rand
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def zipper(x, lam, gen):
    B, S, C, H, W = x.shape
    out = x.clone()
    w = 1 + int(2 * lam)
    amp = 0.3 * lam
    horiz = float(_rand(gen, 0, 1)) < 0.5
    if horiz:
        pos = int(float(_rand(gen, 0.1, 0.9)) * (H - w))
        n = ((torch.rand((w, W), generator=gen) - 0.5) * 2 * amp).to(x.device)
        out[..., pos:pos + w, :] = (out[..., pos:pos + w, :] + n).clamp(0, 1)
    else:
        pos = int(float(_rand(gen, 0.1, 0.9)) * (W - w))
        n = ((torch.rand((H, w), generator=gen) - 0.5) * 2 * amp).to(x.device)
        out[..., :, pos:pos + w] = (out[..., :, pos:pos + w] + n).clamp(0, 1)
    return out


@register_pixel
class Zipper(PixelAugmentation):
    """RF-interference stripe (width 1-3, amplitude 0.3·λ), random row/col."""
    name = "zipper"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return zipper(x, lam, gen)
