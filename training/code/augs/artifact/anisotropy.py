"""NEW op (2026-08-23). Anisotropic in-plane resolution: downsample ONE axis
by up to 3x·λ then upsample — simulates rectangular acquisition matrices /
phase-encode resolution loss (TorchIO RandomAnisotropy analogue)."""
import torch
import torch.nn.functional as F

from augs._shared import _rand
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def anisotropy(x, lam, gen):
    B, S, C, H, W = x.shape
    f = 1.0 + 2.0 * lam * float(_rand(gen, 0.3, 1.0))
    along_h = float(_rand(gen, 0, 1)) < 0.5
    size = (max(16, int(H / f)), W) if along_h else (H, max(16, int(W / f)))
    flat = x.reshape(-1, 1, H, W)
    out = F.interpolate(F.interpolate(flat, size=size, mode="area"),
                        size=(H, W), mode="bilinear", align_corners=False)
    return out.reshape(B, S, C, H, W)


@register_pixel
class Anisotropy(PixelAugmentation):
    """Single-axis resolution loss up to 3x·λ (rectangular matrix / phase-encode)."""
    name = "anisotropy"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return anisotropy(x, lam, gen)
