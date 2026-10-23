"""NEW op (2026-08-23). Chemical-shift displacement: bright (fat-like)
structures are displaced 1-3 px along the frequency-encode axis and blended —
the classic fat/water shift artifact at tissue boundaries."""
import torch

from augs._shared import _rand
from augs.base import PixelAugmentation
from augs.registry import register_pixel


def chemical_shift(x, lam, gen):
    s = 1 + int(round(2 * lam * float(_rand(gen, 0, 1))))
    rolled = torch.roll(x, shifts=s, dims=-1)
    m = (rolled > 0.55).to(x.dtype)
    a = 0.5 * lam
    return ((1 - a * m) * x + a * m * rolled).clamp(0, 1)


@register_pixel
class ChemicalShift(PixelAugmentation):
    """Bright-structure shift 1-3 px along W, masked blend (α=0.5·λ)."""
    name = "chemical_shift"
    category = "mri-artifact"

    def apply(self, x, lam, gen):
        return chemical_shift(x, lam, gen)
