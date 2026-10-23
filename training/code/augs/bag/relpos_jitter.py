"""NEW bag op (2026-08-23). Gaussian jitter on the relative-position channel
of the metadata — slice numbering/coverage varies between scanners, so the
model should not over-trust exact depth positions (cf. the flip-bug lesson
I-058: laterality read off a scalar feature)."""
import torch

from augs.base import BagAugmentation
from augs.registry import register_bag


def relpos_jitter(imgs, meta, mask, lam, gen):
    if lam <= 0:
        return imgs, meta, mask
    meta = meta.clone()
    noise = ((torch.rand(meta.shape[:2], generator=gen) - 0.5)
             * 2 * 0.08 * lam).to(meta.device)
    meta[..., 3] = (meta[..., 3] + noise).clamp(0, 1)
    return imgs, meta, mask


@register_bag
class RelposJitter(BagAugmentation):
    """relpos ± 0.08·λ uniform jitter, clamped to [0, 1]."""
    name = "relpos_jitter"

    def apply(self, imgs, meta, mask, lam, gen):
        return relpos_jitter(imgs, meta, mask, lam, gen)
