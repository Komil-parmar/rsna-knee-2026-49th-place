"""NEW bag op (2026-08-23). Zero the fluid/fat-sat metadata flags of random
tokens — those flags come from test_series.csv at inference and can be wrong
or uninformative; the model should degrade gracefully without them."""
import torch

from augs.base import BagAugmentation
from augs.registry import register_bag


def meta_dropout(imgs, meta, mask, lam, gen):
    if lam <= 0:
        return imgs, meta, mask
    meta = meta.clone()
    drop = (torch.rand(meta.shape[:2], generator=gen) < 0.5 * lam).to(
        meta.device)
    meta[..., 1] = torch.where(drop, torch.zeros_like(meta[..., 1]),
                               meta[..., 1])
    meta[..., 2] = torch.where(drop, torch.zeros_like(meta[..., 2]),
                               meta[..., 2])
    return imgs, meta, mask


@register_bag
class MetaDropout(BagAugmentation):
    """Zero fluid+fatsat flags per token with P=0.5·λ."""
    name = "meta_dropout"

    def apply(self, imgs, meta, mask, lam, gen):
        return meta_dropout(imgs, meta, mask, lam, gen)
