"""NEW bag op (2026-08-23). Drop ALL tokens of one series-type (same
plane/fluid/fatsat signature) from the bag via the mask — trains robustness to
the test set's protocol variation (studies have 1-13 series; a sequence type
present in training may be absent at test time). Never leaves < MIN_TOKENS."""
import torch

from augs.base import MIN_TOKENS, BagAugmentation
from augs.registry import register_bag


def series_dropout(imgs, meta, mask, lam, gen):
    if lam <= 0:
        return imgs, meta, mask
    mask = mask.clone()
    B = mask.shape[0]
    for b in range(B):
        if float(torch.rand(1, generator=gen)) > 0.7 * lam:
            continue
        live = mask[b] > 0.5
        keys = meta[b, :, :3].round().long()
        uniq = torch.unique(keys[live], dim=0)
        if len(uniq) < 2:
            continue
        pick = uniq[int(torch.randint(len(uniq), (1,), generator=gen))]
        drop = (keys == pick).all(-1) & live
        if int(live.sum() - drop.sum()) >= MIN_TOKENS:
            mask[b, drop] = 0.0
    return imgs, meta, mask


@register_bag
class SeriesDropout(BagAugmentation):
    """Mask out one whole series-type from the bag (P=0.7·λ per sample)."""
    name = "series_dropout"

    def apply(self, imgs, meta, mask, lam, gen):
        return series_dropout(imgs, meta, mask, lam, gen)
