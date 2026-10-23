"""NEW bag op (2026-08-23). Mask out random bag tokens (bag-size jitter) —
the eval-time bag may cover the study differently; the pooling must be robust
to fewer/other tokens. Never leaves < MIN_TOKENS."""
import torch

from augs.base import MIN_TOKENS, BagAugmentation
from augs.registry import register_bag


def token_dropout(imgs, meta, mask, lam, gen):
    if lam <= 0:
        return imgs, meta, mask
    mask = mask.clone()
    B, L = mask.shape
    for b in range(B):
        live = torch.nonzero(mask[b] > 0.5).flatten()
        k = int(len(live) * 0.3 * lam * float(torch.rand(1, generator=gen)))
        k = min(k, max(0, len(live) - MIN_TOKENS))
        if k > 0:
            perm = torch.randperm(len(live), generator=gen)[:k]
            mask[b, live[perm]] = 0.0
    return imgs, meta, mask


@register_bag
class TokenDropout(BagAugmentation):
    """Mask up to 30%·λ random tokens per sample (keeps ≥ MIN_TOKENS)."""
    name = "token_dropout"

    def apply(self, imgs, meta, mask, lam, gen):
        return token_dropout(imgs, meta, mask, lam, gen)
