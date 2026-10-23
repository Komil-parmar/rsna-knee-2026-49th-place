"""Augmentation policies over the registries.

TrivialAugmentMRI  (Müller & Hutter, ICCV 2021): ONE op per sample, magnitude
                   drawn U(0,1). Zero hyper-parameters beyond the op pool.
                   With p_bag > 0 the single op may be a bag op instead.
RandAugmentMRI     (Cubuk 2020): N ops per sample at magnitude U(lo, hi).
apply_subpolicy    FAA-style [(name, prob, magnitude), ...] — byte-compatible
                   with the legacy mri_augs.apply_subpolicy.
All randomness through the passed CPU torch.Generator (C13)."""
import torch

from augs.registry import BAG_REGISTRY, PIXEL_REGISTRY


def _u(gen, lo=0.0, hi=1.0):
    return lo + (hi - lo) * float(torch.rand(1, generator=gen))


class TrivialAugmentMRI:
    def __init__(self, pixel_names=None, bag_names=None, p_bag=0.25):
        self.pixel = [PIXEL_REGISTRY[n] for n in
                      (pixel_names or sorted(PIXEL_REGISTRY))]
        self.bag = [BAG_REGISTRY[n] for n in
                    (bag_names or sorted(BAG_REGISTRY))]
        self.p_bag = p_bag if self.bag else 0.0

    def __call__(self, imgs01, meta, mask, gen):
        """imgs01 float [B, L, C, H, W] in [0,1]; returns same triple."""
        B = imgs01.shape[0]
        for b in range(B):
            lam = _u(gen)
            if self.p_bag > 0 and _u(gen) < self.p_bag:
                op = self.bag[int(torch.randint(len(self.bag), (1,),
                                                generator=gen))]
                i, m, k = op(imgs01[b:b + 1], meta[b:b + 1], mask[b:b + 1],
                             lam, gen)
                imgs01[b:b + 1], meta[b:b + 1], mask[b:b + 1] = i, m, k
            else:
                op = self.pixel[int(torch.randint(len(self.pixel), (1,),
                                                  generator=gen))]
                imgs01[b:b + 1] = op(imgs01[b:b + 1], lam, gen)
        return imgs01, meta, mask


class RandAugmentMRI:
    def __init__(self, n_ops=2, lo=0.3, hi=0.8, pixel_names=None):
        self.n = n_ops
        self.lo, self.hi = lo, hi
        self.pixel = [PIXEL_REGISTRY[n] for n in
                      (pixel_names or sorted(PIXEL_REGISTRY))]

    def __call__(self, imgs01, gen):
        B = imgs01.shape[0]
        for b in range(B):
            for _ in range(self.n):
                op = self.pixel[int(torch.randint(len(self.pixel), (1,),
                                                  generator=gen))]
                imgs01[b:b + 1] = op(imgs01[b:b + 1], _u(gen, self.lo, self.hi),
                                     gen)
        return imgs01


def apply_subpolicy(x, subpolicy, gen):
    """subpolicy: list of (op_name, prob, magnitude). x float [B,S,C,H,W]."""
    for name, p, lam in subpolicy:
        if float(torch.rand(1, generator=gen)) < p:
            x = PIXEL_REGISTRY[name](x, float(lam), gen)
    return x


class FAAPolicyMRI:
    """Searched Fast-AutoAugment policy (Lim 2019), exp-034b output.

    Per sample: with prob p_bag apply ONE random bag op at magnitude U(0,1)
    (identical to TrivialAugmentMRI's bag branch, so a retrain against the
    TA control changes only the pixel branch); otherwise apply ONE random
    searched sub-policy via apply_subpolicy (each op fires with its searched
    prob at its searched magnitude)."""

    def __init__(self, sub_policies, bag_names=None, p_bag=0.0):
        self.subs = [[(n, float(p), float(l)) for n, p, l in sub]
                     for sub in sub_policies]
        assert self.subs, "empty policy"
        for sub in self.subs:
            for n, _, _ in sub:
                assert n in PIXEL_REGISTRY, f"unknown pixel op: {n}"
        self.bag = [BAG_REGISTRY[n] for n in
                    (bag_names or sorted(BAG_REGISTRY))]
        self.p_bag = p_bag if self.bag else 0.0

    @classmethod
    def from_json(cls, path, p_bag=0.0):
        import json
        return cls(json.load(open(path))["sub_policies"], p_bag=p_bag)

    def __call__(self, imgs01, meta, mask, gen):
        """imgs01 float [B, L, C, H, W] in [0,1]; returns same triple."""
        B = imgs01.shape[0]
        for b in range(B):
            if self.p_bag > 0 and _u(gen) < self.p_bag:
                lam = _u(gen)
                op = self.bag[int(torch.randint(len(self.bag), (1,),
                                                generator=gen))]
                i, m, k = op(imgs01[b:b + 1], meta[b:b + 1], mask[b:b + 1],
                             lam, gen)
                imgs01[b:b + 1], meta[b:b + 1], mask[b:b + 1] = i, m, k
            else:
                sub = self.subs[int(torch.randint(len(self.subs), (1,),
                                                  generator=gen))]
                imgs01[b:b + 1] = apply_subpolicy(imgs01[b:b + 1], sub, gen)
        return imgs01, meta, mask
