"""Base classes for the knee-MRI augmentation library.

Two families:

PixelAugmentation — operates on image tensors.
    apply(x, lam, gen) -> tensor
    x   : float tensor [B, S, C, H, W] in [0, 1]. For the slot pipeline
          S = 6 slots, C = 9 slices; for slice-bag input S = L bag tokens,
          C = 3 triplet slices. Ops must not assume specific S or C.
    lam : magnitude in [0, 1] (FAA convention; each op maps it to its own
          physical range, documented in the op's docstring).
    gen : torch.Generator on CPU — ALL randomness must be drawn on CPU via
          this generator and moved to x.device (debug-log C13: CPU generators
          cannot feed CUDA factory functions).
    Contract: output has the same shape/dtype/device as x, values stay finite
    (ops that can exceed [0, 1] must clamp); lam == 0 must be identity or
    near-identity; the op is deterministic given (x, lam, gen state).

BagAugmentation — operates on slice-bag structure/metadata, not pixels.
    apply(imgs, meta, mask, lam, gen) -> (imgs, meta, mask)
    imgs [B, L, 3, H, W] (any dtype — bag ops must not touch pixel values
    unless documented), meta [B, L, 4] = (plane_id, fluid, fatsat, relpos),
    mask [B, L] with 1 = token present. Contract: shapes preserved; at least
    MIN_TOKENS tokens stay unmasked per sample; deterministic given inputs
    and gen state.

Every concrete op lives in its own file under a category folder and registers
itself via `augs.registry.register_pixel` / `register_bag`, so a misbehaving
augmentation is traceable to exactly one small file.
"""

MIN_TOKENS = 8            # bag ops must never leave fewer live tokens


class PixelAugmentation:
    name = "?"
    category = "?"

    def apply(self, x, lam, gen):
        raise NotImplementedError

    def __call__(self, x, lam, gen):
        return self.apply(x, lam, gen)

    def __repr__(self):
        return f"<PixelAug {self.name} ({self.category})>"


class BagAugmentation:
    name = "?"
    category = "bag"

    def apply(self, imgs, meta, mask, lam, gen):
        raise NotImplementedError

    def __call__(self, imgs, meta, mask, lam, gen):
        return self.apply(imgs, meta, mask, lam, gen)

    def __repr__(self):
        return f"<BagAug {self.name}>"
