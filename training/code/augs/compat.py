"""Backward-compatibility surface for legacy `mri_augs` consumers.
AUGS maps name -> (callable(x, lam, gen), category) for every PIXEL op
(the original 23 plus the 2026-08-23 additions — FAA search spaces built from
AUGS automatically include the new ops)."""
from augs.policies import apply_subpolicy  # noqa: F401
from augs.registry import PIXEL_REGISTRY

AUGS = {name: (op.apply, op.category) for name, op in
        sorted(PIXEL_REGISTRY.items())}
