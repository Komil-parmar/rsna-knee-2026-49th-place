"""Compatibility shim (2026-08-23): canonical implementations moved to the
`augs` package (src/augs locally; the augs/ folder next to the trainers on
Colab/Kaggle). Existing imports keep working:
    from mri_augs import AUGS, apply_subpolicy
New code should import from `augs` directly."""
from augs.compat import AUGS  # noqa: F401
from augs.policies import apply_subpolicy  # noqa: F401
