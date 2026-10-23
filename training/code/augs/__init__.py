"""Knee-MRI augmentation library (canonical since 2026-08-23; the old
src/train/mri_augs.py is a shim over this package). See README.md."""
import importlib
import pathlib
import pkgutil

_dir = pathlib.Path(__file__).parent
for _sub in ("geometric", "intensity", "artifact", "structure", "bag"):
    for _m in pkgutil.iter_modules([str(_dir / _sub)]):
        importlib.import_module(f"{__name__}.{_sub}.{_m.name}")

from augs.registry import (BAG_REGISTRY, PIXEL_REGISTRY, bag_names,  # noqa
                           categories, pixel_names)
from augs.policies import (FAAPolicyMRI, RandAugmentMRI,  # noqa
                           TrivialAugmentMRI, apply_subpolicy)
from augs.compat import AUGS  # noqa
