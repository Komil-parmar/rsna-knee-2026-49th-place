"""Central registries. Each op file registers exactly one class at import
time; `augs/__init__.py` walks the category packages so importing `augs`
populates both registries."""

PIXEL_REGISTRY = {}       # name -> PixelAugmentation instance
BAG_REGISTRY = {}         # name -> BagAugmentation instance


def register_pixel(cls):
    inst = cls()
    assert inst.name not in PIXEL_REGISTRY, f"duplicate pixel op {inst.name}"
    assert cls.__doc__, f"{inst.name}: docstring required"
    PIXEL_REGISTRY[inst.name] = inst
    return cls


def register_bag(cls):
    inst = cls()
    assert inst.name not in BAG_REGISTRY, f"duplicate bag op {inst.name}"
    assert cls.__doc__, f"{inst.name}: docstring required"
    BAG_REGISTRY[inst.name] = inst
    return cls


def pixel_names(category=None):
    return sorted(n for n, op in PIXEL_REGISTRY.items()
                  if category is None or op.category == category)


def bag_names():
    return sorted(BAG_REGISTRY)


def categories():
    out = {}
    for op in PIXEL_REGISTRY.values():
        out.setdefault(op.category, []).append(op.name)
    if BAG_REGISTRY:
        out["bag"] = sorted(BAG_REGISTRY)
    return out
