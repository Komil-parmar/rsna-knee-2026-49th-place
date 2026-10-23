# augs — knee-MRI augmentation library

Canonical home of every training-time augmentation (since 2026-08-23; the old
`src/train/mri_augs.py` monolith is now a shim over this package). One op =
one file = one registered class, so any misbehaving augmentation traces to a
single small file.

## Layout
- `base.py` — contracts (tensor shapes, magnitude semantics, determinism, C13
  generator rule, `MIN_TOKENS` guard for bag ops).
- `registry.py` — `PIXEL_REGISTRY` / `BAG_REGISTRY`; ops self-register.
- `geometric/ intensity/ artifact/ structure/` — pixel ops
  (`apply(x, lam, gen)`, x float `[B,S,C,H,W]` in [0,1], lam in [0,1]).
- `bag/` — slice-bag structural/metadata ops
  (`apply(imgs, meta, mask, lam, gen)`), unique to the bag architecture.
- `policies.py` — `TrivialAugmentMRI`, `RandAugmentMRI`, FAA
  `apply_subpolicy`.
- `compat.py` — legacy `AUGS` dict; FAA search spaces built from it
  automatically include new ops.

## Provenance
23 ops were mechanically AST-migrated (verbatim bodies) from `mri_augs.py` by
`scripts/build_augs_pkg.py`; equivalence against the frozen legacy snapshot
(`tests/fixtures/mri_augs_legacy.py`) is enforced by
`tests/test_augs_lib.py`. 12 ops added 2026-08-23 by
`scripts/build_augs_pkg2.py` (8 MRI-physics pixel ops, 4 bag ops).

## Adding an op
Copy any small op file, keep the contract in `base.py`, give the class a
one-line clinical rationale docstring, register it, and add it to the
per-op test table in `tests/test_augs_lib.py`. Never draw randomness except
through the passed CPU generator.
