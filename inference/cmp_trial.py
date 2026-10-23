"""Compare every member's raw trial output of a candidate with the base kernel's trial (same 3 example studies).
usage: cmp_trial.py <trial dir A> <trial dir B>; R6 is torch.compiled, so |dp| < 1e-3 counts as equal."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
A, B = Path(sys.argv[1]), Path(sys.argv[2])
for f in ("s89c_sub.csv", "raptor_raw.csv", "raptor_sub.csv", "raptor224_sub.csv", "s6_sub.csv", "r7_sub.csv", "submission.csv"):
    if not (A / f).is_file() or not (B / f).is_file():
        print(f"{f:20s} missing in {'A' if not (A / f).is_file() else 'B'}"); continue
    a = pd.read_csv(A / f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").sort_index()
    b = pd.read_csv(B / f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").sort_index()
    same_shape = a.shape == b.shape and list(a.index) == list(b.index)
    d = float(np.nanmax(np.abs(a.to_numpy(float) - b.reindex_like(a).to_numpy(float)))) if same_shape else float("nan")
    cols = [c for c in a.columns if same_shape and not np.allclose(a[c], b[c], atol=1e-3)]
    print(f"{f:20s} shape {a.shape} max|dp| {d:.2e}  columns differing >1e-3: {cols}")
