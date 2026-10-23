"""XP-0573: build the next label round (v14) from any set of OOF banks, with the v13D construction.

usage: python team/a3/scripts/build_round.py --name labels_v14_r7 --bank team/exchange/oof/XP-0004/oof.npz \
           --bank <more oof.npz ...> [--r 0.7] [--lam 0.15]

Each bank is an npz with P (4407 x 12 probabilities) in notes/mlevolve/oof_bank/common.npz row order (the layout
deliver_bank.py writes). Banks of the same family (seeds) are averaged in probability space first so each family
counts once; pass them as --bank a.npz+b.npz.
Teacher T: per-family rank, equal family rank-mean E, S78 decoupling D = E - lam * E @ W.T (W = clipped column
correlations, as in build_v13d.py), then each column quantile-mapped onto the sorted mean-probability distribution.
Label = (1 - r) * v06 + r * T. Prints coupling (mean |corr| over 66 pairs; gold truth .197) and gold-58 agreement for
v06, v11, v13D and the new labels, and writes <name>.csv + <name>_diagnostics.json to team/a3/labels/.
Diagnostics only; the decision is a matched LB pair (teacher-derived labels are leak-inflated on report-val).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v13d import coupling, corr_weights, gold_macro  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "notes/mlevolve/oof_bank"
OUT = ROOT / "team/a3/labels"


def load_family(spec, n):
    Ps = [np.load(ROOT / p if not Path(p).is_absolute() else p)["P"].astype(np.float64) for p in spec.split("+")]
    for P in Ps:
        assert P.shape == (n, 12) and not np.isnan(P).any(), spec
    return np.mean(Ps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bank", action="append", required=True, help="one family; seeds joined with '+'")
    ap.add_argument("--r", type=float, default=0.7)
    ap.add_argument("--lam", type=float, default=0.15)
    a = ap.parse_args()

    c = np.load(BANK / "common.npz", allow_pickle=True)
    ids, T_names, G, gold = c["ids"], list(c["targets"]), c["G"], c["is_gold"].astype(bool)
    n = len(ids)
    fams = [load_family(b, n) for b in a.bank]
    Pm = np.mean(fams, axis=0)
    E = np.mean([np.column_stack([rankdata(F[:, j]) / n for j in range(12)]) for F in fams], axis=0)
    D = E - a.lam * (E @ corr_weights(E).T)
    T = np.empty_like(D)
    for j in range(12):
        T[np.argsort(D[:, j], kind="stable"), j] = np.sort(Pm[:, j])

    def lab(name):
        return pd.read_csv(ROOT / "data/labels" / name).set_index("StudyInstanceUID").loc[ids, T_names].to_numpy()

    v06 = lab("labels_v06.csv")
    new = (1 - a.r) * v06 + a.r * T
    v13d = pd.read_csv(OUT / "labels_v13D_r7.csv").set_index("StudyInstanceUID").loc[ids, T_names].to_numpy()
    rows = {}
    for name, Y in [("v06", v06), ("v11_r7", lab("labels_v11_exp108_r7.csv")), ("v13D_r7", v13d),
                    ("teacher_mean", Pm), ("teacher_decoupled", T), (a.name, new)]:
        g, per = gold_macro(Y, G, gold)
        rows[name] = {"coupling": round(coupling(Y), 4), "gold58_macro": round(g, 4),
                      "gold58_per_target": dict(zip(T_names, [round(x, 4) for x in per]))}
        print(f"{name:18s} coupling {coupling(Y):.4f}  gold58 macro {g:.4f}")
    df = pd.DataFrame(np.clip(new, 0, 1), columns=T_names)
    df.insert(0, "StudyInstanceUID", ids)
    df.to_csv(OUT / f"{a.name}.csv", index=False)
    json.dump({"banks": a.bank, "r": a.r, "lam": a.lam, "rows": rows},
              open(OUT / f"{a.name}_diagnostics.json", "w"), indent=1)
    print("wrote", OUT / f"{a.name}.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
