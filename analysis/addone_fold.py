"""Per-fold add-one increment of a candidate member to the v14 trio (XP-0587 pre-registered read).

usage: python addone_fold.py <fold> <cand_oof_pred.npz> [<control_oof_pred.npz>]
Trio = a57-v14 s1000 (XP-0573) + 4a0c-v14 (XP-0576) + 8bd0-v14 (XP-0579) banks, equal rank-mean on the fold's rows.
Reports macro AUC vs labels_v06 for: trio, trio + cand (4 equal members), trio + control, the increments, cand solo,
and mean Spearman of the cand to the trio. oof_pred.npz = run output (ids, pred) for that fold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
c = np.load(ROOT / "notes/mlevolve/oof_bank/common.npz", allow_pickle=True)
ids, folds, T = c["ids"], c["fold"], list(c["targets"])
pos = {s: i for i, s in enumerate(ids)}
v06 = pd.read_csv(ROOT / "data/labels/labels_v06.csv").set_index("StudyInstanceUID").loc[ids, T].to_numpy()


def auc(y, s):
    y = np.asarray(y) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(P, rows):
    return float(np.nanmean([auc(v06[rows, j], P[:, j]) for j in range(12)]))


def ranks(P):
    return np.column_stack([rankdata(P[:, j]) / len(P) for j in range(P.shape[1])])


def fold_pred(path, rows):
    z = np.load(path, allow_pickle=True)
    idx = np.array([pos[s] for s in z["ids"]])
    out = np.full((len(ids), 12), np.nan)
    out[idx] = z["pred"]
    P = out[rows]
    assert not np.isnan(P).any(), f"{path}: missing rows for this fold"
    return P


def main():
    k = int(sys.argv[1])
    rows = np.where(folds == k)[0]
    trio = [np.load(ROOT / f"team/exchange/oof/{x}/oof.npz")["P"][rows] for x in ("XP-0573", "XP-0576", "XP-0579")]
    R3 = sum(ranks(P) for P in trio)
    base = macro(R3, rows)
    cand = fold_pred(sys.argv[2], rows)
    with_c = macro(R3 + ranks(cand), rows)
    sp = np.mean([np.corrcoef(rankdata(cand[:, j]), R3[:, j])[0, 1] for j in range(12)])
    print(f"fold {k}: trio {base:.5f} | cand solo {macro(cand, rows):.5f} | trio+cand {with_c:.5f} "
          f"(inc {with_c - base:+.5f}) | spearman cand-trio {sp:.4f}")
    if len(sys.argv) > 3:
        ctl = fold_pred(sys.argv[3], rows)
        with_k = macro(R3 + ranks(ctl), rows)
        spk = np.mean([np.corrcoef(rankdata(ctl[:, j]), R3[:, j])[0, 1] for j in range(12)])
        print(f"fold {k}: control solo {macro(ctl, rows):.5f} | trio+control {with_k:.5f} (inc {with_k - base:+.5f}) "
              f"| spearman ctl-trio {spk:.4f} || cand inc - control inc = {(with_c - base) - (with_k - base):+.5f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
