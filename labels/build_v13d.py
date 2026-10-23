"""XP-0004 / XP-0006: build labels v13 (plain) and v13D (decoupled teacher), plus the LBL-SYN arm.

Teacher = the 3 S78 families' 5-fold OOF (banks aira_{a57ba940,8bd0d5e4,4a0c41e2}_s1000, probabilities, rows in
common.npz order). v13  = 0.3*v06 + 0.7*mean(P).
v13D: rank-mean E -> S78 decoupling D = E - lam * E @ W.T (W = clipped column correlations of E, as in
scripts/kaggle_submit/delivery_2026-10-01/ensemble_select_analysis/declib.py), then each column is quantile-mapped
onto the sorted values of mean(P)[:, j] (decoupled ORDER, teacher CALIBRATION), then 0.3*v06 + 0.7*T.
v13D_syn: as v13D but Synovitis/PF OA/Lateral OA use r = .9 (0.1*v06 + 0.9*T).
s78T_r9 (XP-0007 DISTILL-EFF, 2026-10-05): the same decoupled teacher T on ALL 12 targets at r = .9
(0.1*v06 + 0.9*T) = soft targets for the cheap stack4 ConvNeXt-336 efficiency student (box recipe s4c).

Diagnostics only (nothing here selects anything): cross-target coupling (mean |corr| over 66 pairs, I-165) and
label-vs-gold macro AUC on the 58 gold rows (I-106 style agreement). Leak note: teacher OOF rows were predicted by
models that saw the other 4 folds, so any student val on these labels is leak-inflated (I-144/I-150).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "notes/mlevolve/oof_bank"
OUT = ROOT / "team/a3/labels"
FAMS = ["aira_a57ba94073_s1000", "aira_8bd0d5e41f_s1000", "aira_4a0c41e2eb_s1000"]
LAM = 0.15
SYN_TARGETS = ["Synovitis", "PF OA", "Lateral OA"]


def corr_weights(E):
    C = np.corrcoef(E.T)
    W = np.zeros((12, 12))
    for j in range(12):
        o = np.delete(np.arange(12), j)
        w = np.clip(C[j, o], 0, None)
        W[j, o] = w / w.sum()
    return W


def coupling(Y):
    C = np.corrcoef(np.asarray(Y, float).T)
    iu = np.triu_indices(12, 1)
    return float(np.abs(C[iu]).mean())


def auc(y, s):
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def gold_macro(Y, G, gold):
    vals = [auc(G[gold, j] > 0.5, Y[gold, j]) for j in range(12)]
    return float(np.nanmean(vals)), vals


def main():
    c = np.load(BANK / "common.npz", allow_pickle=True)
    ids, targets, G, is_gold = c["ids"], list(c["targets"]), c["G"], c["is_gold"].astype(bool)
    P = np.stack([np.load(BANK / "banks" / f"{f}.npz")["P"].astype(np.float64) for f in FAMS])
    assert P.shape == (3, len(ids), 12)

    v06 = pd.read_csv(ROOT / "data/labels/labels_v06.csv").set_index("StudyInstanceUID").loc[ids, targets].to_numpy()
    v11 = pd.read_csv(ROOT / "data/labels/labels_v11_exp108_r7.csv").set_index("StudyInstanceUID").loc[ids, targets].to_numpy()
    folds = pd.read_csv(ROOT / "data/folds.csv").set_index("StudyInstanceUID").loc[ids, "fold"].to_numpy()
    assert (folds == c["fold"]).all(), "bank row order != folds.csv"

    Pm = P.mean(0)
    R = np.stack([np.column_stack([rankdata(P[f][:, j]) / len(ids) for j in range(12)]) for f in range(3)])
    E = R.mean(0)
    D = E - LAM * (E @ corr_weights(E).T)
    T = np.empty_like(D)
    for j in range(12):
        T[np.argsort(D[:, j], kind="stable"), j] = np.sort(Pm[:, j])

    v13 = 0.3 * v06 + 0.7 * Pm
    v13d = 0.3 * v06 + 0.7 * T
    v13d_syn = v13d.copy()
    for t in SYN_TARGETS:
        j = targets.index(t)
        v13d_syn[:, j] = 0.1 * v06[:, j] + 0.9 * T[:, j]
    s78t_r9 = 0.1 * v06 + 0.9 * T

    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, Y in [("v06", v06), ("v11_r7", v11), ("teacher_mean", Pm), ("teacher_decoupled", T),
                    ("v13_r7", v13), ("v13D_r7", v13d), ("v13D_syn9", v13d_syn), ("s78T_r9", s78t_r9)]:
        g, per = gold_macro(Y, G, is_gold)
        rows[name] = {"coupling": round(coupling(Y), 4), "gold58_macro": round(g, 4),
                      "gold58_per_target": dict(zip(targets, [round(x, 4) for x in per]))}
        print(f"{name:18s} coupling {coupling(Y):.4f}  gold58 macro {g:.4f}")
    if not np.isnan(G[is_gold]).all():
        print(f"{'gold truth':18s} coupling {coupling(G[is_gold]):.4f} (58 rows)")
    for name, Y in [("labels_v13_r7", v13), ("labels_v13D_r7", v13d), ("labels_v13D_syn9", v13d_syn),
                    ("labels_s78T_r9", s78t_r9)]:
        df = pd.DataFrame(np.clip(Y, 0, 1), columns=targets)
        df.insert(0, "StudyInstanceUID", ids)
        df.to_csv(OUT / f"{name}.csv", index=False)
    json.dump({"families": FAMS, "lam": LAM, "rows": rows}, open(OUT / "v13_diagnostics.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
