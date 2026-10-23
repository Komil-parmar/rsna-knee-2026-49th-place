"""Assemble a 5-fold OOF bank from a3 box runs and deliver it to team/exchange/oof/<XP>/.

usage: python team/a3/scripts/deliver_bank.py XP-0004 a3_a57ba94073_v13D_s1000_f{k} [--compare aira_a57ba94073_s1000]

Pulls runs/<run>/{oof_pred.npz,result.json} for k=0..4 from box 54302922 (Vast proxy), checks every val fold is
complete and disjoint, writes P (4407 x 12, rows in notes/mlevolve/oof_bank/common.npz order, same layout as
oof_bank/banks/*.npz) + per-fold results + pooled macro AUC vs labels_v06 (the shared instrument, FINDINGS a1 15:29).
Test predictions come from the fold checkpoints (ckpt_fold<k>.pt, left on the box for a1's publish_ckpts.py).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "notes/mlevolve/oof_bank"
SSH = ["ssh", "-i", str(Path.home() / ".ssh/vast_ed25519"), "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=20",
       "-p", "17284", "root@ssh4.vast.ai"]
RUNS = "/workspace/rsna-knee/work/runs"


def auc(y, s):
    y = np.asarray(y) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(P, Y, m):
    return float(np.nanmean([auc(Y[m, j], P[m, j]) for j in range(12)]))


def fetch(run, dest, host=("17284", "root@ssh4.vast.ai")):
    """host = (port, user@host); default = a3's box via the Vast proxy. Other boxes: team/exchange/boxes_ssh.json."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("oof_pred.npz", "result.json"):
        cmd = ["scp", "-q", "-i", SSH[2], "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
               "-P", host[0], f"{host[1]}:{RUNS}/{run}/{f}", str(dest / f)]
        for _ in range(4):
            if subprocess.run(cmd).returncode == 0:
                break
        else:
            raise SystemExit(f"could not fetch {run}/{f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xp")
    ap.add_argument("pattern", help="run name with {k} for the fold")
    ap.add_argument("--compare", default="aira_a57ba94073_s1000")
    ap.add_argument("--fold-src", action="append", default=[],
                    help="k=<agent>:<run name> to take fold k from another agent's box (boxes_ssh.json)")
    a = ap.parse_args()

    c = np.load(BANK / "common.npz", allow_pickle=True)
    ids, folds, T = c["ids"], c["fold"], list(c["targets"])
    pos = {s: i for i, s in enumerate(ids)}
    v06 = pd.read_csv(ROOT / "data/labels/labels_v06.csv").set_index("StudyInstanceUID").loc[ids, T].to_numpy()

    out = ROOT / "team/exchange/oof" / a.xp
    raw = out / "runs"
    P = np.full((len(ids), 12), np.nan, np.float32)
    per = {}
    boxes = json.load(open(ROOT / "team/exchange/boxes_ssh.json"))
    src = {}
    for spec in a.fold_src:
        k, rest = spec.split("=", 1)
        agent, run = rest.split(":", 1)
        src[int(k)] = (tuple(boxes[agent]), run)
    for k in range(5):
        host, run = src.get(k, (("17284", "root@ssh4.vast.ai"), a.pattern.format(k=k)))
        fetch(run, raw / run, host)
        z = np.load(raw / run / "oof_pred.npz", allow_pickle=True)
        rid, pred = z["ids"], z["pred"]
        idx = np.array([pos[s] for s in rid])
        assert (folds[idx] == k).all(), f"{run}: rows outside fold {k}"
        assert set(idx) == set(np.where(folds == k)[0]), f"{run}: fold {k} incomplete"
        assert np.isnan(P[idx]).all(), "overlapping folds"
        P[idx] = pred
        r = json.load(open(raw / run / "result.json"))
        per[k] = {"run": run, "fold_auc_v06": round(macro(P, v06, folds == k), 5),
                  "oof_val_auc_report": r.get("oof_val_auc_report"), "gold_auc": r.get("gold_auc"),
                  "selected_epoch": r.get("selected_epoch")}
    assert not np.isnan(P).any()
    pooled = macro(P, v06, np.ones(len(ids), bool))
    ref = np.load(BANK / "banks" / f"{a.compare}.npz")["P"]
    ref_pooled = macro(ref, v06, np.ones(len(ids), bool))
    sp = float(np.mean([np.corrcoef(rankdata(P[:, j]), rankdata(ref[:, j]))[0, 1] for j in range(12)]))
    np.savez(out / "oof.npz", P=P, ids=ids, fold=folds, targets=np.array(T))
    meta = {"xp": a.xp, "pattern": a.pattern, "pooled_auc_v06": round(pooled, 5), "compare": a.compare,
            "compare_pooled_auc_v06": round(ref_pooled, 5), "delta": round(pooled - ref_pooled, 5),
            "mean_spearman_vs_compare": round(sp, 4), "per_fold": per,
            "ckpts": f"box 54302922 {RUNS}/<run>/ckpt_fold<k>.pt (exchange/boxes_ssh.json a3)",
            "note": "rows in notes/mlevolve/oof_bank/common.npz order; same layout as oof_bank/banks/*.npz"}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("pooled_auc_v06", "compare_pooled_auc_v06", "delta",
                                           "mean_spearman_vs_compare")}))
    for k, v in per.items():
        print(k, v)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
