"""Build labels-v04: ensemble of labels-v02, labels-v03, pilkwang v2.

Writes data/labels/labels_v04.csv with, per target:
  <target>        probability-mean ensemble (BCE-trainable soft label)
  <target>__rank  rank-percentile-mean ensemble (pure ranking reference)
Parse-failure rows of v02/v03 (flat 0.5) are masked to NaN so the other
labelers carry those studies. Prints gold-58 AUC for every variant.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa",
        "lateral_oa", "pf_oa", "effusion", "synovitis", "bakers",
        "contusion", "fracture"]

train = pd.read_csv("data/train.csv")
gold = train[train[TARGETS].notna().all(axis=1)].set_index("StudyInstanceUID")


def load_ours(path, fail_mask_col):
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    fail = df.filter(like=fail_mask_col).apply(
        lambda c: c.astype(str).str.contains("PARSE_FAIL")).any(axis=1)
    out = pd.DataFrame(index=df.index)
    for t, k in zip(TARGETS, KEYS):
        out[t] = pd.to_numeric(df[f"{k}__score"], errors="coerce")
    out[fail] = np.nan
    print(f"{path}: {len(df)} rows, {int(fail.sum())} masked (parse fails)")
    return out


def load_pilkwang(path):
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    out = df[TARGETS].apply(pd.to_numeric, errors="coerce")
    print(f"{path}: {len(df)} rows")
    return out


def evaluate(scores, name):
    common = gold.index.intersection(scores.index)
    aucs = []
    for t in TARGETS:
        y = gold.loc[common, t].astype(int)
        p = scores.loc[common, t]
        ok = p.notna()
        aucs.append(roc_auc_score(y[ok], p[ok]))
    macro = float(np.mean(aucs))
    print(f"  {name:34s} macro {macro:.3f} | " +
          " ".join(f"{a:.2f}" for a in aucs))
    return macro


v02 = load_ours("data/labels/labels_v02.csv", "__evidence")
v03 = load_ours("data/labels/labels_v03.csv", "__m")
pil = load_pilkwang(
    "data/public_labels/rsna-knee-llm-labels/report_labels_v2.csv")

idx = v02.index.union(v03.index).union(pil.index)
frames = {"v02": v02, "v03": v03, "pilkwang": pil}

prob = sum(f.reindex(idx)[TARGETS] for f in frames.values())
prob = pd.concat(
    [f.reindex(idx)[TARGETS] for f in frames.values()],
    keys=frames).groupby(level=1).mean()          # NaN-aware mean
rank = pd.concat(
    [f.reindex(idx)[TARGETS].rank(pct=True) for f in frames.values()],
    keys=frames).groupby(level=1).mean()

print(f"\nensemble over {len(idx)} studies "
      f"(targets order: {', '.join(TARGETS)})")
for name, f in frames.items():
    evaluate(f, name + " solo")
m_prob = evaluate(prob, "v04 PROB-mean (training target)")
m_rank = evaluate(rank, "v04 RANK-mean (reference)")

out = prob.copy()
for t in TARGETS:
    out[f"{t}__rank"] = rank[t]
out.index.name = "StudyInstanceUID"
out.reset_index().to_csv("data/labels/labels_v04.csv", index=False)
print(f"\nwrote data/labels/labels_v04.csv ({len(out)} studies); "
      f"prob-macro {m_prob:.3f}, rank-macro {m_rank:.3f}")
