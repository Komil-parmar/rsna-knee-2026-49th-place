"""Assemble labels_v05.csv from agent part-files; score it; build+score v06.

v05 = Sonnet-5 agent labels (descriptive factored -> code-mapped scores).
v06 = NaN-aware prob-mean(v02, v03, v05, pilkwang) - candidate new training set.
"""
import glob
import json
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
M_SCORE = {"NORM": 0.03, "NM": 0.15, "MINOR": 0.40}
S_SCORE = {0: 0.62, 1: 0.62, 2: 0.80, 3: 0.95}

rows = {}
bad = 0
for f in sorted(glob.glob("data/labels_v05/out/chunk_*_part*.jsonl")):
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            rec = {"StudyInstanceUID": d["uid"],
                   "lang": str(d.get("lang", ""))[:8]}
            for t, k in zip(TARGETS, KEYS):
                item = d[k]
                m = str(item["m"]).strip().upper()
                assert m in ("NM", "NORM", "MINOR", "PRESENT")
                s = item.get("s", 0)
                try:
                    s = max(0, min(3, int(s)))
                except (TypeError, ValueError):
                    s = 0
                rec[t] = (S_SCORE[s if s else 1] if m == "PRESENT"
                          else M_SCORE[m])
                rec[f"{k}__m"] = m
                rec[f"{k}__s"] = s
                rec[f"{k}__e"] = str(item.get("e", ""))[:120]
            rows[d["uid"]] = rec  # last write wins on duplicates
        except Exception:
            bad += 1

v05 = pd.DataFrame(list(rows.values()))
v05.to_csv("data/labels/labels_v05.csv", index=False, encoding="utf-8")
print(f"labels_v05.csv: {len(v05)} studies ({bad} bad lines skipped)")
print("language mix:", v05["lang"].str.lower().value_counts().head(8).to_dict())

train = pd.read_csv("data/train.csv")
gold = train[train[TARGETS].notna().all(axis=1)].set_index("StudyInstanceUID")


def evaluate(scores, name):
    common = gold.index.intersection(scores.index)
    aucs = []
    for t in TARGETS:
        y = gold.loc[common, t].astype(int)
        p = scores.loc[common, t]
        ok = p.notna()
        aucs.append(roc_auc_score(y[ok], p[ok]))
    macro = float(np.mean(aucs))
    print(f"{name:14s} macro {macro:.4f} | " +
          " ".join(f"{a:.2f}" for a in aucs))
    return macro


def load_ours(path, fail_col):
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    fail = df.filter(like=fail_col).apply(
        lambda c: c.astype(str).str.contains("PARSE_FAIL")).any(axis=1)
    out = pd.DataFrame(index=df.index)
    for t, k in zip(TARGETS, KEYS):
        out[t] = pd.to_numeric(df[f"{k}__score"], errors="coerce")
    out[fail] = np.nan
    return out


v02 = load_ours("data/labels/labels_v02.csv", "__evidence")
v03 = load_ours("data/labels/labels_v03.csv", "__m")
pil = pd.read_csv("data/public_labels/rsna-knee-llm-labels/"
                  "report_labels_v2.csv").set_index("StudyInstanceUID")[
    TARGETS].apply(pd.to_numeric, errors="coerce")
v05s = v05.set_index("StudyInstanceUID")[TARGETS]

print(f"\n(targets: {', '.join(TARGETS)})")
evaluate(v02, "v02")
evaluate(v03, "v03")
evaluate(pil, "pilkwang")
m05 = evaluate(v05s, "v05 (Sonnet)")

idx = v02.index.union(v03.index).union(pil.index).union(v05s.index)
frames = [f.reindex(idx)[TARGETS] for f in (v02, v03, pil, v05s)]
v06 = pd.concat(frames, keys=range(4)).groupby(level=1).mean()
m06 = evaluate(v06, "v06 = mean(4)")
v06.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_csv(
    "data/labels/labels_v06.csv", index=False, encoding="utf-8")
print(f"\nwrote data/labels/labels_v06.csv ({len(v06)} studies)")
print(f"SUMMARY v05_macro={m05:.4f} v06_macro={m06:.4f} (v04 was 0.895)")
