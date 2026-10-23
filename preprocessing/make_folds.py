"""Build data/folds.csv: scanner-fingerprint GroupKFold over all 4,407 studies.

Rationale (I-005, I-017): random CV inflates ~0.05 macro AUC via site
memorization; raw fingerprints over-fragment (3,243 groups), so normalize:
manufacturer spelling variants collapsed, model upper-cased, ImagingFrequency
rounded to integer MHz, coil upper-cased. SoftwareVersions dropped (pure
fragmentation). Gold-58 studies are flagged: they may sit in any fold but must
NEVER be trained on — validation only.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
N_FOLDS = 5
SEED = 20260808

sm = pd.read_parquet("data/meta_kernel/series_meta.parquet")
tr = sm[sm["split"] == "train_series"].copy()
train = pd.read_csv("data/train.csv")


def norm_manu(s):
    s = str(s).upper()
    if "SIEMENS" in s:
        return "SIEMENS"
    if "PHILIPS" in s:
        return "PHILIPS"
    if s.startswith("GE"):
        return "GE"
    if "TOSHIBA" in s or "CANON" in s:
        return "CANON"
    if "HITACHI" in s or "FUJI" in s:
        return "FUJI-HITACHI"
    return s[:12]


tr["manu_n"] = tr["Manufacturer"].map(norm_manu)
tr["model_n"] = (tr["ManufacturerModelName"].fillna("NA").astype(str)
                 .str.upper().str.replace(r"\s+", "", regex=True))
tr["freq_n"] = pd.to_numeric(tr["ImagingFrequency"], errors="coerce") \
    .round(0).fillna(-1).astype(int)
tr["coil_n"] = (tr["ReceiveCoilName"].fillna("NA").astype(str)
                .str.upper().str.replace(r"\s+", "", regex=True))
tr["fp"] = (tr["manu_n"] + "|" + tr["model_n"] + "|" +
            tr["freq_n"].astype(str) + "|" + tr["coil_n"])

study_fp = tr.groupby("StudyInstanceUID")["fp"].agg(
    lambda s: s.value_counts().index[0])
vc = study_fp.value_counts()
cum = vc.cumsum() / len(study_fp)
print(f"normalized fingerprints: {len(vc)} groups over {len(study_fp)} studies")
print(f"largest group: {vc.iloc[0]} studies; top-10 cover {cum.iloc[9]:.1%}, "
      f"top-20 cover {cum.iloc[min(19, len(cum) - 1)]:.1%}")

# GroupKFold balanced by size: assign groups (desc by size) greedily to the
# currently-smallest fold — deterministic and much more even than sklearn's.
rng = np.random.default_rng(SEED)
groups = list(vc.index)
fold_sizes = np.zeros(N_FOLDS, int)
fold_of_group = {}
for g in groups:
    f = int(np.argmin(fold_sizes + rng.random(N_FOLDS) * 0.5))
    fold_of_group[g] = f
    fold_sizes[f] += vc[g]

folds = study_fp.map(fold_of_group).rename("fold").to_frame()
folds["fingerprint"] = study_fp
gold_ids = set(train.loc[train[TARGETS].notna().all(axis=1),
                         "StudyInstanceUID"])
folds["is_gold"] = folds.index.isin(gold_ids).astype(int)
print(f"fold sizes: {folds['fold'].value_counts().sort_index().tolist()}")
print(f"gold per fold: "
      f"{folds[folds['is_gold'] == 1]['fold'].value_counts().sort_index().to_dict()}")

# prevalence balance check using the best public soft labels
pil = pd.read_csv(
    "data/public_labels/rsna-knee-llm-labels/report_labels_v2.csv"
).set_index("StudyInstanceUID")
chk = folds.join(pil[TARGETS], how="left")
prev = chk.groupby("fold")[TARGETS].mean()
print("\nper-fold mean soft prevalence (pilkwang v2) - want rows similar:")
print(prev.round(3).to_string())
spread = (prev.max() - prev.min()).max()
print(f"max prevalence spread across folds: {spread:.3f}")

folds.reset_index().to_csv("data/folds.csv", index=False)
print(f"\nwrote data/folds.csv ({len(folds)} rows)")
