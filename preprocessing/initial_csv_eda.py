"""First-pass EDA on the competition CSVs (no DICOMs). Output feeds EDA.md."""
import sys
import re
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]

train = pd.read_csv("data/train.csv")
series = pd.read_csv("data/train_series.csv")
test = pd.read_csv("data/test.csv")
test_series = pd.read_csv("data/test_series.csv")

print("## train.csv")
print(f"shape: {train.shape}")
print(f"columns: {list(train.columns)}")
print(f"unique studies: {train['StudyInstanceUID'].nunique()}")

present_targets = [t for t in TARGETS if t in train.columns]
missing_targets = [t for t in TARGETS if t not in train.columns]
print(f"target cols present: {len(present_targets)}; missing: {missing_targets}")

lab = train[present_targets]
n_all_labeled = lab.notna().all(axis=1).sum()
n_any_labeled = lab.notna().any(axis=1).sum()
print(f"rows with ALL {len(present_targets)} labels: {n_all_labeled} "
      f"({n_all_labeled / len(train):.1%})")
print(f"rows with ANY label: {n_any_labeled} ({n_any_labeled / len(train):.1%})")

print("\nper-label: n_labeled, n_pos, prevalence (among labeled)")
for t in present_targets:
    col = train[t]
    n = col.notna().sum()
    pos = (col == 1).sum()
    print(f"  {t:18s} n={n:5d}  pos={pos:5d}  prev={pos / max(n, 1):.3f}")

labeled_mask = lab.notna().all(axis=1)
if labeled_mask.sum() > 1:
    co = lab[labeled_mask].astype(float).corr().round(2)
    print("\nlabel correlation (labeled subset):")
    print(co.to_string())

print("\n## Reports")
rep = train["Report"].fillna("")
print(f"non-empty reports: {(rep.str.len() > 0).sum()} / {len(train)}")
print(f"report length chars: min={rep.str.len().min()}, "
      f"median={int(rep.str.len().median())}, mean={int(rep.str.len().mean())}, "
      f"max={rep.str.len().max()}")

MARKERS = {
    "english": r"\b(the|and|with|of|is|are|no evidence|intact)\b",
    "spanish": r"\b(el|la|los|las|de|del|con|sin|derecha|izquierda|rodilla)\b",
    "portuguese": r"\b(joelho|não|com|sem|direito|esquerdo|ausência)\b",
    "german": r"\b(der|die|das|und|kein|keine|Knie|rechts|links)\b",
    "french": r"\b(le|la|les|du|genou|droit|gauche|sans|avec)\b",
    "italian": r"\b(il|lo|gli|ginocchio|destro|sinistro|senza)\b",
    "dutch": r"\b(de|het|knie|rechter|linker|geen)\b",
    "turkish": r"\b(diz|sağ|sol|ve|yok|izlenmiştir|düzeyinde)\b",
}
counts = {}
for lang, rx in MARKERS.items():
    counts[lang] = rep.str.contains(rx, flags=re.IGNORECASE, regex=True).sum()
print("language marker hits (non-exclusive):",
      {k: int(v) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])})
non_ascii = rep.apply(lambda s: bool(re.search(r"[^\x00-\x7F]", s)))
print(f"reports containing non-ASCII chars: {non_ascii.sum()} ({non_ascii.mean():.1%})")

print("\nsample report (first labeled study, first 600 chars):")
idx = train.index[labeled_mask][0] if labeled_mask.sum() else 0
print(repr(train.loc[idx, "Report"])[:600])
print("\nsample report (first UNlabeled study, first 600 chars):")
unl = train.index[~labeled_mask]
if len(unl):
    print(repr(train.loc[unl[0], "Report"])[:600])

print("\n## train_series.csv")
print(f"shape: {series.shape}; unique series: {series['SeriesInstanceUID'].nunique()}; "
      f"studies: {series['StudyInstanceUID'].nunique()}")
per_study = series.groupby("StudyInstanceUID").size()
print(f"series per study: min={per_study.min()}, median={per_study.median()}, "
      f"mean={per_study.mean():.1f}, max={per_study.max()}")
print("\nplane distribution:")
print(series["Anatomical_Plane"].value_counts(dropna=False).to_string())
print("\nfluid/fatsat combos:")
print(series.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"])
      .size().to_string())

combo = series.assign(
    slot=series["Anatomical_Plane"].astype(str) + "_F" +
    series["Fluid_Sensitive"].astype(str) + "_S" + series["Fat_Suppression"].astype(str))
cov = combo.groupby("StudyInstanceUID")["slot"].apply(set)
for name, plane, fl, fs in [("SAG fluid+fatsat", "Sagittal", 1, 1),
                            ("COR fluid+fatsat", "Coronal", 1, 1),
                            ("AX  fluid+fatsat", "Axial", 1, 1),
                            ("SAG fluid nofs", "Sagittal", 1, 0),
                            ("COR T1-ish", "Coronal", 0, 0),
                            ("SAG T1-ish", "Sagittal", 0, 0)]:
    key = f"{plane}_F{fl}_S{fs}"
    frac = cov.apply(lambda s: key in s).mean()
    print(f"studies with {name}: {frac:.1%}")

print("\n## test")
print(f"test.csv shape: {test.shape}")
print(f"test_series.csv shape: {test_series.shape}")
print(test_series.to_string())

print("\n## PatientSex check")
print(f"'PatientSex' in train.csv columns: {'PatientSex' in train.columns}")
