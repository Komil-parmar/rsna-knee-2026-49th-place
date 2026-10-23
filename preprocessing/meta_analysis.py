"""Analysis of the full DICOM metadata parquet (feeds EDA.md)."""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

sm = pd.read_parquet("data/meta_kernel/series_meta.parquet")
train = pd.read_csv("data/train.csv")
tr = sm[sm["split"] == "train_series"].copy()
print(f"series_meta: {sm.shape}; train series: {len(tr)}; "
      f"studies: {tr['StudyInstanceUID'].nunique()}")

print("\n## Transfer syntaxes (train series)")
print(tr["TransferSyntaxUID"].value_counts(dropna=False).to_string())

print("\n## Scanner / site structure")
for c in ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
          "InstitutionName", "StationName", "BodyPartExamined"]:
    if c in tr.columns:
        vc = tr[c].value_counts(dropna=False)
        print(f"\n{c} (top 8 of {tr[c].nunique(dropna=True)} unique):")
        print(vc.head(8).to_string())
    else:
        print(f"\n{c}: NOT PRESENT")

fp_cols = [c for c in ["Manufacturer", "ManufacturerModelName",
                       "SoftwareVersions", "ImagingFrequency",
                       "ReceiveCoilName"] if c in tr.columns]
print(f"\nfingerprint columns present: {fp_cols}")
tr["fp"] = tr[fp_cols].astype(str).agg("|".join, axis=1)
study_fp = tr.groupby("StudyInstanceUID")["fp"].agg(
    lambda s: s.value_counts().index[0])
print(f"scanner fingerprints: {study_fp.nunique()} unique across "
      f"{len(study_fp)} studies")
cum = study_fp.value_counts().cumsum() / len(study_fp)
print(f"top-10 fingerprints cover {cum.iloc[9]:.1%}, "
      f"top-20 cover {cum.iloc[min(19, len(cum)-1)]:.1%}")

print("\n## Patient-level tags")
for c in ["PatientSex", "PatientAge", "PatientWeight"]:
    if c in tr.columns:
        stud = tr.groupby("StudyInstanceUID")[c].first()
        print(f"{c}: present for {stud.notna().mean():.1%} of studies; "
              f"values: {dict(stud.value_counts(dropna=False).head(6))}")
    else:
        print(f"{c}: NOT PRESENT")

print("\n## Laterality")
for c in ["Laterality", "ImageLaterality"]:
    if c in tr.columns:
        stud = tr.groupby("StudyInstanceUID")[c].apply(
            lambda s: s.dropna().iloc[0] if s.notna().any() else None)
        print(f"{c}: {stud.notna().mean():.1%} of studies; "
              f"{dict(stud.value_counts(dropna=False).head(5))}")

print("\n## Geometry / resolution")
tr["rows_n"] = pd.to_numeric(tr["Rows"], errors="coerce")
tr["cols_n"] = pd.to_numeric(tr["Columns"], errors="coerce")
tr["px"] = pd.to_numeric(tr["PixelSpacing"].str.split("|").str[0],
                         errors="coerce")
tr["thick"] = pd.to_numeric(tr.get("SliceThickness"), errors="coerce")
print(tr[["rows_n", "cols_n", "px", "thick", "n_files",
          "geo_step_median"]].describe(
    percentiles=[.01, .05, .5, .95, .99]).round(2).to_string())

print("\n## Stack integrity")
print(f"series with n_shapes>1 (mixed slice shapes): "
      f"{(tr['n_shapes'] > 1).sum()}")
if "geo_monotonic_by_instance" in tr.columns:
    mono = tr["geo_monotonic_by_instance"]
    print(f"instance-order == spatial-order: {mono.mean():.1%} "
          f"(n={mono.notna().sum()}); non-monotonic: {(mono == False).sum()}")
dup = tr["geo_n_unique_pos"] < tr["geo_n_pos"]
print(f"series with duplicate slice positions (multi-echo/localizer-like): "
      f"{dup.sum()}")
big = tr[tr["n_files"] > 100]
print(f"series with >100 slices: {len(big)}; max {tr['n_files'].max()}")

print("\n## ImagingFrequency vs field strength")
if "ImagingFrequency" in tr.columns:
    f = pd.to_numeric(tr["ImagingFrequency"], errors="coerce")
    print(f"ImagingFrequency: {f.notna().mean():.1%} present; "
          f"quartiles {f.quantile([.25, .5, .75]).round(1).tolist()}")

print("\n## Site fingerprint x label prevalence (gold 58)")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
gold = train[train[TARGETS].notna().all(axis=1)]
gold_fp = study_fp.reindex(gold["StudyInstanceUID"])
print(f"gold studies with fingerprint: {gold_fp.notna().sum()}/58; "
      f"unique fingerprints among gold: {gold_fp.nunique()}")

manu = tr.groupby("StudyInstanceUID")["Manufacturer"].first() \
    if "Manufacturer" in tr.columns else None
if manu is not None:
    print("\nstudies per Manufacturer:")
    print(manu.value_counts(dropna=False).to_string())
