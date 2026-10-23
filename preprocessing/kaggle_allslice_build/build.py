"""Build + push 3 CPU kernels that create the ALL-SLICE cache (Phase 4b):
every series, every slice of every train study, 150 mm crop (central 480 of the
512/160mm prep arrays), 224 px, physically sorted + laterality-normalized as in
prep. Part k = contiguous folds.csv study range from allslice_plan.json.
Outputs per kernel (<20 GB): slices_partK.npy [N,224,224] uint8,
index_partK.csv (one row per slice), meta_partK.json.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = r'''"""All-slice cache part __PART__ (see scripts/kaggle_allslice_build/build.py)."""
import io
import json
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

T0 = time.time()
PART = __PART__
IMG, CROP = 224, 480                       # 150 mm of the 512px/160mm arrays
O = (512 - CROP) // 2
INP = Path("/kaggle/input")
print("input dirs:", sorted(p.name for p in INP.iterdir()), flush=True)
KIT = next(INP.rglob("allslice_plan.json")).parent
WORK = Path("/kaggle/working")


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


plan = json.load(open(KIT / "allslice_plan.json"))
order, counts, parts = plan["order"], plan["counts"], plan["parts"]
lo, hi = parts[PART]
mine = order[lo:hi]
offs = np.concatenate([[0], np.cumsum(counts[lo:hi])])
N = int(offs[-1])
sidx = {s: i for i, s in enumerate(mine)}
ts = pd.read_csv(KIT / "train_series.csv")
meta_series = {r.SeriesInstanceUID: (r.Anatomical_Plane, int(r.Fluid_Sensitive),
                                     int(r.Fat_Suppression))
               for r in ts.itertuples()}
slots = pd.read_csv(KIT / "series_slots.csv")
slot_of = {r.SeriesInstanceUID: r.slot for r in slots.itertuples()}
log(f"part {PART}: studies {lo}..{hi} ({len(mine)}), expected slices {N}")

out = np.lib.format.open_memmap(WORK / f"slices_part{PART}.npy", mode="w+",
                                dtype=np.uint8, shape=(N, IMG, IMG))
rows, lock = [], threading.Lock()
done, missing = [0], []


def process_study(study, z):
    i = sidx[study]
    base, budget = int(offs[i]), int(counts[lo + i])
    cur = 0
    recs = []
    for k, uid in enumerate(sorted(z.files)):
        vol = z[uid]                                   # [n, 512, 512] uint8
        n = vol.shape[0]
        if cur + n > budget:                           # should not happen
            n = max(0, budget - cur)
            vol = vol[:n]
        pl, fl, fs = meta_series.get(uid, ("Unknown", -1, -1))
        for j in range(n):
            sl = cv2.resize(vol[j, O:O + CROP, O:O + CROP], (IMG, IMG),
                            interpolation=cv2.INTER_AREA)
            out[base + cur + j] = sl
            recs.append((base + cur + j, lo + i, study, uid, k, pl, fl, fs,
                         slot_of.get(uid, ""), j, n, int(sl.any())))
        cur += n
    if cur < budget:
        missing.append((study, budget - cur))
    return recs


def process_tar(tar_path):
    cnt = 0
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.name.endswith(".npz"):
                continue
            study = member.name[:-4].split("/")[-1]
            if study not in sidx:
                continue
            buf = tf.extractfile(member).read()
            with np.load(io.BytesIO(buf)) as z:
                recs = process_study(study, z)
            with lock:
                rows.extend(recs)
                done[0] += 1
                if done[0] % 200 == 0:
                    log(f"{done[0]}/{len(mine)} studies, {len(rows)} slices")
            cnt += 1
    return cnt


tars = sorted(INP.rglob("shard*_part*.tar"))
log(f"{len(tars)} tar parts")
assert tars, "prep tars not mounted"
with ThreadPoolExecutor(max_workers=4) as pool:
    n_studies = sum(pool.map(process_tar, tars))
assert n_studies == len(mine), f"processed {n_studies} of {len(mine)} studies"
out.flush()
df = pd.DataFrame(rows, columns=["row", "study_idx", "study", "series_uid",
                                 "series_k", "plane", "fluid", "fatsat",
                                 "slot", "slice_idx", "n_slices", "nonzero"])
df = df.sort_values("row")
df.to_csv(WORK / f"index_part{PART}.csv", index=False)
json.dump({"part": PART, "lo": lo, "hi": hi, "n_rows": N,
           "rows_written": int(len(df)), "img": IMG, "crop_px": CROP,
           "crop_mm": 150.0, "missing": missing, "order": "folds.csv"},
          open(WORK / f"meta_part{PART}.json", "w"))
log(f"DONE part {PART}: {len(df)}/{N} slices written, {len(missing)} studies "
    f"short, {df.nonzero.mean():.3f} nonzero fraction")
'''

prep_sources = [f"komilparmar/rsna-knee-prep-full-s{s}" for s in range(8)]
for part in (0, 1, 2):
    d = HERE / f"part{part}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "allslice_build.py").write_text(
        TEMPLATE.replace("__PART__", str(part)), encoding="utf-8")
    meta = {
        "id": f"komilparmar/rsna-allslice224-part{part}",
        "title": f"rsna-allslice224-part{part}",
        "code_file": "allslice_build.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": ["komilparmar/rsna-knee-trainkit"],
        "competition_sources": [],
        "kernel_sources": prep_sources,
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1),
                                            encoding="utf-8")
    if "--push" in sys.argv:
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)],
                           capture_output=True, text=True)
        print(f"part {part}: {(r.stdout + r.stderr).strip().splitlines()[-1]}")
    else:
        print("built", d)
