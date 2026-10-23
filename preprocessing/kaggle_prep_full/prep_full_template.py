"""RSNA Knee - FULL train conversion, shard __SHARD__ of __NSHARDS__ (512 px).

Pipeline validated in scripts/kaggle_prep_proto (63 studies, 0 failures; see
EDA.md "Prototype conversion results"). Train is 100% uncompressed Explicit VR
LE (I-015) so no pixel-decoder plugins are needed and internet stays off.

Per study: decode -> spatial sort -> laterality->left-knee -> per-axis 160 mm
crop/pad -> 1-99 pct norm over good slices -> 512^2 uint8 -> per-study npz
(+ json meta), written to /kaggle/tmp, then packed into ~1.8 GB tar parts in
/kaggle/working (stays under the 20 GB output cap; sidesteps file-count limits).

Outputs: shard__SHARD___part*.tar, manifest.json, stats.json, qa montages (x2).
"""
import json
import os
import tarfile
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

T0 = time.time()
SHARD_IDX = __SHARD__
N_SHARDS = __NSHARDS__
IMG = 512
CROP_MM = 160.0
TIME_BUDGET = 11.0 * 3600
TAR_PART_BYTES = 1.8 * 2**30
LAT_MIN_OFFSET_MM = 5.0
SORT_TAGS = ["ImagePositionPatient", "InstanceNumber", "Laterality",
             "ImageLaterality", "PixelSpacing"]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_root():
    for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection")]:
        if (c / "train_series").is_dir():
            return c
    raise FileNotFoundError("competition mount not found")


ROOT = find_root()
TMP = Path("/kaggle/tmp/prep")
WORK = Path("/kaggle/working")
QA = WORK / "qa"
TMP.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

train_series = pd.read_csv(ROOT / "train_series.csv")
VOCAB = {"Sagittal", "Coronal", "Axial"}
assert set(train_series["Anatomical_Plane"].dropna().unique()) <= VOCAB
PLANE = dict(zip(train_series["SeriesInstanceUID"].astype(str),
                 train_series["Anatomical_Plane"]))


def warn(info, key, example=None):
    info["warnings"][key] = info["warnings"].get(key, 0) + 1
    if example and len(info["warn_examples"]) < 3:
        info["warn_examples"].append(f"{key}:{example}"[:120])


def read_headers(sdir, files):
    out = []
    for name in files:
        h = {"file": name}
        try:
            d = pydicom.dcmread(os.path.join(sdir, name),
                                stop_before_pixels=True, force=True,
                                specific_tags=SORT_TAGS)
        except Exception as exc:
            h["err"] = str(exc)[:80]
            out.append(h)
            continue
        try:
            ipp = getattr(d, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) >= 3:
                h["ipp"] = np.asarray([float(v) for v in ipp[:3]])
        except Exception:
            pass
        try:
            inst = getattr(d, "InstanceNumber", None)
            h["instance"] = None if inst is None else float(inst)
        except Exception:
            h["instance"] = None
        try:
            for t in ("ImageLaterality", "Laterality"):
                v = getattr(d, t, None)
                if v and str(v).strip().upper()[:1] in ("L", "R"):
                    h["lat"] = str(v).strip().upper()[:1]
                    break
        except Exception:
            pass
        try:
            ps = getattr(d, "PixelSpacing", None)
            if ps is not None and len(ps) >= 2:
                h["ps"] = (float(ps[0]), float(ps[1]))
        except Exception:
            pass
        out.append(h)
    return out


def sort_order(headers):
    pos = [h for h in headers if "ipp" in h]
    if len(pos) >= max(2, int(0.8 * len(headers))):
        xyz = np.stack([h["ipp"] for h in pos])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        fallback = float(np.median(xyz[:, axis]))
        key = lambda h: (float(h["ipp"][axis]) if "ipp" in h else fallback,
                         h["instance"] if h.get("instance") is not None
                         else float("inf"))
    elif sum(h.get("instance") is not None for h in headers) >= 2:
        key = lambda h: (h["instance"] if h.get("instance") is not None
                         else float("inf"))
    else:
        key = lambda h: h["file"]
    return sorted(headers, key=key)


def decode_series(sdir, headers, info):
    planes, shapes = [], []
    for h in headers:
        try:
            ds = pydicom.dcmread(os.path.join(sdir, h["file"]), force=True)
            if info["syntax"] is None:
                info["syntax"] = str(getattr(getattr(ds, "file_meta", None),
                                             "TransferSyntaxUID", ""))
            a = ds.pixel_array
            if a.ndim != 2:
                warn(info, "non2d", f"{h['file']}:{a.shape}")
                planes.append(None)
                shapes.append(None)
                continue
            a = a.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * slope + inter
            photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
            info["photometric"] = photo
            if photo == "MONOCHROME1":
                a = -a
                warn(info, "monochrome1")
            planes.append(a)
            shapes.append(a.shape)
        except Exception as exc:
            warn(info, "decode_fail", f"{h['file']}:{str(exc)[:60]}")
            planes.append(None)
            shapes.append(None)
    good_shapes = [s for s in shapes if s is not None]
    if not good_shapes:
        return None, None
    shp = Counter(good_shapes).most_common(1)[0][0]
    fixed, good = [], []
    for p in planes:
        if p is None:
            fixed.append(np.zeros(shp, np.float32))
            good.append(False)
        elif p.shape != shp:
            warn(info, "shape_mismatch", str(p.shape))
            fixed.append(np.zeros(shp, np.float32))
            good.append(False)
        else:
            fixed.append(p)
            good.append(True)
    vol = np.stack(fixed)
    del planes, fixed
    pss = [h["ps"] for h in headers if "ps" in h]
    if pss:
        info["px_r"] = float(np.median([p[0] for p in pss]))
        info["px_c"] = float(np.median([p[1] for p in pss]))
        if abs(info["px_r"] - info["px_c"]) > 1e-3:
            warn(info, "anisotropic_ps", f"{info['px_r']}x{info['px_c']}")
    return vol, np.array(good, bool)


def crop_pad_axis(vol, px, axis, info, tag):
    if not (px and np.isfinite(px) and px > 0):
        warn(info, f"crop_skipped_{tag}")
        return vol, 0
    want = int(round(CROP_MM / px))
    n = vol.shape[axis]
    if want < 16:
        warn(info, f"crop_skipped_{tag}", f"want={want}")
        return vol, 0
    if want < n:
        start = (n - want) // 2
        sl = [slice(None)] * 3
        sl[axis] = slice(start, start + want)
        return vol[tuple(sl)], 0
    if want > n:
        warn(info, f"pad_fov_{tag}", f"{n}->{want}")
        return vol, want - n
    return vol, 0


def process_series(vol, good, info):
    vol, pad_r = crop_pad_axis(vol, info.get("px_r"), 1, info, "r")
    vol, pad_c = crop_pad_axis(vol, info.get("px_c"), 2, info, "c")
    gv = vol[good]
    lo, hi = np.percentile(gv, [1, 99])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if pad_r or pad_c:
        vol = np.pad(vol, ((0, 0),
                           (pad_r // 2, pad_r - pad_r // 2),
                           (pad_c // 2, pad_c - pad_c // 2)))
    interp = (cv2.INTER_AREA if min(vol.shape[1:]) > IMG else cv2.INTER_LINEAR)
    out = np.empty((vol.shape[0], IMG, IMG), np.float32)
    for i in range(vol.shape[0]):
        out[i] = cv2.resize(vol[i], (IMG, IMG), interpolation=interp)
    u8 = (out * 255).round().clip(0, 255).astype(np.uint8)
    u8[~good] = 0
    info["saturation"] = round(float(((u8[good] == 0) |
                                      (u8[good] == 255)).mean()), 4)
    return u8


def study_laterality(series_infos):
    ballots = []
    for si in series_infos:
        lats = {h["lat"] for h in si["headers"] if h.get("lat")}
        if len(lats) == 1:
            ballots.append(next(iter(lats)))
        elif len(lats) > 1:
            ballots.append("conflict")
    votes = [b for b in ballots if b in ("L", "R")]
    conflict = ("conflict" in ballots) or ("L" in votes and "R" in votes)
    tag_side = None
    if votes:
        cl, cr = votes.count("L"), votes.count("R")
        if cl != cr:
            tag_side = "L" if cl > cr else "R"
    ippx_side = None
    xs = []
    for si in series_infos:
        if si["plane"] == "Sagittal":
            sx = [float(h["ipp"][0]) for h in si["headers"] if "ipp" in h]
            if sx:
                xs.append(float(np.median(sx)))
    if xs:
        x = float(np.median(xs))
        if abs(x) >= LAT_MIN_OFFSET_MM:
            ippx_side = "R" if x < 0 else "L"
    if tag_side:
        return tag_side, "tag", tag_side, ippx_side, conflict
    if ippx_side:
        return ippx_side, "ipp_x_sag", tag_side, ippx_side, conflict
    if votes:
        return sorted(set(votes))[0], "tag_tie", tag_side, ippx_side, True
    return None, "none", None, ippx_side, conflict


def normalise_laterality(vol, plane, lat, info):
    if lat != "R":
        return vol
    if plane in ("Coronal", "Axial"):
        return vol[:, :, ::-1].copy()
    if plane == "Sagittal":
        return vol[::-1].copy()
    warn(info, "lat_skip_unknown_plane", plane)
    return vol


def montage(study, series_data, path):
    rows = len(series_data)
    if rows == 0:
        return
    cols, cell = 8, 128
    canvas = Image.new("L", (cols * cell, rows * cell), 0)
    draw = ImageDraw.Draw(canvas)
    for r, (uid, vol, label) in enumerate(series_data):
        picks = np.linspace(0, vol.shape[0] - 1, cols).astype(int)
        for c, si in enumerate(picks):
            img = Image.fromarray(vol[si]).resize((cell, cell))
            canvas.paste(img, (c * cell, r * cell))
        draw.text((4, r * cell + 4), label, fill=255)
    canvas.save(path)


def process_study(job):
    study, make_qa = job
    t0 = time.time()
    rec = {"study": study, "failed": False, "series": []}
    try:
        sdir_base = ROOT / "train_series" / study
        series_dirs = sorted(p for p in sdir_base.iterdir() if p.is_dir())
        series_infos = []
        for sd in series_dirs:
            files = sorted(e.name for e in os.scandir(sd)
                           if e.name.endswith(".dcm"))
            series_infos.append({
                "uid": sd.name, "dir": str(sd),
                "plane": PLANE.get(sd.name, "Unknown"),
                "headers": sort_order(read_headers(str(sd), files)),
            })
        lat, lat_src, tag_side, ippx_side, lat_conflict = \
            study_laterality(series_infos)
        rec.update(laterality=lat, laterality_src=lat_src,
                   lat_conflict=bool(lat_conflict))

        arrays, meta, qa_rows = {}, [], []
        for si in series_infos:
            info = {"uid": si["uid"], "plane": si["plane"], "failed": False,
                    "n_slices": len(si["headers"]), "syntax": None,
                    "photometric": None, "px_r": None, "px_c": None,
                    "warnings": {}, "warn_examples": []}
            vol, good = decode_series(si["dir"], si["headers"], info)
            if vol is None:
                info["failed"] = True
                meta.append(info)
                continue
            u8 = process_series(vol, good, info)
            del vol
            u8 = normalise_laterality(u8, si["plane"], lat, info)
            if lat == "R" and si["plane"] == "Sagittal":
                good = good[::-1]
            info["n_good"] = int(good.sum())
            info["pad_slices"] = [int(i) for i in np.where(~good)[0]]
            arrays[si["uid"]] = u8
            meta.append(info)
            if make_qa:
                qa_rows.append((si["uid"], u8, f"{si['plane'][:3]} {lat_src}"))

        npz_path = TMP / f"{study}.npz"
        np.savez_compressed(npz_path, **arrays)
        with open(TMP / f"{study}.json", "w") as f:
            json.dump({"laterality": lat, "laterality_src": lat_src,
                       "series": meta}, f)
        rec["series"] = meta
        rec["npz_bytes"] = int(npz_path.stat().st_size)
        if make_qa:
            try:
                montage(study, qa_rows, QA / f"{study}.png")
            except Exception:
                pass
    except Exception as exc:
        rec["failed"] = True
        rec["error"] = str(exc)[:300]
        rec["trace"] = traceback.format_exc()[-1500:]
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def pack_tars(studies_done):
    """Pack per-study npz+json into ~1.8 GB tar parts; delete tmp as we go."""
    part, part_bytes, n_parts = None, 0, 0
    packed = []
    for study in studies_done:
        npz, js = TMP / f"{study}.npz", TMP / f"{study}.json"
        if not npz.is_file():
            continue
        need = npz.stat().st_size + js.stat().st_size
        if part is None or part_bytes + need > TAR_PART_BYTES:
            if part is not None:
                part.close()
            name = WORK / f"shard{SHARD_IDX}_part{n_parts:02d}.tar"
            part = tarfile.open(name, "w")
            packed.append(name.name)
            n_parts += 1
            part_bytes = 0
        part.add(npz, arcname=npz.name)
        part.add(js, arcname=js.name)
        part_bytes += need
        npz.unlink()
        js.unlink()
    if part is not None:
        part.close()
    return packed


def main():
    studies = sorted(p.name for p in (ROOT / "train_series").iterdir()
                     if p.is_dir())
    chunk = (len(studies) + N_SHARDS - 1) // N_SHARDS
    mine = studies[SHARD_IDX * chunk:(SHARD_IDX + 1) * chunk]
    log(f"shard {SHARD_IDX}/{N_SHARDS}: {len(mine)} studies "
        f"({mine[0][:40]}... -> ...{mine[-1][-20:]})")

    jobs = [(st, i < 2) for i, st in enumerate(mine)]
    records, partial = [], False
    t_phase = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, rec in enumerate(pool.map(process_study, jobs)):
            records.append(rec)
            if rec["failed"]:
                log(f"STUDY FAILED {rec['study']}: {rec.get('error')}")
            if (i + 1) % 50 == 0:
                log(f"{i + 1}/{len(mine)} studies")
            if time.time() - T0 > TIME_BUDGET:
                log("TIME BUDGET reached")
                partial = True
                break
    phase_s = time.time() - t_phase

    done = [r["study"] for r in records if not r["failed"]]
    log(f"packing {len(done)} studies into tars")
    packed = pack_tars(done)

    warn_tally = {}
    for r in records:
        for s in r["series"]:
            for k, v in s.get("warnings", {}).items():
                warn_tally[k] = warn_tally.get(k, 0) + int(v)
    sizes = [r["npz_bytes"] for r in records if not r["failed"]]
    manifest = {
        "shard": SHARD_IDX, "n_shards": N_SHARDS, "img": IMG,
        "crop_mm": CROP_MM, "partial": partial,
        "n_studies": len(mine), "n_done": len(done),
        "n_failed": int(sum(r["failed"] for r in records)),
        "failed": [r["study"] for r in records if r["failed"]],
        "tar_parts": packed,
        "studies": done,
        "laterality_sources": dict(Counter(
            str(r.get("laterality_src")) for r in records)),
    }
    with open(WORK / "manifest.json", "w") as f:
        json.dump(manifest, f)
    stats = {
        "shard": SHARD_IDX, "partial": partial,
        "n_done": len(done), "n_failed": manifest["n_failed"],
        "warnings": warn_tally,
        "sec_per_study_mean": round(float(np.mean(
            [r["seconds"] for r in records])), 2) if records else None,
        "gb_total": round(float(np.sum(sizes)) / 2**30, 2) if sizes else 0,
        "phase_s": round(phase_s, 1),
        "elapsed_s": round(time.time() - T0, 1),
    }
    with open(WORK / "stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    log(json.dumps(stats, indent=1))


try:
    main()
except Exception:
    traceback.print_exc()
    raise
