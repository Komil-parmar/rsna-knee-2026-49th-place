# ============ OUR ARM: VPT 5-fold member (appended stage) ============
# Reads /kaggle/working/submission.csv (previous stages), computes our VPT
# member's predictions from raw test DICOMs with the EXACT train-time
# preprocessing (512/160mm/laterality -> 6 slots x 9 slices x 336), and
# rank-blends at weight OUR_W. Fully guarded: any failure or time overrun
# leaves the existing submission untouched.
OUR_W = __W__
OUR_TIME_CAP_S = 7.6 * 3600      # container uptime cap before skipping


def run_our_arm():
    import json
    import os
    import re
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd
    import pydicom
    import torch
    import torch.nn as nn

    t0 = time.time()

    def log(m):
        print(f"[OURS {time.time() - t0:6.1f}s] {m}", flush=True)

    def uptime():
        try:
            return float(Path("/proc/uptime").read_text().split()[0])
        except Exception:
            return 0.0

    if uptime() > OUR_TIME_CAP_S:
        log(f"SKIP: container uptime {uptime():.0f}s past cap")
        return

    TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
               "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
               "Baker's", "Contusion", "Fracture"]
    SLOTS = [("SAG_FLUID_FS", "Sagittal", True, True),
             ("COR_FLUID_FS", "Coronal", True, True),
             ("AX_FLUID_FS", "Axial", True, True),
             ("SAG_FLUID_NOFS", "Sagittal", True, False),
             ("COR_T1", "Coronal", False, False),
             ("SAG_T1", "Sagittal", False, False)]
    SLOT_PLANE = ["Sagittal", "Coronal", "Axial", "Sagittal", "Coronal",
                  "Sagittal"]
    N_SLOT, GROUP, N_GROUP = 6, 3, 3
    CACHE_SLICES = GROUP * N_GROUP
    IMG512, IMG, CROP_MM, LAT_MIN_OFFSET_MM = 512, 336, 160.0, 5.0
    POOL_MAX_IDX = [TARGETS.index(t) for t in
                    ["Fracture", "Contusion", "Medial Meniscus",
                     "Lateral Meniscus", "Baker's"]]

    for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection")]:
        if (c / "test_series").is_dir():
            ROOT = c
            break
    else:
        log("SKIP: no test_series dir")
        return
    inp = Path("/kaggle/input")
    log(f"input dirs: {sorted(p.name for p in inp.iterdir())}")
    cands = [inp / "datasets" / "komilparmar" / "rsna-knee-vpt-member-v1",
             inp / "rsna-knee-vpt-member-v1"]
    cands += sorted(q.parent for q in
                    inp.glob("datasets/*/*/ckpt_fold0.pt"))
    CKPT_DIR = next((c for c in cands
                     if (c / "ckpt_fold0.pt").is_file()), None)
    if CKPT_DIR is None:
        log("SKIP: VPT checkpoints not found under /kaggle/input")
        return
    log(f"ckpts at {CKPT_DIR}")

    ts = pd.read_csv(ROOT / "test_series.csv")
    PLANE = dict(zip(ts["SeriesInstanceUID"].astype(str),
                     ts["Anatomical_Plane"]))

    # ---- slot classification: EXACT port of scripts/make_series_slots.py
    FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
    _SEP = re.compile(r"[_\-.]")
    _FATSAT_RX = re.compile(
        r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|"
        r"\bwe\b|water excit|\btirm\b|\bsting\b|\bfatsup\b")
    _T1_RX = re.compile(r"\bt1\b|\bt1w\b")
    _T2_RX = re.compile(r"\bt2\b|\bt2w\b")
    _PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

    def _mv(v):
        if v is None:
            return ""
        try:
            if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                return "|".join(str(x) for x in v)
        except Exception:
            pass
        return str(v)

    def classify_series(first_file, plane, n_files):
        try:
            d = pydicom.dcmread(first_file, stop_before_pixels=True,
                                force=True)
        except Exception:
            return None
        desc = ((_mv(getattr(d, "SeriesDescription", ""))) + " " +
                (_mv(getattr(d, "SequenceName", "")))).lower()
        desc = _SEP.sub(" ", desc)
        opts = _mv(getattr(d, "ScanOptions", "")).upper().split("|")
        fatsat = bool(_FATSAT_RX.search(desc)) or any(
            x.strip() in FATSAT_OPTS for x in opts)
        try:
            trep = float(getattr(d, "RepetitionTime", None))
        except (TypeError, ValueError):
            trep = float("nan")
        try:
            te = float(getattr(d, "EchoTime", None))
        except (TypeError, ValueError):
            te = float("nan")
        gre = "GR" in _mv(getattr(d, "ScanningSequence", "")).upper()
        t1 = bool(_T1_RX.search(desc))
        t2 = bool(_T2_RX.search(desc))
        pdw = bool(_PD_RX.search(desc))
        if t1 and not t2 and not pdw:
            weight = "T1"
        elif t2 and not pdw:
            weight = "T2"
        elif pdw:
            weight = "PD"
        elif gre:
            weight = "GRE"
        elif trep == trep and trep < 800:
            weight = "T1"
        elif te == te and te > 60:
            weight = "T2"
        elif trep == trep and trep >= 800:
            weight = "PD"
        else:
            weight = "UNK"
        return {"plane": plane, "fatsat": fatsat,
                "fluid": weight in ("PD", "T2"), "n_files": n_files}

    def assign_slots(series_feats):
        chosen = {}
        for name, pl, fluid, fs in SLOTS:
            sel = [(u, f) for u, f in series_feats.items()
                   if f and f["plane"] == pl and f["fatsat"] == fs
                   and f["fluid"] == fluid]
            if not sel and fluid is False:
                sel = [(u, f) for u, f in series_feats.items()
                       if f and f["plane"] == pl and not f["fatsat"]
                       and not f["fluid"]]
            if not sel and fluid is False:
                sel = [(u, f) for u, f in series_feats.items()
                       if f and f["plane"] == pl and not f["fatsat"]]
            if sel:
                chosen[name] = max(sel, key=lambda x: x[1]["n_files"])[0]
        return chosen

    # ---- per-study preprocessing: EXACT port of prep_full_template.py
    SORT_TAGS = ["ImagePositionPatient", "InstanceNumber", "Laterality",
                 "ImageLaterality", "PixelSpacing"]

    def read_headers(sdir, files):
        out = []
        for name in files:
            h = {"file": name}
            try:
                d = pydicom.dcmread(os.path.join(sdir, name),
                                    stop_before_pixels=True, force=True,
                                    specific_tags=SORT_TAGS)
            except Exception:
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
            fb = float(np.median(xyz[:, axis]))
            key = lambda h: (float(h["ipp"][axis]) if "ipp" in h else fb,
                             h["instance"] if h.get("instance") is not None
                             else float("inf"))
        elif sum(h.get("instance") is not None for h in headers) >= 2:
            key = lambda h: (h["instance"] if h.get("instance") is not None
                             else float("inf"))
        else:
            key = lambda h: h["file"]
        return sorted(headers, key=key)

    def decode_series(sdir, headers):
        planes, shapes = [], []
        px_r = px_c = None
        for h in headers:
            try:
                ds = pydicom.dcmread(os.path.join(sdir, h["file"]),
                                     force=True)
                a = ds.pixel_array
                if a.ndim != 2:
                    planes.append(None)
                    shapes.append(None)
                    continue
                a = a.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1) or 1)
                inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
                a = a * slope + inter
                if str(getattr(ds, "PhotometricInterpretation",
                               "MONOCHROME2")) == "MONOCHROME1":
                    a = -a
                planes.append(a)
                shapes.append(a.shape)
            except Exception:
                planes.append(None)
                shapes.append(None)
        good_shapes = [s for s in shapes if s is not None]
        if not good_shapes:
            return None, None, None, None
        shp = Counter(good_shapes).most_common(1)[0][0]
        fixed, good = [], []
        for p in planes:
            if p is None or p.shape != shp:
                fixed.append(np.zeros(shp, np.float32))
                good.append(False)
            else:
                fixed.append(p)
                good.append(True)
        vol = np.stack(fixed)
        pss = [h["ps"] for h in headers if "ps" in h]
        if pss:
            px_r = float(np.median([p[0] for p in pss]))
            px_c = float(np.median([p[1] for p in pss]))
        return vol, np.array(good, bool), px_r, px_c

    def crop_pad_axis(vol, px, axis):
        if not (px and np.isfinite(px) and px > 0):
            return vol, 0
        want = int(round(CROP_MM / px))
        n = vol.shape[axis]
        if want < 16:
            return vol, 0
        if want < n:
            start = (n - want) // 2
            sl = [slice(None)] * 3
            sl[axis] = slice(start, start + want)
            return vol[tuple(sl)], 0
        if want > n:
            return vol, want - n
        return vol, 0

    def process_series(vol, good, px_r, px_c):
        vol, pad_r = crop_pad_axis(vol, px_r, 1)
        vol, pad_c = crop_pad_axis(vol, px_c, 2)
        gv = vol[good]
        lo, hi = np.percentile(gv, [1, 99])
        vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        if pad_r or pad_c:
            vol = np.pad(vol, ((0, 0),
                               (pad_r // 2, pad_r - pad_r // 2),
                               (pad_c // 2, pad_c - pad_c // 2)))
        interp = (cv2.INTER_AREA if min(vol.shape[1:]) > IMG512
                  else cv2.INTER_LINEAR)
        out = np.empty((vol.shape[0], IMG512, IMG512), np.float32)
        for i in range(vol.shape[0]):
            out[i] = cv2.resize(vol[i], (IMG512, IMG512),
                                interpolation=interp)
        u8 = (out * 255).round().clip(0, 255).astype(np.uint8)
        u8[~good] = 0
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
        tag_side = None
        if votes:
            cl, cr = votes.count("L"), votes.count("R")
            if cl != cr:
                tag_side = "L" if cl > cr else "R"
        xs = []
        for si in series_infos:
            if si["plane"] == "Sagittal":
                sx = [float(h["ipp"][0]) for h in si["headers"] if "ipp" in h]
                if sx:
                    xs.append(float(np.median(sx)))
        ippx_side = None
        if xs:
            x = float(np.median(xs))
            if abs(x) >= LAT_MIN_OFFSET_MM:
                ippx_side = "R" if x < 0 else "L"
        if tag_side:
            return tag_side
        if ippx_side:
            return ippx_side
        if votes:
            return sorted(set(votes))[0]
        return None

    def normalise_laterality(vol, plane, lat):
        if lat != "R":
            return vol
        if plane in ("Coronal", "Axial"):
            return vol[:, :, ::-1].copy()
        if plane == "Sagittal":
            return vol[::-1].copy()
        return vol

    def band_slices(u8):
        """train_baseline._build_study: central 20-80% band, 9 slices, 336."""
        n = u8.shape[0]
        lo, hi = int(0.20 * (n - 1)), int(0.80 * (n - 1))
        idx = (np.unique(np.linspace(lo, hi, CACHE_SLICES).astype(int))
               if hi > lo else np.array([n // 2]))
        while len(idx) < CACHE_SLICES:
            idx = np.append(idx, idx[-1])
        out = np.zeros((CACHE_SLICES, IMG, IMG), np.uint8)
        for j, k in enumerate(idx[:CACHE_SLICES]):
            out[j] = cv2.resize(u8[int(k)], (IMG, IMG),
                                interpolation=cv2.INTER_AREA)
        return out

    def process_study(study):
        try:
            base = ROOT / "test_series" / study
            sdirs = sorted(p for p in base.iterdir() if p.is_dir())
            feats, infos = {}, []
            for sd in sdirs:
                files = sorted(e.name for e in os.scandir(sd)
                               if e.name.endswith(".dcm"))
                if not files:
                    continue
                feats[sd.name] = classify_series(
                    str(sd / files[0]), PLANE.get(sd.name, "Unknown"),
                    len(files))
                infos.append({"uid": sd.name, "dir": str(sd),
                              "plane": PLANE.get(sd.name, "Unknown"),
                              "files": files})
            for si in infos:
                si["headers"] = sort_order(
                    read_headers(si["dir"], si["files"]))
            lat = study_laterality(infos)
            chosen = assign_slots(feats)
            row = np.zeros((N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
            msk = np.zeros(N_SLOT, np.float32)
            for si_idx, (name, pl, _, _) in enumerate(SLOTS):
                uid = chosen.get(name)
                if uid is None:
                    continue
                si = next(x for x in infos if x["uid"] == uid)
                vol, good, px_r, px_c = decode_series(si["dir"],
                                                      si["headers"])
                if vol is None:
                    continue
                u8 = process_series(vol, good, px_r, px_c)
                del vol
                u8 = normalise_laterality(u8, si["plane"], lat)
                row[si_idx] = band_slices(u8)
                msk[si_idx] = 1.0
                del u8
            return study, row, msk
        except Exception as exc:
            log(f"study fail {study}: {type(exc).__name__} {str(exc)[:80]}")
            return study, np.zeros((N_SLOT, CACHE_SLICES, IMG, IMG),
                                   np.uint8), np.zeros(N_SLOT, np.float32)

    # ---- model: inference copy of src/train/train_vpt.py (no internet:
    # backbone built from config; ALL weights come from the checkpoint)
    from transformers import Dinov2Config, Dinov2Model
    N_TASK, N_PROMPT = 12, 16
    N_TOKENS = N_TASK + N_PROMPT

    class VPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = Dinov2Config(hidden_size=384, num_hidden_layers=12,
                               num_attention_heads=6, mlp_ratio=4,
                               patch_size=14, image_size=518,
                               num_channels=3)
            self.bb = Dinov2Model(cfg)
            d = 384
            self.tokens = nn.Parameter(torch.zeros(N_TOKENS, d))
            self.slot_dim = 64
            self.slot_emb = nn.Parameter(torch.zeros(N_SLOT, self.slot_dim))
            self.slot_gate = nn.Parameter(torch.zeros(N_TASK, N_SLOT))
            self.heads = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(3 * d + self.slot_dim),
                              nn.Linear(3 * d + self.slot_dim, 256),
                              nn.GELU(), nn.Dropout(0.2),
                              nn.Linear(256, 1))
                for _ in range(N_TASK)])
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def encode(self, x):
            emb = self.bb.embeddings(x)
            t = self.tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
            h = torch.cat([emb[:, :1], t, emb[:, 1:]], dim=1)
            h = self.bb.encoder(h)[0]
            h = self.bb.layernorm(h)
            return h[:, 0], h[:, 1:1 + N_TASK], h[:, 1 + N_TOKENS:].mean(1)

        def forward(self, imgs, mask):
            B, S = imgs.shape[:2]
            x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
            x = (x - self.mean) / self.std
            cls, task, gap = self.encode(x)
            slot = self.slot_emb.unsqueeze(0).expand(B, -1, -1)
            slot = slot.reshape(B * S, self.slot_dim)
            cols = [self.heads[t](torch.cat(
                [task[:, t], gap, cls, slot], dim=-1))
                for t in range(N_TASK)]
            z = torch.cat(cols, dim=-1).reshape(B, S, N_TASK)
            gate = self.slot_gate.T.unsqueeze(0).expand(B, -1, -1)
            gate = gate.masked_fill(mask.unsqueeze(-1) < 0.5, -1e4)
            return (z * gate.softmax(1)).sum(1)

    # ---- pipeline
    studies = sorted(p.name for p in (ROOT / "test_series").iterdir()
                     if p.is_dir())
    log(f"{len(studies)} test studies; W={OUR_W}")
    cache_path = "/kaggle/tmp/our_cache.npy"
    Path("/kaggle/tmp").mkdir(exist_ok=True)
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.uint8,
        shape=(len(studies), N_SLOT, CACHE_SLICES, IMG, IMG))
    masks = np.zeros((len(studies), N_SLOT), np.float32)
    sidx = {s: i for i, s in enumerate(studies)}
    n_done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for study, row, msk in pool.map(process_study, studies):
            cache[sidx[study]] = row
            masks[sidx[study]] = msk
            n_done += 1
            if n_done % 100 == 0:
                log(f"prep {n_done}/{len(studies)}")
            if uptime() > OUR_TIME_CAP_S:
                log("TIME CAP during prep - aborting our arm")
                return
    cache.flush()
    empty = masks.sum(1) == 0
    log(f"prep done; {int(empty.sum())} studies with zero slots")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_available()
                 and torch.cuda.is_bf16_supported() else torch.float16)
    probs_sum = np.zeros((len(studies), N_TASK), np.float64)
    for fold in range(5):
        ck = torch.load(CKPT_DIR / f"ckpt_fold{fold}.pt",
                        map_location="cpu", weights_only=False)
        model = VPTModel()
        model.load_state_dict(ck["state_dict"])
        model.to(dev).eval()
        with torch.no_grad():
            out = []
            B = 12
            for b in range(0, len(studies), B):
                rows = torch.from_numpy(
                    np.asarray(cache[b:b + B])).to(dev)
                m = torch.from_numpy(masks[b:b + B]).to(dev)
                gp = []
                for g in range(N_GROUP):
                    imgs = rows[:, :, g * GROUP:(g + 1) * GROUP]
                    with torch.autocast(dev.type, dtype=amp_dtype,
                                        enabled=dev.type == "cuda"):
                        z = model(imgs, m).float()
                    gp.append(torch.sigmoid(z))
                gp = torch.stack(gp)
                pooled = gp.mean(0)
                pooled[:, POOL_MAX_IDX] = \
                    gp[:, :, POOL_MAX_IDX].max(0).values
                out.append(pooled.cpu().numpy())
        probs_sum += np.concatenate(out)
        del model
        torch.cuda.empty_cache()
        log(f"fold {fold} inference done")
        if uptime() > OUR_TIME_CAP_S + 1800:
            log("TIME CAP during inference - aborting our arm")
            return
    probs = probs_sum / 5.0

    sub = pd.read_csv("/kaggle/working/submission.csv",
                      dtype={"StudyInstanceUID": str})
    ours = pd.DataFrame(probs, columns=TARGETS)
    ours.insert(0, "StudyInstanceUID", studies)
    aligned = sub[["StudyInstanceUID"]].merge(ours, on="StudyInstanceUID",
                                              how="left")
    assert len(aligned) == len(sub), "alignment failed"
    theirs_rank = sub[TARGETS].rank(method="average", pct=True) \
        .to_numpy(np.float64)
    ours_vals = aligned[TARGETS].to_numpy(np.float64)
    missing = ~np.isfinite(ours_vals).all(1)
    empty_rows = aligned["StudyInstanceUID"].map(
        dict(zip(studies, empty))).fillna(True).to_numpy(bool)
    ours_rank = pd.DataFrame(ours_vals, columns=TARGETS).rank(
        method="average", pct=True).to_numpy(np.float64)
    bad = missing | empty_rows
    ours_rank[bad] = theirs_rank[bad]      # neutral where our arm is blind
    blended = (1.0 - OUR_W) * theirs_rank + OUR_W * ours_rank
    assert np.isfinite(blended).all()
    sub[TARGETS] = blended
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    log(f"BLEND WRITTEN: W={OUR_W}, {int(bad.sum())} rows kept theirs")


try:
    run_our_arm()
except Exception as _exc:
    # previous stages already wrote a valid submission; never destroy it
    print(f"[OURS] arm skipped safely: {type(_exc).__name__}: {_exc}",
          flush=True)
