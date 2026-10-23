# Campaign copy v1; archived source SHA256 0d97f2a02404761f9b0d7a0e0140c9aeaaf7c59bada53c139a20a1b4532daeab
"""exp-029: whole-study SLICE-BAG MIL (renumbered from exp-026) (Phase 4b, I-052/I-047).

Every slice of every series is a candidate. Training samples a random bag of
--bag slices per study (2.5D triplets of physical neighbours within the same
series), encodes each with DINOv2, adds series-type (plane/fluid/fatsat) and
relative-position embeddings, mixes the bag with a small Transformer, and pools
per target with learned attention queries (attention-MIL). Inference averages
--eval_bags random bags. EMA weights are evaluated and saved (I-051).

Cache layout (scripts/kaggle_allslice_build): <cache_dir>/slices_part{k}.npy
[N,224,224] uint8 (memmap), index_part{k}.csv (one row per slice), k=0..2.
"""
import argparse
import copy
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_baseline import (AMP_DTYPE, SWAP_IDX, TARGETS, batch_weights,
                            build_param_groups, ema_update, eval_targets, log,
                            macro_auc, make_lr_lambda)
from lowres_utils import cosine_hold_lambda, parse_drop_epochs

PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
N_TARGET = len(TARGETS)


# ---------------------------------------------------------------- data
class SliceBank:
    """Memmapped parts + per-study slice tables (no RAM copy of pixels)."""

    def __init__(self, cache_dir, n_parts=3, drop_zero=True, feat_dir=None):
        self.parts, frames = [], []
        for k in range(n_parts):
            self.parts.append(np.load(f"{cache_dir}/slices_part{k}.npy",
                                      mmap_mode="r"))
            idx = pd.read_csv(f"{cache_dir}/index_part{k}.csv")
            idx["part"] = k
            frames.append(idx)
        self.feats, self.feat_dim = None, 0     # exp-061: pooled-feature bank
        if feat_dir:
            self.feats = [np.load(f"{feat_dir}/feats_part{k}.npy",
                                  mmap_mode="r") for k in range(n_parts)]
            assert all(f.shape[0] == p.shape[0]
                       for f, p in zip(self.feats, self.parts)), \
                "feature bank rows misaligned with pixel bank"
            self.feat_dim = int(self.feats[0].shape[1])
        idx = pd.concat(frames, ignore_index=True)
        if drop_zero:
            idx = idx[idx["nonzero"] == 1]
        idx["plane_id"] = idx["plane"].map(PLANE_ID).fillna(3).astype(int)
        idx["relpos"] = idx["slice_idx"] / np.maximum(idx["n_slices"] - 1, 1)
        cols = ["part", "row", "plane_id", "fluid", "fatsat", "slice_idx",
                "n_slices", "relpos"]
        self.by_study = {s: g[cols].to_numpy(np.float32)
                         for s, g in idx.groupby("study", sort=False)}
        self.n_slices_total = len(idx)
        self.img = int(self.parts[0].shape[1])   # exp-040: 224 or 336 bank
        self.xpair = None            # exp-067: cross-contrast partner rows
        self.xcontrast_p = 0.0       # set by run_fold / eval scripts
        self.tri_gap = 1             # exp-079: through-plane token extent

    def triplet(self, part, row, j, n):
        """3 physical neighbours inside the same series (rows are contiguous).

        exp-079: `tri_gap` G sets how far the two side channels reach, so a
        token spans 2*G slice-gaps instead of 2. Compute and memory are
        IDENTICAL (still 3 channels); only the through-plane extent changes.
        Clamped inside the series, so slices near an end degrade to a
        narrower token rather than crossing into a different series."""
        arr = self.parts[int(part)]
        r, g = int(row), int(self.tri_gap)
        a = r - min(g, int(j))
        b = r + min(g, int(n) - 1 - int(j))
        return np.stack([arr[a], arr[r], arr[b]])          # [3,H,W] uint8

    def build_xpair(self):
        """exp-067 CROSS-CONTRAST tokens: for every slice, find the row of the
        SAME anatomical level in a series of the OTHER contrast class in the
        SAME plane. Only two contrast classes exist in the allowlisted
        metadata ((fluid,fatsat) = (0,0) T1/PD-like and (1,1) T2FS/STIR-like,
        EDA 2026-09-03), so `fluid` alone identifies the class and a
        cross-contrast PAIR is the deepest pairing the data supports.
        Levels are matched by relative position within the series; rows of a
        series are contiguous, so start_row = row - slice_idx.
        Returns the fraction of slices that found a partner."""
        self.xpair, have, tot = {}, 0, 0
        for s, tab in self.by_study.items():
            part = tab[:, 0].astype(np.int64)
            row = tab[:, 1].astype(np.int64)
            pl = tab[:, 2].astype(np.int64)
            fl = tab[:, 3].astype(np.int64)
            j = tab[:, 5].astype(np.int64)
            ns = tab[:, 6].astype(np.int64)
            rp = tab[:, 7].astype(np.float64)
            start = row - j                      # series start row in its part
            byplane = {}                         # (plane, fluid) -> series list
            for p_, st_, pl_, fl_, ns_ in set(
                    zip(part.tolist(), start.tolist(), pl.tolist(),
                        fl.tolist(), ns.tolist())):
                byplane.setdefault((pl_, fl_), []).append((p_, st_, ns_))
            out = np.full((len(tab), 2), -1, np.int64)
            for k in range(len(tab)):
                cands = byplane.get((pl[k], 1 - fl[k]))
                if not cands:
                    continue
                p_, st_, ns_ = max(cands, key=lambda c: c[2])   # fullest cover
                jj = int(round(float(rp[k]) * (ns_ - 1)))
                r_ = st_ + min(max(jj, 0), ns_ - 1)
                if 0 <= r_ < self.parts[p_].shape[0]:
                    out[k] = (p_, r_)
                    have += 1
            tot += len(tab)
            self.xpair[s] = out
        return have / max(tot, 1)

    def xtriplet(self, part, row, j, n, xpart, xrow):
        """exp-067 token: [other-contrast @ same level, anchor, anchor+1].
        Channel 0 replaces the anchor's PREVIOUS neighbour, so the 2.5D
        context (anchor, anchor+1) is kept and only one channel changes."""
        arr = self.parts[int(part)]
        r = int(row)
        b = r + 1 if j < n - 1 else r
        return np.stack([self.parts[int(xpart)][int(xrow)], arr[r], arr[b]])


def _window_pick(tab, bag, rng, jitter=0.5):
    """exp-072 SLICE GEOMETRY: instead of a uniform random draw over every
    slice of the study, take a CONTIGUOUS window from each series, centred
    with jitter. A competitor measured centre-adjacent beating equal-spaced
    by +0.018 on fold 0 - larger than our whole 224->336 resolution gain.
    Rationale: a meniscal tear lives on 2-3 adjacent slices; a scattered
    draw rarely shows one with its neighbours.
    Series are identified by (part, row - slice_idx), which is stable even
    though all-zero slices were dropped from the index. Returns tab
    positions (may be shorter than `bag`; caller tops up)."""
    n = len(tab)
    ser = {}
    for i in range(n):
        k = (int(tab[i, 0]), int(tab[i, 1]) - int(tab[i, 5]))
        ser.setdefault(k, []).append(i)
    keys = list(ser)
    rng.shuffle(keys)
    per = max(1, int(bag // max(len(keys), 1)))
    out = []
    for k in keys:
        idx = ser[k]                       # tab positions, slice order
        m = len(idx)
        take = min(per, m, bag - len(out))
        if take <= 0:
            break
        lo_max = m - take
        c = lo_max / 2.0                   # centred
        span = lo_max * float(jitter) / 2.0
        lo = int(round(min(max(c + rng.uniform(-span, span), 0), lo_max)))
        out.extend(idx[lo:lo + take])
        if len(out) >= bag:
            break
    return out


def _token(bank, s, t, part, row, j, ns, xp, rng):
    """exp-067: cross-contrast token with probability xp, else the classic
    3-neighbour triplet (also the fallback when this slice has no partner of
    the other contrast in its plane)."""
    if xp > 0 and bank.xpair is not None:
        if xp >= 1.0 or rng.random() < xp:
            xpart, xrow = bank.xpair[s][t]
            if xpart >= 0:
                return bank.xtriplet(part, row, int(j), int(ns), xpart, xrow)
    return bank.triplet(part, row, int(j), int(ns))


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, bank, studies, bag, seed, train, n_bags=1):
        self.bank, self.studies, self.bag = bank, list(studies), bag
        self.train, self.n_bags = train, n_bags
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.studies) * self.n_bags

    def _tok(self, s, t, part, row, j, ns, xp, rng):
        return _token(self.bank, s, t, part, row, j, ns, xp, rng)

    def __getitem__(self, i):
        s = self.studies[i % len(self.studies)]
        b = i // len(self.studies)
        tab = self.bank.by_study.get(s)
        rng = np.random.default_rng(
            (self.seed * 7919 + self.epoch * 104729 + i * 31) % (2 ** 31))
        L = self.bag
        H = self.bank.img                        # exp-040: bank-native size
        fb = self.bank.feats                     # exp-061: feature mode
        imgs = (np.zeros((L, self.bank.feat_dim), np.float32)
                if fb is not None else np.zeros((L, 3, H, H), np.uint8))
        meta = np.zeros((L, 4), np.float32)
        mask = np.zeros(L, np.float32)
        if tab is not None and len(tab):
            n = len(tab)
            mode = str(getattr(self, "bag_mode", "random"))
            if mode == "window":           # exp-072: contiguous per-series
                w = _window_pick(tab, L, rng,
                                 float(getattr(self, "bag_jitter", 0.5)))
                if len(w) < L:             # top up from anywhere
                    extra = rng.choice(n, L - len(w),
                                       replace=n < L - len(w))
                    w = list(w) + list(extra)
                pick = np.asarray(w[:L], np.int64)
            else:
                pick = (rng.choice(n, L, replace=False) if n >= L
                        else rng.choice(n, L, replace=True))
            xp = float(getattr(self.bank, "xcontrast_p", 0.0))
            for q, t in enumerate(pick):
                part, row, pl, fl, fs, j, ns, rp = tab[t]
                if fb is not None:
                    imgs[q] = fb[int(part)][int(row)]
                else:
                    imgs[q] = self._tok(s, t, part, row, j, ns, xp, rng)
                meta[q] = (pl, fl, fs, rp)
                mask[q] = 1.0
        return (torch.from_numpy(imgs), torch.from_numpy(meta),
                torch.from_numpy(mask), i % len(self.studies))


class WholeStudyDataset(torch.utils.data.Dataset):
    """exp-045: EVERY slice of the study per sample (no bag subsampling —
    kills the bag-label mismatch). Optional train-time slice DROPOUT keeps
    bagging's implicit regularization explicitly (user hypothesis). Studies
    beyond max_slices are uniformly subsampled (rare)."""

    def __init__(self, bank, studies, seed, train, slice_drop=0.0,
                 max_slices=256):
        self.bank, self.studies = bank, list(studies)
        self.seed, self.train = seed, train
        self.slice_drop, self.max_slices = float(slice_drop), int(max_slices)
        self.epoch = 0
        # exp-047 arm B (set post-construction by run_fold; None = off)
        self.protect = None          # study -> int16 [12, k] tab positions
        self.protect_pos = {}        # study -> bool [12] positive targets
        self.p_prot = 0.9

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        s = self.studies[i]
        tab = self.bank.by_study.get(s)
        rng = np.random.default_rng(
            (self.seed * 7919 + self.epoch * 104729 + i * 31) % (2 ** 31))
        n = 0 if tab is None else len(tab)
        idx = np.arange(n)
        if self.train and self.slice_drop > 0 and n > 8:
            keep = rng.random(n) >= self.slice_drop
            if keep.sum() < 8:
                keep[rng.choice(n, 8, replace=False)] = True
            if self.protect is not None:     # exp-047 arm B: evidence floor
                pr = self.protect.get(s)
                pp = self.protect_pos.get(s)
                if pr is not None and pp is not None:
                    for c in np.where(pp)[0]:
                        sl = pr[c][pr[c] >= 0]
                        sl = sl[sl < n]
                        if len(sl) and not keep[sl].any() \
                                and rng.random() < self.p_prot:
                            keep[int(sl[rng.integers(len(sl))])] = True
            idx = idx[keep]
        if len(idx) > self.max_slices:
            idx = np.sort(rng.choice(idx, self.max_slices, replace=False))
        L = max(len(idx), 1)
        H = self.bank.img
        fb = self.bank.feats                     # exp-061: feature mode
        imgs = (np.zeros((L, self.bank.feat_dim), np.float32)
                if fb is not None else np.zeros((L, 3, H, H), np.uint8))
        meta = np.zeros((L, 4), np.float32)
        mask = np.zeros(L, np.float32)
        xp = float(getattr(self.bank, "xcontrast_p", 0.0))
        for q, t in enumerate(idx):
            part, row, pl, fl, fs, j, ns, rp = tab[t]
            imgs[q] = (fb[int(part)][int(row)] if fb is not None
                       else _token(self.bank, s, t, part, row, j, ns, xp, rng))
            meta[q] = (pl, fl, fs, rp)
            mask[q] = 1.0
        return (torch.from_numpy(imgs), torch.from_numpy(meta),
                torch.from_numpy(mask), i)


@torch.no_grad()
def predict_full(model, bank, studies, dev, max_slices=256, workers=4):
    """exp-045 eval: ONE deterministic full-study forward per study.
    Worker-prefetched loader so tensor building overlaps GPU compute."""
    model.eval()
    ds = WholeStudyDataset(bank, studies, seed=0, train=False,
                           max_slices=max_slices)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    out = np.zeros((len(studies), N_TARGET), np.float64)
    for imgs, meta, mask, idx in dl:
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            z = model(imgs.to(dev, non_blocking=True), meta.to(dev),
                      mask.to(dev)).float()
        out[int(idx[0])] = torch.sigmoid(z)[0].cpu().numpy()
    return out


@torch.no_grad()
def compute_protect(model, bank, studies, pos_mask, dev, topk=3, workers=4):
    """exp-047 arm B: EMA attention -> top-k slice TAB positions per positive
    target. max_slices=1024 (> any study) so positions are uncapped/exact.
    Returns (dict study -> int16 [N_TARGET, topk], mean attention entropy
    over positive targets)."""
    model.eval()
    todo = [s for s in studies if pos_mask.get(s) is not None
            and pos_mask[s].any()]
    ds = WholeStudyDataset(bank, todo, seed=0, train=False, max_slices=1024)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    prot, ent = {}, []
    for imgs, meta, mask, i in dl:
        s = todo[int(i)]
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            _, att = model(imgs.to(dev, non_blocking=True), meta.to(dev),
                           mask.to(dev), return_attn=True)
        a = att[0].float()                                 # [N_TARGET, L]
        k = min(topk, a.shape[1])
        pr = np.full((N_TARGET, topk), -1, np.int16)
        pr[:, :k] = a.topk(k, dim=1).indices.cpu().numpy().astype(np.int16)
        prot[s] = pr
        pa = a[torch.from_numpy(pos_mask[s]).to(a.device)]
        if len(pa):
            ent.append(float((-(pa.clamp_min(1e-9).log() * pa)
                              .sum(-1)).mean()))
    return prot, (float(np.mean(ent)) if ent else float("nan"))


def make_train_loader(ds, batch, workers):
    """P3.5: persistent_workers MUST be False - `ds.epoch` is set on the
    parent's copy each epoch and only reaches worker processes when they are
    re-spawned; with persistent workers every study would see the SAME bag
    for all epochs (diagnosis 2026-08-21 §2.4, tests/test_day1_fixes.py)."""
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                       num_workers=workers, pin_memory=True,
                                       drop_last=True,
                                       persistent_workers=False)


def flip_bag(imgs, meta):
    """LEGACY (off by default, --flip_p 0): hflip coronal/axial pixels; for
    sagittal, reverse the triplet order and mirror the relative position.
    Defect: the centred triplet stays at the same absolute depth, so only
    `relpos` carries the mirror -> teaches laterality from a scalar feature."""
    out = imgs.clone()
    pl = meta[..., 0]
    sag = pl == 0
    out[~sag] = torch.flip(out[~sag], dims=[-1])
    out[sag] = torch.flip(out[sag], dims=[1])
    m2 = meta.clone()
    m2[..., 3] = torch.where(sag, 1.0 - m2[..., 3], m2[..., 3])
    return out, m2


class _TimmViTWrap(nn.Module):
    """exp-039: expose a timm ViT (DINOv3 etc.) through the HF-Dinov2 surface
    the trainer relies on — .encoder.layer (blocks), .layernorm (final norm),
    .config.hidden_size, forward(pixel_values=x).last_hidden_state — with
    REGISTER tokens dropped (cls stays token 0, patches follow), so cls+mean
    pooling and every downstream module work unchanged."""

    def __init__(self, name):
        super().__init__()
        import types

        import os

        import timm
        pre = os.environ.get("TIMM_NO_PRETRAIN") != "1"   # tests: offline
        self.m = timm.create_model(name, pretrained=pre, num_classes=0)
        self.n_prefix = int(getattr(self.m, "num_prefix_tokens", 1))
        self.encoder = types.SimpleNamespace(layer=self.m.blocks)
        self.layernorm = self.m.norm            # alias; params dedup by id
        self.config = types.SimpleNamespace(hidden_size=self.m.num_features)
        cfg = getattr(self.m, "pretrained_cfg", {}) or {}
        self.norm_mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        self.norm_std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))

    def forward(self, pixel_values):
        import types
        f = self.m.forward_features(pixel_values)      # [B, P+N, D]
        out = torch.cat([f[:, :1], f[:, self.n_prefix:]], 1)
        return types.SimpleNamespace(last_hidden_state=out)


class _TimmHybridWrap(nn.Module):
    """exp-059: expose a timm conv-attention hybrid (CoAtNet/MaxViT family)
    through the same HF surface the trainer relies on. forward_features
    yields an NCHW map; tokens = flattened cells, with a synthetic 'cls'
    slot = spatial max-pool (so cls+mean pooling reads max+mean). Input is
    resized to the model's native resolution inside the wrap so window/grid
    attention shapes stay valid regardless of the cache size."""

    def __init__(self, name, resize=True):
        super().__init__()
        import os
        import types

        import timm
        pre = os.environ.get("TIMM_NO_PRETRAIN") != "1"
        pretrained_file = os.environ.get('RSNA_CAMPAIGN_PRETRAINED_FILE', '')
        assert pre and pretrained_file, 'campaign requires pinned generic pretrained weights'
        self.m = timm.create_model(name, pretrained=True, num_classes=0,
            pretrained_cfg_overlay={'file': pretrained_file, 'source': ''})
        # exp-082: a PURE conv (ConvNeXt/ResNet/EffNet) is shape-agnostic, so
        # feeding the cache's own resolution is both cheaper and sharper than
        # resampling to the pretrain size. Only window/grid-attention hybrids
        # (CoAtNet/MaxViT) actually need the native shape.
        self.resize = bool(resize)
        blocks = []
        for st in self.m.stages:
            blocks += list(st.blocks)
        assert blocks, "hybrid wrap found no stage blocks"
        self.encoder = types.SimpleNamespace(layer=blocks)
        fn = getattr(self.m, "norm", None)
        if fn is None or isinstance(fn, nn.Identity):
            head = getattr(self.m, "head", None)
            fn = getattr(head, "norm", None) if head is not None else None
        self.layernorm = fn if fn is not None else nn.Identity()
        self.config = types.SimpleNamespace(hidden_size=self.m.num_features)
        cfg = getattr(self.m, "pretrained_cfg", {}) or {}
        self.norm_mean = tuple(cfg.get("mean", (0.5, 0.5, 0.5)))
        self.norm_std = tuple(cfg.get("std", (0.5, 0.5, 0.5)))
        self.native = int(cfg.get("input_size", (3, 384, 384))[-1])

    def forward(self, pixel_values):
        import types
        x = pixel_values
        if self.resize and x.shape[-1] != self.native:
            x = F.interpolate(x, size=(self.native, self.native),
                              mode="bilinear", align_corners=False)
        f = self.m.forward_features(x)               # [B, C, h, w]
        tok = f.flatten(2).transpose(1, 2)           # [B, hw, C]
        cls = f.amax(dim=(2, 3)).unsqueeze(1)        # max-pool 'cls' slot
        return types.SimpleNamespace(
            last_hidden_state=torch.cat([cls, tok], 1))


class _LoRALinear(nn.Module):
    """exp-064 (I-095): frozen base Linear + trainable low-rank delta.
    B is zero-init -> the wrapped layer computes EXACTLY the base function
    at step 0 (function preservation); gradients reach B first, then A.
    Merge for deployment: W <- W + (B @ A) * scale, restoring plain naming."""

    def __init__(self, base, r, alpha, n_split=1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.n_split = int(n_split)
        if self.n_split == 1:
            self.A = nn.Parameter(torch.zeros(r, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:                     # exp-066: fused qkv -> per-projection
            assert base.out_features % self.n_split == 0     # factor pairs
            o = base.out_features // self.n_split
            self.A = nn.Parameter(torch.zeros(self.n_split, r,
                                              base.in_features))
            self.B = nn.Parameter(torch.zeros(self.n_split, o, r))
            for i in range(self.n_split):
                nn.init.kaiming_uniform_(self.A[i], a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        if self.n_split == 1:
            return y + F.linear(F.linear(x, self.A), self.B) * self.scale
        d = torch.cat([F.linear(F.linear(x, self.A[i]), self.B[i])
                       for i in range(self.n_split)], dim=-1)
        return y + d * self.scale


class _FeatStub(nn.Module):
    """exp-061: precomputed pooled features replace the backbone. Presents
    the minimal HF-ish surface (config / encoder.layer / layernorm) so the
    param-grouping and unfreeze machinery run unchanged (they see zero
    blocks and no parameters)."""

    def __init__(self, dim):
        super().__init__()
        import types
        self.config = types.SimpleNamespace(hidden_size=dim)
        self.encoder = types.SimpleNamespace(layer=[])
        self.layernorm = nn.Identity()


# ---------------------------------------------------------------- model
def _study_affine(x, B, p, rot, scl, sh):
    # Pose/scale aug: with prob p per study, ONE random in-plane rotation
    # (+-rot deg), zoom (x (1+scl)^U[-1,1]) and shift (+-sh of the half-width)
    # shared by all of its slices (site positioning, coil centring, knee size).
    L = x.shape[0] // B
    th = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    th[:, 0, 0] = 1.0
    th[:, 1, 1] = 1.0
    hit = False
    for b in range(B):
        if float(torch.rand(())) < p:
            hit = True
            a = math.radians(rot) * (2.0 * float(torch.rand(())) - 1.0)
            s = math.exp(math.log(1.0 + scl) * (2.0 * float(torch.rand(())) - 1.0))
            c, n = math.cos(a) / s, math.sin(a) / s
            th[b, 0, 0], th[b, 0, 1], th[b, 1, 0], th[b, 1, 1] = c, -n, n, c
            th[b, 0, 2] = sh * (2.0 * float(torch.rand(())) - 1.0)
            th[b, 1, 2] = sh * (2.0 * float(torch.rand(())) - 1.0)
    if not hit:
        return x
    th = th.repeat_interleave(L, dim=0)
    g = F.affine_grid(th, list(x.shape), align_corners=False)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


def _down_up(x, B, p, smin):
    # Acquisition-resolution aug: with prob p per study, resample all its
    # slices down to a random scale in [smin, 1] (antialiased) and back up,
    # mimicking a coarser in-plane matrix of another scanner/protocol.
    L = x.shape[0] // B
    H, W = x.shape[-2:]
    out = []
    for b in range(B):
        xb = x[b * L:(b + 1) * L]
        if float(torch.rand(())) < p:
            s = smin + (1.0 - smin) * float(torch.rand(()))
            h, w = max(8, int(round(H * s))), max(8, int(round(W * s)))
            xb = F.interpolate(xb, size=(h, w), mode="bilinear",
                               align_corners=False, antialias=True)
            xb = F.interpolate(xb, size=(H, W), mode="bilinear",
                               align_corners=False)
        out.append(xb)
    return torch.cat(out, 0)


def _recon_aug(x, B, p, nmax, smax, bmax):
    # Reconstruction/SNR aug: with prob p per study, add Rician noise of a
    # random level in [0, nmax] (x in [0, 1]), then one random in-plane kernel:
    # unsharp amount a in [-bmax, smax] (a < 0 smooths, a > 0 sharpens).
    L, C = x.shape[0] // B, x.shape[1]
    g = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype) / 16.0
    kh = g.view(1, 1, 1, 5).repeat(C, 1, 1, 1)
    kv = g.view(1, 1, 5, 1).repeat(C, 1, 1, 1)
    out = []
    for b in range(B):
        xb = x[b * L:(b + 1) * L]
        if float(torch.rand(())) < p:
            s = nmax * float(torch.rand(()))
            a = -bmax + (smax + bmax) * float(torch.rand(()))
            n1, n2 = torch.randn_like(xb) * s, torch.randn_like(xb) * s
            xb = torch.sqrt((xb + n1) ** 2 + n2 ** 2)
            bl = F.conv2d(F.pad(xb, (2, 2, 0, 0), mode="reflect"), kh, groups=C)
            bl = F.conv2d(F.pad(bl, (0, 0, 2, 2), mode="reflect"), kv, groups=C)
            xb = (xb + a * (xb - bl)).clamp(0.0, 1.0)
        out.append(xb)
    return torch.cat(out, 0)


_SOFT_GCE_CFG = [0.0, 0.7]   # [mix weight lam, q]; lam 0 = plain BCE


def _soft_gce_bce(logits, target, weight=None, size_average=None, reduce=None,
                  reduction="mean", pos_weight=None):
    # Proper binary density-power score replaces the original GCE term.
    # Its gradient is 2^q*p*(1-p)*(p^(q-1)+(1-p)^(q-1))*(p-y).
    # Thus the soft-target optimum remains p=y; scale matches BCE at p=0.5.
    # Only while grads are enabled (training); eval/val paths stay plain BCE.
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight,
                                             reduction="none",
                                             pos_weight=pos_weight)
    lam, q = float(_SOFT_GCE_CFG[0]), float(_SOFT_GCE_CFG[1])
    if lam > 0 and torch.is_grad_enabled() and logits.shape == target.shape:
        p = torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)
        t = target.float()
        r = 1.0 - p
        g = (2.0 ** q) * (
            (p.pow(q + 1.0) + r.pow(q + 1.0)) / (q + 1.0)
            - (t * p.pow(q) + (1.0 - t) * r.pow(q)) / q
            + 1.0 / (q * (q + 1.0)))
        if weight is not None:
            g = g * weight.float()
        bce = ((1.0 - lam) * bce.float() + lam * g).to(bce.dtype)
    if reduction == "mean":
        return bce.mean()
    if reduction == "sum":
        return bce.sum()
    return bce


class SliceBagModel(nn.Module):
    def __init__(self, backbone, unfreeze_last, d_model=384, n_mix=2,
                 n_heads=6, drop=0.1, use_xattn=False, pool="cls_mean",
                 ms_layers=(), aux_backbone="", proj_adapter=0,
                 lora_r=0, lora_alpha=16, quad_pool=0, lora_targets="qv",
                 full_finetune=False):
        super().__init__()
        self.feat_mode = backbone.startswith("feat:")   # exp-061
        if self.feat_mode:
            assert not ms_layers and not aux_backbone and not use_xattn \
                and pool == "cls_mean", \
                "feat mode: taps/aux/xattn/focal need pixels"
            bb = _FeatStub(int(backbone[5:]))
        elif backbone.startswith("timm:"):        # exp-039: e.g. timm:vit_small_patch16_dinov3.lvd1689m
            nm = backbone[5:]
            if any(k in nm for k in ("coatnet", "maxvit", "maxxvit")):
                bb = _TimmHybridWrap(nm)         # exp-059: conv-attn hybrids
            elif any(k in nm for k in ("convnext", "resnet", "efficientnet",
                                       "regnet")):
                bb = _TimmHybridWrap(nm, resize=False)   # exp-082: pure convs
            else:
                bb = _TimmViTWrap(nm)
        else:
            from transformers import AutoModel
            bb = AutoModel.from_pretrained(backbone)
        for p in bb.parameters():
            p.requires_grad = False
        if unfreeze_last > 0:
            for blk in bb.encoder.layer[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in bb.layernorm.parameters():
                p.requires_grad = True
        self.full_finetune = bool(full_finetune)
        if self.full_finetune:
            assert not self.feat_mode and lora_r == 0, \
                'full_finetune needs a pixel backbone and cannot be combined with LoRA'
            # Stage transitions and the stem are OUTSIDE stages[*].blocks.
            # Unfreezing all listed blocks alone silently leaves them frozen.
            for p in bb.parameters():
                p.requires_grad = True
        self.lora_r = int(lora_r)
        self.lora_targets = str(lora_targets or "qv")
        if self.lora_r > 0:                      # exp-064 (I-095): LoRA
            assert unfreeze_last == 0 and not self.feat_mode, \
                "LoRA replaces unfreezing and needs a real backbone"
            assert self.lora_targets in ("qv", "all"), self.lora_targets
            all_lin = self.lora_targets == "all"     # exp-066: all-linear
            r_, a_ = self.lora_r, lora_alpha
            n_inj = 0
            for blk in bb.encoder.layer:
                if hasattr(blk, "attention"):    # HF Dinov2
                    att = blk.attention.attention
                    att.query = _LoRALinear(att.query, r_, a_)
                    att.value = _LoRALinear(att.value, r_, a_)
                    n_inj += 2
                    if all_lin:                  # + k, out-proj, MLP
                        att.key = _LoRALinear(att.key, r_, a_)
                        blk.attention.output.dense = _LoRALinear(
                            blk.attention.output.dense, r_, a_)
                        n_inj += 2
                        for nm_ in ("fc1", "fc2"):
                            if isinstance(getattr(blk.mlp, nm_, None),
                                          nn.Linear):
                                setattr(blk.mlp, nm_, _LoRALinear(
                                    getattr(blk.mlp, nm_), r_, a_))
                                n_inj += 1
                elif hasattr(blk, "attn") and isinstance(
                        getattr(blk.attn, "qkv", None), nn.Linear):
                    if all_lin:                  # timm ViT: split qkv
                        blk.attn.qkv = _LoRALinear(blk.attn.qkv, r_, a_,
                                                   n_split=3)
                        n_inj += 1
                        if isinstance(getattr(blk.attn, "proj", None),
                                      nn.Linear):
                            blk.attn.proj = _LoRALinear(blk.attn.proj, r_, a_)
                            n_inj += 1
                        for nm_ in ("fc1", "fc2"):
                            if isinstance(getattr(blk.mlp, nm_, None),
                                          nn.Linear):
                                setattr(blk.mlp, nm_, _LoRALinear(
                                    getattr(blk.mlp, nm_), r_, a_))
                                n_inj += 1
                    else:
                        blk.attn.qkv = _LoRALinear(blk.attn.qkv, r_, a_)
                        n_inj += 1
            assert n_inj > 0, "LoRA: no attention Linears found to wrap"
            self._lora_n_inj = n_inj
        self.bb = bb
        d = bb.config.hidden_size
        self.pool = pool                        # exp-044: cls_mean(_focal)
        self.ms_layers = tuple(ms_layers)       # exp-058: detached hypercols
        if self.ms_layers:
            assert not backbone.startswith("timm:"), \
                "ms_layers needs HF hidden_states (not the timm wrap)"
        self.bb2 = None                         # exp-060: frozen aux tower
        d2 = 0
        if aux_backbone:
            assert aux_backbone.startswith("timm:"), \
                "aux_backbone: timm names only"
            nm2 = aux_backbone[5:]
            self.bb2 = (_TimmHybridWrap(nm2) if any(
                k in nm2 for k in ("coatnet", "maxvit", "maxxvit"))
                else _TimmViTWrap(nm2))
            for p in self.bb2.parameters():     # FROZEN forever (Meta-style)
                p.requires_grad = False
            d2 = 2 * self.bb2.config.hidden_size
        self.quad_pool = bool(quad_pool)        # exp-065 QUAD (jury winner):
        if self.quad_pool:                      # 2x2 hemifield patch means -
            assert not self.feat_mode, "quad_pool needs pixel patches"
        d_base = (3 if pool == "cls_mean_focal" else 2) * d
        d_extra = 2 * len(self.ms_layers) * d + d2 \
            + (4 * d if self.quad_pool else 0)
        # exp-058c/060: function-preserving expansion — base proj keeps its
        # legacy width (loads warm from tap-free ckpts), extra features enter
        # through a parallel zero-init branch that contributes exactly 0 at
        # init, so a warm-started model IS the source model on step one.
        self.proj_adapter = bool(proj_adapter) and d_extra > 0
        if self.proj_adapter:
            self.proj = nn.Sequential(nn.LayerNorm(d_base),
                                      nn.Linear(d_base, d_model))
            self.tap_proj = nn.Sequential(nn.LayerNorm(d_extra),
                                          nn.Linear(d_extra, d_model))
            nn.init.zeros_(self.tap_proj[1].weight)
            nn.init.zeros_(self.tap_proj[1].bias)
        else:
            self.proj = nn.Sequential(nn.LayerNorm(d_base + d_extra),
                                      nn.Linear(d_base + d_extra, d_model))
        self.plane_emb = nn.Embedding(4, d_model)
        self.fluid_emb = nn.Embedding(3, d_model)
        self.fatsat_emb = nn.Embedding(3, d_model)
        self.pos_mlp = nn.Sequential(nn.Linear(1, 64), nn.GELU(),
                                     nn.Linear(64, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           dropout=drop, batch_first=True,
                                           norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, n_mix)
        self.queries = nn.Parameter(torch.randn(N_TARGET, d_model) * 0.02)
        self.key = nn.Linear(d_model, d_model)
        self.acmil_p = 0.0                     # exp-042: set via flag
        self.acmil_k = 1
        self.zoom_px = 0                       # exp-051: set via flag
        self.saff = (0.0, 10.0, 0.12, 0.08)    # study affine aug: set via flag
        self.dres_p, self.dres_min = 0.0, 0.6  # down-up aug: set via flag
        self.recon_p, self.recon_cfg = 0.0, (0.035, 1.0, 0.6)  # recon aug: set via flag
        self.use_xattn = bool(use_xattn)       # exp-043: patch-token decoder
        if self.use_xattn:                     # params exist ONLY when on
            d_bb = bb.config.hidden_size       # (keeps legacy ckpts clean)
            self.px_key = nn.Linear(d_bb, d_model)
            self.px_val = nn.Linear(d_bb, d_model)
            self.px_gate = nn.Parameter(torch.zeros(N_TARGET, 1))
        self.out_w = nn.Parameter(torch.randn(N_TARGET, d_model) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(N_TARGET))
        self.drop = nn.Dropout(drop)
        nm = getattr(bb, "norm_mean", (0.485, 0.456, 0.406))
        ns = getattr(bb, "norm_std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(nm).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(ns).view(1, 3, 1, 1))

    def encode(self, x, return_patches=False):
        if self.ms_layers:                      # exp-058: multi-scale pooled
            o = self.bb(pixel_values=x, output_hidden_states=True)
            out = o.last_hidden_state
        else:
            out = self.bb(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        base = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":       # exp-044: top-k patch mean
            k = max(1, patch.shape[1] // 8)
            base.append(patch.topk(k, dim=1).values.mean(1))
        extra = []
        for li in self.ms_layers:               # DETACHED intermediates (no
            h = o.hidden_states[li + 1].detach()  # grad amplification; user-
            extra += [h[:, 0], h[:, 1:].mean(1)]  # validated pattern)
        if self.bb2 is not None:                # exp-060: frozen aux tower -
            with torch.no_grad():               # stationary by construction
                o2 = self.bb2(pixel_values=x).last_hidden_state
            extra += [o2[:, 0], o2[:, 1:].mean(1)]
        if self.quad_pool:                      # exp-065 QUAD: compartment-
            P, D = patch.shape[1], patch.shape[2]   # sized 2x2 means; fixed
            g = int(round(P ** 0.5))                # anatomical identity via
            assert g * g == P and g % 2 == 0, \
                "quad_pool needs an even square patch grid"
            q = patch.reshape(-1, 2, g // 2, 2, g // 2, D).mean(dim=(2, 4))
            extra += list(q.reshape(-1, 4, D).unbind(1))
        if self.proj_adapter:                   # base block contiguous first
            parts = base + extra
        else:                                   # legacy layout: focal LAST
            parts = base[:2] + extra + base[2:]
        pooled = torch.cat(parts, dim=-1)
        return (pooled, patch) if return_patches else pooled

    def forward(self, imgs, meta, mask, return_hpix=False,
                return_study=False, return_ctx=False, return_attn=False):
        B, L = imgs.shape[:2]
        if self.feat_mode:                     # exp-061: pooled features in,
            h = self.proj(imgs.float())        # no backbone in the graph
            patches = None
        else:
            x = imgs.reshape(B * L, *imgs.shape[2:])
            # float input: .float() is a no-copy view -> div_ would MUTATE the
            # caller's tensor (double-div when a second forward reuses the
            # batch, e.g. the exp-047 EMA scoring pass). uint8 keeps in-place.
            x = (x.float() / 255.0) if x.dtype != torch.uint8 \
                else x.float().div_(255.0)
            if self.zoom_px:                   # exp-051: center-zoom (same
                Hs = x.shape[-1]               # grid -> pos-embeds untouched;
                c0 = (Hs - self.zoom_px) // 2  # applies at train AND eval)
                x = F.interpolate(x[..., c0:c0 + self.zoom_px,
                                    c0:c0 + self.zoom_px], size=(Hs, Hs),
                                  mode="bilinear", align_corners=False)
            if self.training and self.saff[0] > 0:  # train-only study affine aug
                x = _study_affine(x, B, *self.saff)
            if self.training and self.dres_p > 0:   # train-only down-up aug
                x = _down_up(x, B, self.dres_p, self.dres_min)
            if self.training and self.recon_p > 0:  # train-only recon aug
                x = _recon_aug(x, B, self.recon_p, *self.recon_cfg)
            x = (x - self.mean) / self.std
            if self.use_xattn:                 # exp-043
                enc, patches = self.encode(x, return_patches=True)
            else:
                enc, patches = self.encode(x), None
            if self.proj_adapter:              # warm base + zero-init taps
                db = self.proj[0].normalized_shape[0]
                h = self.proj(enc[..., :db]) + self.tap_proj(enc[..., db:])
            else:
                h = self.proj(enc)
            h = h.reshape(B, L, -1)
        hpix = h if return_hpix else None    # exp-038: PRE-metadata features
        h = (h + self.plane_emb(meta[..., 0].long().clamp(0, 3))
             + self.fluid_emb(meta[..., 1].long().clamp(0, 2))
             + self.fatsat_emb(meta[..., 2].long().clamp(0, 2))
             + self.pos_mlp(meta[..., 3:4]))
        pad = mask < 0.5
        h = self.mixer(h, src_key_padding_mask=pad)
        k = self.key(h)                                          # [B,L,D]
        att = torch.einsum("bld,td->btl", k, self.queries) / math.sqrt(k.shape[-1])
        att = att.masked_fill(pad.unsqueeze(1), -1e4)
        if self.training and self.acmil_p > 0:   # exp-042: ACMIL top-k mask
            top = att.topk(self.acmil_k, dim=-1).indices
            drop = (torch.rand(att.shape[:2], device=att.device)
                    < self.acmil_p)
            m_top = torch.zeros_like(att, dtype=torch.bool).scatter_(
                -1, top, True)
            att = att.masked_fill(m_top & drop.unsqueeze(-1), -1e4)
        att = att.softmax(-1)
        ctx = self.drop(torch.einsum("btl,bld->btd", att, h))  # [B,T,D]
        if self.use_xattn:                     # exp-043: patch cross-attn,
            P = patches.shape[1]               # zero-gated residual on ctx
            pk = self.px_key(patches).reshape(B, L * P, -1)
            pv = self.px_val(patches).reshape(B, L * P, -1)
            pk = pk + h.repeat_interleave(P, dim=1)   # slice conditioning
            a2 = torch.einsum("td,bnd->btn", self.queries, pk) / math.sqrt(
                pk.shape[-1])
            pad_px = pad.repeat_interleave(P, dim=1)
            a2 = a2.masked_fill(pad_px.unsqueeze(1), -1e4).softmax(-1)
            px_ctx = torch.einsum("btn,bnd->btd", a2, pv)
            ctx = ctx + self.px_gate.unsqueeze(0) * self.drop(px_ctx)
        z = (ctx * self.out_w.unsqueeze(0)).sum(-1) + self.out_b
        if return_attn:                      # exp-047 arm B: [B,T,L] slice attn
            return z, att
        if return_ctx:                       # exp-038 arm 2: per-target feats
            return z, ctx
        if return_study:                     # exp-038 arm 3: study embedding
            semb = ((h * mask.unsqueeze(-1)).sum(1)
                    / mask.sum(1, keepdim=True).clamp(min=1.0))
            return z, semb
        return (z, hpix) if return_hpix else z


class AuxGeoHeads(nn.Module):
    """exp-038 arm 1 (AUX-GEO): per-slice acquisition heads on the PIXEL-side
    features (pre-metadata-embedding), so the answer is not in the input.
    Separate module — NOT part of SliceBagModel.state_dict, so checkpoints and
    submission kernels are untouched. Targets come from the PRE-augmentation
    meta (true acquisition labels; TA bag ops may perturb the input meta)."""

    def __init__(self, d_model=384):
        super().__init__()
        self.plane = nn.Linear(d_model, 4)
        self.fluid = nn.Linear(d_model, 3)
        self.fatsat = nn.Linear(d_model, 3)
        self.relpos = nn.Linear(d_model, 1)

    def loss(self, hpix, meta_orig, mask):
        """hpix [B,L,D]; meta_orig [B,L,4] pre-TA; mask [B,L]. Returns
        (loss over valid tokens, stats dict of per-head accuracy / L1)."""
        v = mask.reshape(-1) > 0.5
        h = hpix.reshape(-1, hpix.shape[-1])[v]
        m = meta_orig.reshape(-1, 4)[v]
        stats = {}
        total = 0.0
        for name, head, n_cls in (("plane", self.plane, 4),
                                  ("fluid", self.fluid, 3),
                                  ("fatsat", self.fatsat, 3)):
            logits = head(h)
            tgt = m[:, ("plane", "fluid", "fatsat").index(name)].long().clamp(
                0, n_cls - 1)
            total = total + F.cross_entropy(logits.float(), tgt)
            stats[name] = (logits.argmax(-1) == tgt).float().mean().item()
        rp = self.relpos(h).squeeze(-1).float()
        rp_tgt = m[:, 3].float()
        total = total + F.l1_loss(rp, rp_tgt)
        stats["relpos_l1"] = F.l1_loss(rp, rp_tgt).item()
        return total / 4.0, stats


class AuxTxtHead(nn.Module):
    """exp-038 arm 3 (AUX-TXT): ConVIRT-style joint report alignment. Projects
    the study embedding into the frozen LaBSE space; InfoNCE against ALL 4,407
    report embeddings as negatives every step (they are known up front — no
    queue needed). Separate module: NOT in the checkpoint; vanishes at test."""

    def __init__(self, d_model=384, d_txt=768, tau=0.07):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, 512), nn.GELU(),
                                  nn.Linear(512, d_txt))
        self.tau = tau

    def loss(self, semb, emb_all, rows):
        """semb [B,D_model]; emb_all [N,768] L2-normalized; rows [B] target
        row indices into emb_all."""
        q = F.normalize(self.proj(semb).float(), dim=-1)
        logits = q @ emb_all.t().float() / self.tau          # [B, N]
        la = F.cross_entropy(logits, rows)
        acc = (logits.argmax(-1) == rows).float().mean().item()
        return la, {"txt_acc": acc}


class AuxSevHead(nn.Module):
    """exp-038 arm 2 (AUX-SEV): CORN rank-consistent ordinal heads on the
    per-target decoder features ctx [B,T,D]. 5 severity levels (0-4) -> 4
    cumulative binary tasks per target; task k trains only on cells with
    grade >= k (CORN conditional sets); NaN grades masked entirely.
    Separate module — NOT in the checkpoint."""

    def __init__(self, d_model=384, n_targets=12, n_thresh=4):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_targets, n_thresh, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_targets, n_thresh))
        self.n_thresh = n_thresh

    def loss(self, ctx, sev):
        """ctx [B,T,D]; sev [B,T] float grades 0-4 with NaN = unlabeled."""
        logits = torch.einsum("btd,tkd->btk", ctx.float(), self.w) + self.b
        known = torch.isfinite(sev)
        total, n_terms, correct, n_pred = 0.0, 0, 0.0, 0
        for k in range(self.n_thresh):
            cond = known & (torch.nan_to_num(sev, nan=-1.0) >= k)  # CORN set
            if cond.sum() == 0:
                continue
            lab = (sev > k).float()
            lk = logits[..., k]
            total = total + _soft_gce_bce(
                lk[cond], lab[cond])
            n_terms += 1
            correct += (((lk[cond] > 0) == (lab[cond] > 0.5)).float().sum()
                        .item())
            n_pred += int(cond.sum().item())
        if n_terms == 0:
            return ctx.sum() * 0.0, {"sev_acc": float("nan")}
        return total / n_terms, {"sev_acc": correct / max(n_pred, 1)}


@torch.no_grad()
def predict(model, bank, studies, dev, bag, n_bags, seed, batch=8, workers=4):
    model.eval()
    ds = BagDataset(bank, studies, bag, seed, train=False, n_bags=n_bags)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    acc = np.zeros((len(studies), N_TARGET), np.float64)
    cnt = np.zeros(len(studies), np.float64)
    for imgs, meta, mask, sidx in dl:
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            z = model(imgs.to(dev, non_blocking=True), meta.to(dev),
                      mask.to(dev)).float()
        p = torch.sigmoid(z).cpu().numpy()
        for q, i in enumerate(sidx.numpy()):
            acc[i] += p[q]
            cnt[i] += 1
    return acc / np.maximum(cnt, 1)[:, None]


def make_lr_lambda_restarts(total, warm_frac, head_only_steps, cycle_steps,
                            cycle_decay=1.0, is_backbone=False):
    """exp-037 arm B: [head-only hold (backbone)] -> linear warmup -> cosine
    WITH WARM RESTARTS (SGDR, fixed cycle length T_mult=1). Each cycle re-peaks
    at cycle_decay**k of the base LR and cosines to 0 within the cycle.
    cycle_steps is clamped to the post-hold span, so cycle_steps >= span
    degenerates EXACTLY to the plain cosine of make_lr_lambda."""
    hold = int(head_only_steps) if is_backbone else 0
    span = max(total - hold, 1)
    warm = int(round(warm_frac * span))
    cyc = max(1, min(int(cycle_steps), span))

    def f(step):
        if step < hold:
            return 0.0
        t = min(step - hold, span - 1)
        wf = min(1.0, (t + 1) / warm) if warm > 0 else 1.0
        k, pos = divmod(t, cyc)
        return (wf * (cycle_decay ** k)
                * 0.5 * (1.0 + math.cos(math.pi * pos / cyc)))
    return f


# ---------------------------------------------------------------- train
_original_ema_update = ema_update


@torch.no_grad()
def ema_update(ema, model, mu, step):
    _original_ema_update(ema, model, mu, step)
    source_buffers = dict(model.named_buffers())
    for name, buffer in ema.named_buffers():
        buffer.copy_(source_buffers[name])


def _finite_json(value):
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _atomic_json(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(_finite_json(value), stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _artifact_digest(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    _atomic_json({'file': Path(path).name, 'bytes': Path(path).stat().st_size,
                  'sha256': digest.hexdigest()}, str(path) + '.sha256.json')


def _atomic_numpy(value, path):
    assert np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(), 'invalid predictions'
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    _artifact_digest(path)


def _atomic_save(obj, path):
    """torch.save to <path>.tmp then os.replace (same filesystem) so a kill
    mid-write never leaves a truncated state file (exp-066)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    _artifact_digest(path)


def _load_state(path):
    """Load a full-state file; if it is unreadable (Drive upload cut short),
    fall back to the newest intact state_ep*.pt beside it."""
    p = Path(path)
    for q in p.parent.glob("state_*.pt.tmp"):
        q.unlink(missing_ok=True)
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:                       # noqa: BLE001
        cands = sorted(p.parent.glob("state_ep*.pt"),
                       key=lambda q: int(q.stem[len("state_ep"):]))
        for q in reversed(cands):
            try:
                st = torch.load(q, map_location="cpu", weights_only=False)
                log(f"resume: {p.name} unreadable ({e}); using {q.name}")
                return st
            except Exception:                    # noqa: BLE001
                continue
        raise


RESUME_KEYS = (
    "full_finetune", "grad_checkpoint", "data_fingerprint",
    "cosine_epochs", "cosine_floor", "replay_drop_epochs", "replay_factor",
    "replay_floor", "replay_warmup_epochs",
    "backbone", "unfreeze_last", "lora_r", "lora_alpha", "lora_targets",
    "d_model", "n_mix",
    "decoder", "pool", "ms_layers", "aux_backbone", "proj_adapter",
    "quad_pool", "zoom_px", "bag", "batch", "accum", "seed", "fold",
    "lr_backbone", "lr_head", "wd", "llrd", "ema", "warmup_frac",
    "head_only_ep", "sched", "cycle_ep", "cycle_decay", "adaptive_unfreeze",
    "adap_span_ep", "adap_min_lr", "adap_stages", "adap_min_delta",
    "xcontrast_p", "bag_mode", "bag_jitter",
    "full_data", "select_on", "pos_weight", "topk", "hard_labels",
    "tri_gap", "rank_lambda", "rank_queue", "rank_margin", "rank_conf",
    "whole_study", "slice_drop", "slice_drop_end", "drop_ramp_ep0",
    "drop_ramp_ep1", "max_slices", "trivial_aug", "faa_policy", "ta_p_bag",
    "flip_p", "conf_weight", "aux_geo", "aux_txt", "aux_sev", "chop_pos",
    "ep_drop_protect", "labels_csv", "eval_labels_csv", "gold2_csv",
    "weight_csv", "holdout_csv")


def run_fold(fold, studies, bank, soft, soft_eval, gold_y, gold_idx, folds_map,
             args, dev):
    assert len(gold_idx) == 0 and gold_y.shape == (0, N_TARGET), 'gold evaluation disabled in campaign'
    assert str(args.select_on) == 'val' and str(args.plateau_metric) == 'val'
    seed = int(args.seed)
    torch.manual_seed(seed * 100 + fold)
    torch.cuda.manual_seed_all(seed * 100 + fold)
    if bool(getattr(args, "hard_labels", False)):      # exp-074
        # Threshold the TRAINING targets at 0.5, keeping unknown (NaN) cells
        # unknown. soft_eval is untouched. NOTE: conf_weight is derived from
        # the same probability (0.3 + 0.7|2p-1|), so with 0/1 targets it
        # collapses to a constant - hardening removes the soft signal from
        # BOTH the target and the weight, which is the honest "hard" arm.
        soft = np.where(np.isfinite(soft), (soft > 0.5).astype(np.float32),
                        np.float32("nan"))
        log("HARD LABELS: training targets thresholded at 0.5 "
            "(conf_weight becomes constant); soft_eval untouched")
    is_val = np.array([folds_map[s] == fold for s in studies])
    is_gold = np.zeros(len(studies), bool)
    is_gold[gold_idx] = True
    tr_idx = np.where(~is_val & ~is_gold)[0]
    va_idx = np.where(is_val & ~is_gold)[0]
    if bool(getattr(args, "full_data", False)):      # exp-073 (I-107)
        # Every non-gold study trains. The val fold is then INSIDE train, so
        # va_auc becomes a fit metric and CANNOT select; gold-58 must.
        tr_idx = np.where(~is_gold)[0]
        # exp-077: a --holdout_csv jury is pulled OUT of tr_idx below and
        # replaces va_auc, so it is a clean selector even under --full_data.
        assert (str(getattr(args, "select_on", "val")) == "gold"
                or str(getattr(args, "holdout_csv", "") or "")),             ("--full_data requires --select_on gold, or a --holdout_csv "
             "jury (the fold-0 val split is inside train and cannot select)")
    holdout = None            # exp-063: jury holdout selects (Raptor-style)
    hold_csv = str(getattr(args, "holdout_csv", "") or "")
    if hold_csv:
        ph = Path(hold_csv)
        if not ph.is_file():
            ph = Path(getattr(args, "csv_dir", ".")) / hold_csv
        hdf = pd.read_csv(ph).set_index("StudyInstanceUID")
        pos_h = {s: i for i, s in enumerate(studies)}
        h_st = [s for s in hdf.index
                if s in pos_h and not is_gold[pos_h[s]]]
        h_idx = np.array([pos_h[s] for s in h_st])
        h_y = hdf.reindex(h_st)[TARGETS].values.astype(float)
        assert not str(getattr(args, "gold2_csv", "") or ""), \
            "holdout_csv and gold2_csv are mutually exclusive (index clash)"
        holdout = (h_idx, h_y, np.isfinite(h_y))
        n0 = len(tr_idx)
        tr_idx = tr_idx[~np.isin(tr_idx, h_idx)]
        log(f"HOLDOUT-JURY ON (exp-063): {len(h_idx)} labeled studies "
            f"removed from train ({n0} -> {len(tr_idx)}); holdout AUC "
            f"SELECTS the ckpt; gold-58 stays report-only")
    # P3.1 NaN-aware labels: unknown train cells -> zero weight; unknown or
    # tied (0.5) eval cells -> excluded from the val AUC.
    soft = np.asarray(soft, dtype=np.float32)
    label_mask = torch.from_numpy(np.isfinite(soft))
    soft = np.nan_to_num(soft, nan=0.5)
    y_eval_bin, y_eval_valid = eval_targets(soft_eval)
    flip_p = float(getattr(args, "flip_p", 0.0))
    llrd = float(getattr(args, "llrd", 1.0))
    head_only_ep = int(getattr(args, "head_only_ep", 0))
    warm_frac = float(getattr(args, "warmup_frac", 0.05))
    log(f"slicebag fold {fold}: train {len(tr_idx)} val {len(va_idx)} gold "
        f"{len(gold_idx)} | bag {args.bag} | seed {seed} | "
        f"{torch.cuda.get_device_name(0)}")
    log(f"P3 flags: flip_p {flip_p} | llrd {llrd} | warmup {warm_frac} | "
        f"head_only_ep {head_only_ep} | unknown train cells "
        f"{int((~label_mask).sum())} | excluded eval cells "
        f"{int((~y_eval_valid).sum())}")
    # gold_v2 dual instrument: gold-58 rows are held out of ALL training so
    # they are clean for every fold; NEW gold_v2 studies are ordinary train
    # studies, clean only when they sit in THIS fold's val split.
    gold2 = None
    gold2_csv = str(getattr(args, "gold2_csv", "") or "")
    if gold2_csv:
        p2 = Path(gold2_csv)
        if not p2.is_file():
            p2 = Path(getattr(args, "csv_dir", ".")) / gold2_csv
        g2 = pd.read_csv(p2).set_index("StudyInstanceUID")[TARGETS]
        gd_pos = {studies[i]: k for k, i in enumerate(gold_idx)}
        va_pos = {studies[i]: k for k, i in enumerate(va_idx)}
        g2_src, g2_rows = [], []
        for s, row in g2.iterrows():
            if s in gd_pos:
                g2_src.append(("g", gd_pos[s]))
                g2_rows.append(row.values)
            elif s in va_pos:
                g2_src.append(("v", va_pos[s]))
                g2_rows.append(row.values)
        g2_raw = np.asarray(g2_rows, dtype=np.float32)
        g2_valid = np.isfinite(g2_raw)
        gold2 = (g2_src, np.nan_to_num(g2_raw, nan=0.0).astype(int), g2_valid)
        n_g = sum(1 for t, _ in g2_src if t == "g")
        log(f"gold2 [{p2.name}]: {len(g2_src)} eligible studies ({n_g} "
            f"gold-58 + {len(g2_src) - n_g} fold-{fold} val) | "
            f"{int(g2_valid.sum())} labeled cells | "
            f"{len(g2) - len(g2_src)} train-split studies excluded")
    # LLM-agreement sample weights (weights_v01): per-cell multipliers on the
    # BCE weight — down-weight cells the report readers disagree on.
    w_cell = None
    weight_csv = str(getattr(args, "weight_csv", "") or "")
    if weight_csv:
        pw = Path(weight_csv)
        if not pw.is_file():
            pw = Path(getattr(args, "csv_dir", ".")) / weight_csv
        w_arr = (pd.read_csv(pw).set_index("StudyInstanceUID")[TARGETS]
                 .reindex(studies).values.astype(np.float32))
        n_miss = int(np.isnan(w_arr).all(axis=1).sum())
        w_cell = torch.from_numpy(np.nan_to_num(w_arr, nan=1.0))
        log(f"sample weights [{pw.name}]: min {np.nanmin(w_arr):.2f} mean "
            f"{np.nanmean(w_arr):.2f} | {n_miss} studies default to 1.0")
    # exp-047 arm A POS-LLR: zero-weight evidence-free POSITIVE draws (the
    # EMA model sees the same dropped view; p < tau on a positive cell means
    # this draw's evidence is gone - corruption is per-draw, the cell returns
    # next epoch). Budgets capped per target from the drop^(k+2) math.
    chop_on = bool(getattr(args, "chop_pos", False))
    if chop_on:
        assert getattr(args, "whole_study", False) and \
            float(getattr(args, "slice_drop", 0)) > 0, \
            "chop_pos requires --whole_study with slice_drop > 0"
        CHOP_FOCAL, CHOP_MID = (2, 3, 10, 11), (0, 1, 9)
        chop_rho = np.full(12, float(getattr(args, "chop_rho_dif", 0.01)))
        chop_rho[list(CHOP_MID)] = float(getattr(args, "chop_rho_mid", 0.05))
        chop_rho[list(CHOP_FOCAL)] = float(getattr(args, "chop_rho_focal",
                                                   0.15))
        chop_hi = float(getattr(args, "chop_tau", 0.5))
        chop_ramp = max(1, int(getattr(args, "chop_ramp_ep", 4)))
        chop_min_n = int(getattr(args, "chop_min_n", 50))
        chop_pos_total = np.maximum(
            (soft[tr_idx] > 0.5).sum(0).astype(np.float64), 1.0)
        chop_cnt = np.zeros(12)
        # v2 RANK FORM (the registered primary; v1's global tau sat below
        # common-target prevalence and never fired): per-target threshold =
        # the (rho*ramp)-quantile of the PREVIOUS epoch's positive EMA
        # predictions. Epoch 0 chops nothing (no history) = natural ramp-in.
        chop_qtau = np.zeros(12)
        chop_qtau_t = torch.zeros(12, device=dev)
        chop_pacc = [[] for _ in range(12)]
        log(f"POS-LLR chop ON (rank form): ceiling {chop_hi} | rho f/m/d "
            f"{chop_rho[2]}/{chop_rho[0]}/{chop_rho[4]} | ramp {chop_ramp} "
            f"ep | train positives {chop_pos_total.astype(int).tolist()}")
    # exp-047 arm B EP-DROP: evidence-preserving slice dropout (protect one
    # of the top-k EMA-attended slices per positive target from total wipe)
    prot_on = bool(getattr(args, "ep_drop_protect", False))
    if prot_on:
        assert getattr(args, "whole_study", False) and \
            float(getattr(args, "slice_drop", 0)) > 0, \
            "ep_drop_protect requires --whole_study with slice_drop > 0"
        assert not chop_on, "arms A and B never run together (pre-reg)"
        prot_pos = {studies[i]: (soft[i] > 0.5) for i in tr_idx}
        prot_refresh = max(1, int(getattr(args, "ep_prot_refresh", 5)))
        log(f"EP-DROP protect ON: p {float(getattr(args, 'ep_prot_p', 0.9))} "
            f"| top-{int(getattr(args, 'ep_prot_k', 3))} | refresh every "
            f"{prot_refresh} ep")

    ms_layers = tuple(int(x) for x in
                      str(getattr(args, "ms_layers", "")).split(",") if x)
    aux_bb = str(getattr(args, "aux_backbone", "") or "")
    model = SliceBagModel(args.backbone, args.unfreeze_last, args.d_model,
                          args.n_mix,
                          use_xattn=(getattr(args, "decoder", "mil")
                                     == "xattn"),
                          pool=getattr(args, "pool", "cls_mean"),
                          ms_layers=ms_layers, aux_backbone=aux_bb,
                          proj_adapter=int(getattr(args, "proj_adapter", 0)),
                          lora_r=int(getattr(args, "lora_r", 0)),
                          lora_alpha=float(getattr(args, "lora_alpha", 16)),
                          lora_targets=str(getattr(args, "lora_targets",
                                                   "qv")),
                          quad_pool=int(getattr(args, "quad_pool",
                                                0)),
                          full_finetune=bool(getattr(args, "full_finetune",
                                                    False))).to(dev)
    if bool(getattr(args, "grad_checkpoint", False)):
        inner = getattr(model.bb, "m", model.bb)
        if hasattr(inner, 'set_grad_checkpointing'):
            inner.set_grad_checkpointing(True)
        elif hasattr(inner, 'gradient_checkpointing_enable'):
            inner.gradient_checkpointing_enable()
        else:
            raise ValueError('backbone does not support gradient checkpointing')
        log('GRADIENT CHECKPOINTING enabled')
    if model.full_finetune:
        frozen = [n for n, p in model.bb.named_parameters() if not p.requires_grad]
        assert not frozen, f'full_finetune left frozen parameters: {frozen}'
        log('FULL FINETUNE: every backbone parameter trainable, including stem/transitions')
    if model.quad_pool:
        log("QUAD POOL ON (exp-065, jury winner): 2x2 hemifield patch means "
            "join the decoder input (TL,TR,BL,BR; ~65mm compartments at "
            "336/130) - spatial identity the global mean destroys")
    if model.lora_r > 0:
        n_lp = sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and (".A" in n or ".B" in n))
        log(f"LoRA ON (I-095): r {model.lora_r} | alpha "
            f"{float(getattr(args, 'lora_alpha', 16))} | "
            f"{model._lora_n_inj} attention Linears wrapped | "
            f"{n_lp / 1e6:.2f}M low-rank params | backbone base FROZEN | "
            f"LoRA LR = lr_backbone via block groups (use --llrd 1.0)")
    if model.proj_adapter:
        log("PROJ ADAPTER ON: base proj at legacy width (warm-loadable); "
            "taps/aux enter via zero-init parallel branch -> a lax warm "
            "start reproduces the source model EXACTLY at step 0")
    if model.feat_mode:                          # exp-061
        assert bank.feats is not None, "feat: backbone needs --feat_dir"
        assert bank.feat_dim == 2 * model.bb.config.hidden_size, \
            (bank.feat_dim, model.bb.config.hidden_size)
        assert not (getattr(args, "trivial_aug", False)
                    or getattr(args, "faa_policy", "")
                    or int(getattr(args, "zoom_px", 0))
                    or int(getattr(args, "adaptive_unfreeze", 0))
                    or float(getattr(args, "flip_p", 0))), \
            "feat mode: pixel-space augs/zoom/unfreeze unavailable"
        log(f"FEAT MODE: decoder-only on precomputed pooled features "
            f"(dim {bank.feat_dim}; no backbone in the graph)")
    if aux_bb:
        log(f"AUX TOWER ON (frozen forever): {aux_bb} -> +2x"
            f"{model.bb2.config.hidden_size} pooled dims into proj "
            f"(shares the main normalization - pair towers with matching "
            f"input stats)")
    if model.pool != "cls_mean":
        log(f"ENCODER POOL: {model.pool}")
    if ms_layers:
        log(f"MULTI-SCALE ON: detached pooled features from blocks "
            f"{list(ms_layers)} + final (proj widened)")
    if model.use_xattn:
        log("DECODER: patch-token cross-attention (zero-gated residual)")
    # exp-066: exact full-state resume (a session death is a pause, not an
    # I-077 restart). "auto" = <out_dir>/state_last.pt if present.
    resume_path = str(getattr(args, "resume_state", "") or "")
    if resume_path == "auto":
        cand = Path(args.out_dir) / "state_last.pt"
        resume_path = str(cand) if cand.is_file() else ""
        log(f"resume_state auto -> {resume_path or 'fresh start'}")
    st = None
    if resume_path:
        st = _load_state(resume_path)
        if getattr(args, "init_backbone", "") or getattr(args, "init_full", ""):
            log(f"init_backbone/init_full IGNORED: resuming from {resume_path}")
    if getattr(args, "init_backbone", "") and st is None:   # exp-041
        sd = torch.load(args.init_backbone, map_location="cpu",
                        weights_only=False)["state_dict"]
        bb_sd = {k: v for k, v in sd.items() if k.startswith("bb.")}
        assert bb_sd, "init_backbone ckpt holds no bb.* tensors"
        missing, unexpected = model.load_state_dict(bb_sd, strict=False)
        assert not unexpected, unexpected[:3]
        log(f"backbone warm-started from {args.init_backbone} "
            f"({len(bb_sd)} tensors; heads fresh)")
    if getattr(args, "init_full", "") and st is None:   # exp-045c warm start
        assert not getattr(args, "init_backbone", ""), \
            "init_full and init_backbone are mutually exclusive"
        ck0 = torch.load(args.init_full, map_location="cpu",
                         weights_only=False)
        if int(getattr(args, "init_lax", 0)):    # exp-058b: shape-tolerant
            sd, own = ck0["state_dict"], model.state_dict()
            keep = {k: v for k, v in sd.items()
                    if k in own and own[k].shape == v.shape}
            skipped = sorted(set(sd) - set(keep)) + \
                sorted(k for k in own if k not in sd)
            missing, unexpected = model.load_state_dict(keep, strict=False)
            log(f"LAX warm start from {args.init_full}: {len(keep)} tensors "
                f"loaded, fresh/skipped: {skipped}")
        else:
            model.load_state_dict(ck0["state_dict"], strict=True)
        log(f"FULL warm start from {args.init_full} "
            f"(source metrics: {ck0.get('metrics', {})})")
    if getattr(args, "acmil_p", 0) > 0:          # exp-042
        model.acmil_p = float(args.acmil_p)
        model.acmil_k = int(args.acmil_k)
        log(f"ACMIL ON: p {model.acmil_p} k {model.acmil_k} (train-mode only)")
    if int(getattr(args, "zoom_px", 0)) > 0:     # exp-051
        model.zoom_px = int(args.zoom_px)
        log(f"ZOOM-CROP ON: central {model.zoom_px}px resized back "
            f"(train AND eval; effective FOV "
            f"{150 * model.zoom_px / 224:.0f} mm at 224)")
    if float(getattr(args, "saff_p", 0.0)) > 0:
        model.saff = (float(args.saff_p), float(args.saff_rot),
                      float(args.saff_scale), float(args.saff_shift))
        log(f"STUDY AFFINE AUG ON: p/rot/scale/shift {model.saff} (train only)")
    if float(getattr(args, "dres_p", 0.0)) > 0:
        model.dres_p = float(args.dres_p)
        model.dres_min = float(args.dres_min)
        log(f"DOWN-UP RESAMPLE ON: p {model.dres_p} per study, scale "
            f"[{model.dres_min}, 1] (train only)")
    if float(getattr(args, "recon_p", 0.0)) > 0:
        model.recon_p = float(args.recon_p)
        model.recon_cfg = (float(args.recon_noise), float(args.recon_sharp),
                           float(args.recon_smooth))
        log(f"RECON AUG ON: p {model.recon_p} per study, noise/sharp/smooth "
            f"{model.recon_cfg} (train only)")
    _SOFT_GCE_CFG[0] = float(getattr(args, "soft_gce", 0.0))
    _SOFT_GCE_CFG[1] = float(getattr(args, "gce_q", 0.7))
    if _SOFT_GCE_CFG[0] > 0:
        log(f"SOFT_GCE ON: loss = (1-lam)*BCE + lam*GCE, lam/q {_SOFT_GCE_CFG} "
            f"(grad-enabled passes only)")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    # P3.4 LLRD param groups over model.bb blocks; everything else = head
    groups = build_param_groups(model, args.lr_backbone, args.lr_head,
                                args.wd, llrd, backbone_attr="bb")
    bb_params = [p for g in groups if g["kind"] == "bb" for p in g["params"]]
    head_params = [p for g in groups if g["kind"] == "head"
                   for p in g["params"]]
    n_tr = sum(p.numel() for p in bb_params + head_params)
    log(f"trainable {n_tr / 1e6:.2f}M (backbone {sum(p.numel() for p in bb_params) / 1e6:.2f}M)")
    aux = None
    if getattr(args, "aux_geo", False):          # exp-038 arm 1
        aux = AuxGeoHeads(args.d_model).to(dev)
        groups.append({"params": list(aux.parameters()), "lr": args.lr_head,
                       "weight_decay": args.wd, "kind": "head"})
        head_params = head_params + list(aux.parameters())
        log(f"AUX-GEO ON: per-slice plane/fluid/fatsat/relpos heads on "
            f"pixel-side features, lambda {getattr(args, 'aux_lambda', 0.1)}")
    aux_txt, txt_emb, txt_rows = None, None, None
    if getattr(args, "aux_txt", ""):             # exp-038 arm 3
        assert aux is None, "--aux_geo and --aux_txt: run arms solo"
        emb = np.load(args.aux_txt + ".npy")
        emb_st = json.load(open(args.aux_txt + "_studies.json"))
        row_of_emb = {s: i for i, s in enumerate(emb_st)}
        txt_emb = torch.from_numpy(emb).float().to(dev)
        txt_rows = torch.tensor([row_of_emb[s] for s in studies],
                                dtype=torch.long)
        aux_txt = AuxTxtHead(args.d_model, emb.shape[1],
                             float(getattr(args, "aux_txt_tau", 0.07))).to(dev)
        groups.append({"params": list(aux_txt.parameters()),
                       "lr": args.lr_head, "weight_decay": args.wd,
                       "kind": "head"})
        head_params = head_params + list(aux_txt.parameters())
        log(f"AUX-TXT ON: InfoNCE vs {emb.shape[0]} report embeddings "
            f"({emb.shape[1]}d), tau {getattr(args, 'aux_txt_tau', 0.07)}, "
            f"lambda {getattr(args, 'aux_txt_lambda', 0.1)}")
    aux_sev, sev_t = None, None
    if getattr(args, "aux_sev", ""):             # exp-038 arm 2
        assert aux is None and aux_txt is None, "run aux arms solo"
        sev_df = pd.read_csv(args.aux_sev).set_index("StudyInstanceUID")
        sev_np = sev_df.reindex(studies)[TARGETS].values.astype(np.float32)
        sev_t = torch.from_numpy(sev_np)
        aux_sev = AuxSevHead(args.d_model, len(TARGETS)).to(dev)
        groups.append({"params": list(aux_sev.parameters()),
                       "lr": args.lr_head, "weight_decay": args.wd,
                       "kind": "head"})
        head_params = head_params + list(aux_sev.parameters())
        cov = float(np.isfinite(sev_np).mean())
        log(f"AUX-SEV ON: CORN ordinal heads (5 levels), coverage "
            f"{cov:.2f}, lambda {getattr(args, 'aux_sev_lambda', 0.1)}")
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    accum = max(1, int(getattr(args, "accum", 1)))    # exp-040: grad accum
    steps_per_ep = len(tr_idx) // (args.batch * accum)   # optimizer steps/ep
    steps = args.epochs * steps_per_ep
    assert steps_per_ep > 0, 'not enough training studies for batch * accum'
    # P3.3 head-only hold (backbone) -> linear warmup -> cosine
    # exp-037: --sched cosine_restarts = SGDR warm restarts (--cycle_ep)
    if getattr(args, "sched", "cosine") == "cosine_restarts":
        cyc_steps = max(1, int(getattr(args, "cycle_ep", 10))) * steps_per_ep
        lam = [make_lr_lambda_restarts(steps, warm_frac,
                                       head_only_ep * steps_per_ep, cyc_steps,
                                       float(getattr(args, "cycle_decay", 1.0)),
                                       is_backbone=(g["kind"] == "bb"))
               for g in groups]
        log(f"scheduler: cosine_restarts cycle_ep {getattr(args, 'cycle_ep', 10)}"
            f" decay {getattr(args, 'cycle_decay', 1.0)}")
    elif getattr(args, "sched", "cosine") == "cosine_hold":
        horizon = int(getattr(args, 'cosine_epochs', 40)) * steps_per_ep
        warm_n = int(getattr(args, 'warm_steps', 0)) or max(int(warm_frac * horizon), 1)
        lam = [cosine_hold_lambda(horizon, warm_n,
                                 head_only_ep * steps_per_ep if g['kind'] == 'bb' else 0,
                                 float(getattr(args, 'cosine_floor', 0.015)))
               for g in groups]
        log(f'SCHED cosine_hold: horizon {horizon} steps, warmup {warm_n}, '
            f'floor {getattr(args, "cosine_floor", 0.015)}, cap {args.epochs} epochs')
    else:
        lam = [make_lr_lambda(steps, warm_frac, head_only_ep * steps_per_ep,
                              is_backbone=(g["kind"] == "bb")) for g in groups]
    # exp-057: ADAPTIVE PROGRESSIVE UNFREEZE (user design, LP-FT/ULMFiT):
    # heads-only until val stalls `adaptive_unfreeze` epochs; then unfreeze
    # the TOP half of the trainable blocks (+bb_rest); next stall unfreezes
    # the rest. Each fired group warms up 5% of its remaining steps then
    # cosines to the shared end. LLRD base lrs unchanged.
    adap_k = int(getattr(args, "adaptive_unfreeze", 0))
    adap_boxes, adap_stages = {}, []
    if adap_k > 0:
        assert getattr(args, "sched", "cosine") in ("cosine", "plateau",
                                                    "wsd"), \
            "adaptive_unfreeze needs the cosine, plateau or wsd sched"
        # OPEN-ENDED schedule (user spec): no fixed horizon - each fired
        # group warms up, cosines over adap_span_ep, then HOLDS at
        # adap_min_lr forever; --epochs is only a hard cap and the run ends
        # by val early stopping (--patience, guarded until all stages fire).
        adap_span = max(1, int(getattr(args, "adap_span_ep", 15))) \
            * steps_per_ep
        adap_floor = float(getattr(args, "adap_min_lr", 0.05))
        tids = [i for i, blk in enumerate(model.bb.encoder.layer)
                if any(p.requires_grad for p in blk.parameters())]
        assert tids, "adaptive_unfreeze: no trainable backbone blocks " \
            "(need unfreeze_last > 0 or lora_r > 0)"
        n_st = max(1, min(2, int(getattr(args, "adap_stages", 2))))
        if n_st == 1:                     # exp-064-df: decoder-first LoRA -
            adap_stages = [[f"block{i}" for i in tids] + ["bb_rest"]]
        else:
            half = len(tids) // 2
            adap_stages = [[f"block{i}" for i in tids[half:]] + ["bb_rest"],
                           [f"block{i}" for i in tids[:half]]]

        def _span_lam(step, t0, span=adap_span, floor=adap_floor):
            if t0 is None or step < t0:
                return 0.0
            t = step - t0
            wu = max(int(0.05 * span), 1)
            if t < wu:
                return t / wu
            if t >= span:
                return floor
            c = 0.5 * (1.0 + math.cos(math.pi * (t - wu) / max(span - wu, 1)))
            return floor + (1.0 - floor) * c
        lam = []
        for g in groups:
            if g["kind"] != "bb":
                lam.append(lambda step: _span_lam(step, 0))
                continue
            box = {"t0": None}
            adap_boxes[g["name"]] = box
            lam.append(lambda step, b=box: _span_lam(step, b["t0"]))
        log(f"ADAPTIVE UNFREEZE ON (open-ended): patience {adap_k} | span "
            f"{getattr(args, 'adap_span_ep', 15)} ep -> floor {adap_floor} | "
            + " | ".join(f"stage{j + 1} {s}"
                         for j, s in enumerate(adap_stages))
            + f" | epoch cap {args.epochs}")
    adap_stage_i, adap_best, adap_last = 0, 0.0, 0
    # exp-064-df / I-098: ReduceLROnPlateau — horizon-agnostic LR (warmup ->
    # hold, then halve on val plateau). Composes with decoder-first: LoRA
    # stays frozen (LR 0) until its trigger fires, then rides the plateau
    # scale; the plateau controller only starts after ALL unfreeze stages
    # fire, so the unfreeze trigger governs the decoder-only phase.
    plateau_box = {"scale": 1.0}
    plat_best, plat_wait = 0.0, 0
    plat_factor = float(getattr(args, "plateau_factor", 0.5))
    plat_patience = int(getattr(args, "plateau_patience", 3))
    plat_thresh = float(getattr(args, "plateau_delta", 0.002))
    plat_min = float(getattr(args, "plateau_min", 0.02))
    # exp-066 v3 (user design): per-round LADDER "pat:delta,pat:delta,...":
    # round k uses ladder[k] (last entry repeats); LR x plateau_factor on
    # every plateau; a plateau AT the floor ends the run (the endpoint).
    plat_ladder = [(plat_patience, plat_thresh)]
    if str(getattr(args, "plateau_ladder", "") or "").strip():
        plat_ladder = [(int(p), float(d)) for p, d in
                       (x.split(":") for x in
                        str(args.plateau_ladder).split(",") if x.strip())]
    plat_metric = str(getattr(args, "plateau_metric", "val"))
    plat_round, plat_best_gold, plat_best_mean = 0, 0.0, 0.0
    plat_done = False
    warm_steps = None
    replay_on = getattr(args, 'sched', 'cosine') == 'plateau_replay'
    replay_drops = parse_drop_epochs(str(getattr(args, 'replay_drop_epochs', '')))
    if replay_on:
        assert adap_k == 0 and not getattr(args, 'branch_cooldown', False), \
            'plateau_replay cannot use adaptive unfreezing or cooldown branches'
        assert replay_drops, 'plateau_replay needs replay_drop_epochs'
        assert 0 < args.replay_factor <= 1 and 0 < args.replay_floor <= 1
    if getattr(args, "sched", "cosine") in ("plateau", "wsd", "plateau_replay"):
        warm = max(int(warm_frac * steps), 1)
        if replay_on:
            warm = max(int(warm_frac * int(args.replay_warmup_epochs) * steps_per_ep), 1)
        if int(getattr(args, "warm_steps", 0)) > 0:   # exp-066: fixed steps
            warm = int(args.warm_steps)                # (not tied to --epochs)
        if st is not None and st.get("warm"):
            warm = int(st["warm"])        # exact even if --epochs changed
        warm_steps = warm

        def _plat(step, t0, box=plateau_box, warm=warm):
            if t0 is None or step < t0:
                return 0.0
            return min(1.0, (step - t0 + 1) / warm) * box["scale"]
        lam = []
        for g in groups:
            if adap_k > 0 and g["kind"] == "bb" and g["name"] in adap_boxes:
                lam.append(lambda step, b=adap_boxes[g["name"]]:
                           _plat(step, b["t0"]))      # frozen until fired
            elif g["kind"] == "bb" and head_only_ep > 0:
                lam.append(lambda step, hs=head_only_ep * steps_per_ep:
                           _plat(step, hs))
            else:
                lam.append(lambda step: _plat(step, 0))
        if getattr(args, "sched", "cosine") == "plateau":
            log(f"SCHED plateau (RLROP ladder): factor {plat_factor} | floor "
                f"{plat_min} | metric {plat_metric} | rounds "
                + " ".join(f"[{p}ep,{d}]" for p, d in plat_ladder)
                + " (last repeats; a plateau at the floor ends the run)"
                + (" | starts after all unfreeze stages"
                   if adap_k > 0 else ""))
        elif replay_on:
            log(f'SCHED plateau_replay: drops visible in history epochs {replay_drops}; '
                f'factor {args.replay_factor}, floor {args.replay_floor}, warmup {warm} steps')
        else:
            log(f"SCHED wsd: warmup {warm} steps -> CONSTANT (per group, "
                f"from its own start) -> cooldown to 0 when triggered")
    # exp-066: WSD (warmup -> constant -> cooldown-to-zero) + BRANCH
    # cooldowns. The wrapper leaves the trunk schedule untouched until
    # cool_box["t0"] is set, then decays every group from its frozen scale
    # s0[i] to exactly 0 over cool_box["len"] steps with the chosen shape.
    cool_box = {"t0": None, "len": 0, "s0": None,
                "shape": str(getattr(args, "cooldown_shape", "linear"))}
    cool_best, cool_wait = 0.0, 0
    branch_mode = bool(getattr(args, "branch_cooldown", False))
    wsd_on = (getattr(args, "sched", "cosine") == "wsd" or branch_mode
              or (st is not None
                  and st["boxes"]["cool"]["t0"] is not None))
    if wsd_on:
        def _cool_shape(u, shape):
            if shape == "linear":
                return 1.0 - u
            if shape == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * u))
            if shape == "sqrt":                   # "1-sqrt" (Hagele 2024)
                return 1.0 - math.sqrt(u)
            raise ValueError(f"cooldown_shape {shape}")

        def _wrap_cool(base, gi, box=cool_box):
            def f(step):
                if box["t0"] is None or step < box["t0"]:
                    return base(step)
                u = min(1.0, (step - box["t0"]) / max(int(box["len"]), 1))
                return box["s0"][gi] * _cool_shape(u, box["shape"])
            return f
        lam = [_wrap_cool(f, i) for i, f in enumerate(lam)]
        log(f"COOLDOWN wrapper ON: shape {cool_box['shape']} | cooldown_ep "
            f"{getattr(args, 'cooldown_ep', 0)} | trigger at_ep "
            f"{getattr(args, 'cooldown_at_ep', -1)} / val-patience "
            f"{getattr(args, 'cooldown_patience', 0)}"
            + (" | BRANCH mode" if branch_mode else ""))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)
    y_soft = torch.from_numpy(soft).float()
    ta, ta_gen = None, None
    if getattr(args, "faa_policy", ""):          # exp-034c (searched policy)
        assert not getattr(args, "trivial_aug", False), \
            "--faa_policy and --trivial_aug are mutually exclusive"
        from augs import FAAPolicyMRI
        ta = FAAPolicyMRI.from_json(args.faa_policy,
                                    p_bag=float(getattr(args, "ta_p_bag",
                                                        0.25)))
        ta_gen = torch.Generator().manual_seed(seed * 31 + fold)
        log(f"FAA policy ON: {len(ta.subs)} sub-policies from "
            f"{args.faa_policy}, p_bag {ta.p_bag}")
    elif getattr(args, "trivial_aug", False):
        from augs import TrivialAugmentMRI       # exp-034a (augs package)
        ta = TrivialAugmentMRI(p_bag=float(getattr(args, "ta_p_bag", 0.25)))
        ta_gen = torch.Generator().manual_seed(seed * 31 + fold)
        log(f"TrivialAugment-MRI ON: {len(ta.pixel)} pixel + {len(ta.bag)} "
            f"bag ops, p_bag {ta.p_bag}")
    pos_w = None                                          # exp-073 (I-107)
    if bool(getattr(args, "pos_weight", False)):
        prev = np.nanmean(np.clip(soft[tr_idx], 0.0, 1.0), axis=0)
        prev = np.clip(prev, 1e-3, 1 - 1e-3)
        pw = np.clip((1.0 - prev) / prev, 1.0, 10.0)
        pos_w = torch.tensor(pw, dtype=torch.float32, device=dev)
        log("POS_WEIGHT ON (clip((1-prev)/prev, 1, 10) per target): "
            + " ".join(f"{t.split()[0][:6]} {v:.2f}"
                       for t, v in zip(TARGETS, pw)))
    xcp = float(getattr(args, "xcontrast_p", 0.0))        # exp-067
    if xcp > 0:
        t_x = time.time()
        cov = bank.build_xpair()
        bank.xcontrast_p = xcp
        log(f"CROSS-CONTRAST TOKENS ON: p {xcp} | ch0 = other-contrast slice "
            f"at the same level in the same plane | partner coverage "
            f"{cov:.3f} of slices (rest fall back to the 3-neighbour "
            f"triplet) | built in {time.time() - t_x:.0f}s")
    tr_studies = [studies[i] for i in tr_idx]
    whole = bool(getattr(args, "whole_study", False))     # exp-045
    if whole:
        assert args.batch == 1, "whole-study mode: ragged lengths need batch 1"
        ds = WholeStudyDataset(bank, tr_studies, seed + fold, train=True,
                               slice_drop=getattr(args, "slice_drop", 0.0),
                               max_slices=getattr(args, "max_slices", 256))
        log(f"WHOLE-STUDY mode: slice_drop {ds.slice_drop} | max_slices "
            f"{ds.max_slices} | eval = one full-study forward")
    else:
        ds = BagDataset(bank, tr_studies, args.bag, seed + fold, train=True)
        ds.bag_mode = str(getattr(args, "bag_mode", "random"))     # exp-072
        ds.bag_jitter = float(getattr(args, "bag_jitter", 0.5))
        if ds.bag_mode != "random":
            log(f"BAG SAMPLING: {ds.bag_mode} (contiguous per-series windows, "
                f"jitter {ds.bag_jitter}) instead of a uniform random draw")
    dl = make_train_loader(ds, args.batch, args.workers)   # P3.5
    rng = np.random.default_rng(seed + fold)
    best = {"val_auc": 0.0, "gold_auc": 0.0}
    history = []
    topk_k = int(getattr(args, "topk", 0))             # exp-073
    topk_pool = []
    bank.tri_gap = int(getattr(args, "tri_gap", 1))    # exp-079
    assert bank.tri_gap >= 1, "--tri_gap must be >= 1"
    if bank.tri_gap > 1:
        assert float(getattr(args, "xcontrast_p", 0.0)) == 0.0,             "--tri_gap > 1 and --xcontrast_p > 0 are not composable: the "            "cross-contrast token is built at +-1 and would be a second "            "uncontrolled variable"
        log(f"TOKEN EXTENT (exp-079): tri_gap {bank.tri_gap} -> each 2.5D "
            f"token spans {2 * bank.tri_gap} slice-gaps (was 2), same compute")
    # ---- exp-078: pairwise ranking surrogate against a FIFO of past logits
    rank_lam = float(getattr(args, "rank_lambda", 0.0))
    rq_s = rq_y = rq_m = None
    rq_ptr, rq_fill = 0, 0
    if rank_lam > 0:
        rq_n = int(getattr(args, "rank_queue", 1024))
        rq_s = torch.zeros(rq_n, N_TARGET, device=dev)
        rq_y = torch.zeros(rq_n, N_TARGET, device=dev)
        rq_m = torch.zeros(rq_n, N_TARGET, dtype=torch.bool, device=dev)
        log(f"RANK LOSS (exp-078): lambda {rank_lam} | queue {rq_n} | margin "
            f"{float(getattr(args, 'rank_margin', 1.0))} | conf "
            f"{float(getattr(args, 'rank_conf', 0.25))} (soft labels inside "
            f"the conf band define no pair)")
    global_step = 0
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    start_ep, end_ep = 0, int(args.epochs)
    resumed_from, branch_from = None, None
    bank_id = {"n_slices": int(getattr(bank, "n_slices_total", -1)),
               "img": int(bank.img),
               "n_studies": len(getattr(bank, "by_study", [])),
               "n_train": int(len(tr_idx))}

    def _start_cooldown(ep_now):
        """Freeze every group's current scale and decay it to 0 over
        --cooldown_ep epochs; the run ends when the cooldown ends."""
        nonlocal end_ep
        cool_ep = int(getattr(args, "cooldown_ep", 0))
        if cool_ep <= 0:          # 20% of the trunk, floor 2 ep (Hagele/MiniCPM)
            cool_ep = max(int(getattr(args, "cooldown_min_ep", 2)),
                          int(round(float(getattr(args, "cooldown_frac", 0.2))
                                    * (ep_now + 1))))
        cool_box["s0"] = [float(sched.lr_lambdas[i](global_step))
                          for i in range(len(opt.param_groups))]
        cool_box["t0"], cool_box["len"] = int(global_step), cool_ep * steps_per_ep
        end_ep = ep_now + 1 + cool_ep
        log(f"COOLDOWN start after ep {ep_now} @ step {global_step}: "
            f"{cool_ep} ep {cool_box['shape']} -> 0 | s0 max "
            f"{max(cool_box['s0']):.3f} | end_ep {end_ep}")

    def _save_state(ep_now, stopped):
        state = {
            "format": 1, "arch": "slicebag-v1", "fold": fold, "ep": ep_now,
            "global_step": int(global_step), "steps_per_ep": steps_per_ep,
            "end_ep": int(end_ep), "warm": warm_steps,
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "ema": {k: v.cpu() for k, v in ema.state_dict().items()},
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "optimizer_lrs": [float(g['lr']) for g in opt.param_groups],
            "scheduler_next_lrs": [float(g['initial_lr'] * sched.lr_lambdas[i](global_step))
                                   for i, g in enumerate(opt.param_groups)],
            "scaler": scaler.state_dict(),
            "aux": aux.state_dict() if aux is not None else None,
            "aux_txt": aux_txt.state_dict() if aux_txt is not None else None,
            "aux_sev": aux_sev.state_dict() if aux_sev is not None else None,
            "rng": {"torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                    "np": rng.bit_generator.state,
                    "ta_gen": (ta_gen.get_state() if ta_gen is not None
                               else None),
                    "py": random.getstate()},
            "boxes": {"adap_t0": {n: b["t0"] for n, b in adap_boxes.items()},
                      "adap_stage_i": adap_stage_i, "adap_best": adap_best,
                      "adap_last": adap_last,
                      "plateau_scale": plateau_box["scale"],
                      "plat_best": plat_best, "plat_wait": plat_wait,
                      "plat_round": plat_round,
                      "plat_best_gold": plat_best_gold,
                      "plat_best_mean": plat_best_mean,
                      "plat_done": plat_done,
                      "cool": dict(cool_box), "cool_best": cool_best,
                      "cool_wait": cool_wait},
            "best": best, "history": history,
            "chop_qtau": (chop_qtau.copy() if chop_on else None),
            "protect": (getattr(ds, "protect", None) if prot_on else None),
            "bank": bank_id, "args": dict(vars(args)),
            "torch": torch.__version__, "early_stopped": bool(stopped)}
        _atomic_save(state, Path(args.out_dir) / "state_last.pt")
        k_keep = int(getattr(args, "keep_state_every", 0))
        if k_keep > 0 and (ep_now + 1) % k_keep == 0:
            _atomic_save(state, Path(args.out_dir) / f"state_ep{ep_now}.pt")

    if st is not None:                           # ---- exact resume
        bad = [k for k in RESUME_KEYS if k in st["args"]
               and st["args"][k] != getattr(args, k, st["args"][k])]
        assert not bad, f"resume: args differ from the saved run: {bad}"
        soft_diff = [k for k in ("plateau_ladder", "plateau_metric",
                                 "plateau_factor", "plateau_min", "epochs",
                                 "warm_steps", "cooldown_frac")
                     if k in st["args"] and getattr(args, k, None) is not None
                     and st["args"][k] != getattr(args, k)]
        if soft_diff:
            log(f"resume NOTE: schedule flags changed vs the saved run "
                f"(CLI wins): {soft_diff}")
        if getattr(args, "sched", "cosine") in ("cosine", "cosine_restarts"):
            assert int(st["args"].get("epochs", args.epochs)) == int(args.epochs), \
                "resume under a fixed-horizon sched needs the same --epochs"
        assert int(st["steps_per_ep"]) == steps_per_ep, \
            (st["steps_per_ep"], steps_per_ep)
        if st.get("bank") not in (None, bank_id):
            assert not getattr(args, 'data_fingerprint', ''), \
                'resume: bank identity changed in a fingerprinted run'
            log(f"WARNING resume: bank identity differs {st.get('bank')} vs "
                f"{bank_id}")
        model.load_state_dict(st["model"], strict=True)
        ema.load_state_dict(st["ema"], strict=True)
        for mod, key in ((aux, "aux"), (aux_txt, "aux_txt"),
                         (aux_sev, "aux_sev")):
            if mod is not None:
                mod.load_state_dict(st[key])
        assert len(st["opt"]["param_groups"]) == len(opt.param_groups), \
            "resume: optimizer group count differs (aux flags?)"
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        bx = st["boxes"]
        assert set(bx["adap_t0"]) == set(adap_boxes), \
            (sorted(bx["adap_t0"]), sorted(adap_boxes))
        for nm_, t0_ in bx["adap_t0"].items():
            adap_boxes[nm_]["t0"] = t0_
        adap_stage_i, adap_best, adap_last = (bx["adap_stage_i"],
                                              bx["adap_best"], bx["adap_last"])
        plateau_box["scale"] = bx["plateau_scale"]
        plat_best, plat_wait = bx["plat_best"], bx["plat_wait"]
        plat_round = int(bx.get("plat_round", 0))
        plat_best_gold = float(bx.get("plat_best_gold", 0.0))
        plat_best_mean = float(bx.get("plat_best_mean", 0.0))
        plat_done = bool(bx.get("plat_done", False))
        cool_box.update(bx["cool"])
        cool_best, cool_wait = bx["cool_best"], bx["cool_wait"]
        global_step = int(st["global_step"])
        resumed_from = int(st["ep"])
        start_ep = resumed_from + 1
        end_ep = (int(st["end_ep"]) if cool_box["t0"] is not None
                  else int(args.epochs))
        best, history = dict(st["best"]), list(st["history"])
        if chop_on and st.get("chop_qtau") is not None:
            chop_qtau = np.asarray(st["chop_qtau"]).copy()
            chop_qtau_t = torch.from_numpy(chop_qtau).float().to(dev)
        if prot_on and st.get("protect") is not None:
            ds.protect = st["protect"]
            ds.protect_pos = prot_pos
            ds.p_prot = float(getattr(args, "ep_prot_p", 0.9))
        r_ = st["rng"]
        torch.set_rng_state(r_["torch_cpu"])
        torch.cuda.set_rng_state_all(r_["torch_cuda"])
        rng.bit_generator.state = r_["np"]
        if ta_gen is not None and r_.get("ta_gen") is not None:
            ta_gen.set_state(r_["ta_gen"])
        if r_.get("py") is not None:
            random.setstate(r_["py"])
        assert sched.last_epoch == global_step, (sched.last_epoch, global_step)
        for gi_, g_ in enumerate(opt.param_groups):
            want = g_["initial_lr"] * sched.lr_lambdas[gi_](global_step)
            tol = 1e-9 * max(1.0, g_["initial_lr"]) + 1e-15
            if 'optimizer_lrs' in st:
                # An epoch-end controller may have changed its scale AFTER
                # the last scheduler step. Preserve both sides of that
                # boundary, including the first optimizer step on resume.
                assert abs(g_['lr'] - st['optimizer_lrs'][gi_]) <= tol
                assert abs(want - st['scheduler_next_lrs'][gi_]) <= tol, \
                    (gi_, g_.get('name'), want, st['scheduler_next_lrs'][gi_])
            else:
                assert abs(g_["lr"] - want) <= tol, \
                    (gi_, g_.get("name"), g_["lr"], want)
        if st.get("torch") != torch.__version__:
            log(f"WARNING resume: torch {st.get('torch')} -> "
                f"{torch.__version__}")
        if branch_mode:
            if cool_box["t0"] is None:           # new branch off the trunk
                assert Path(args.out_dir).resolve() != \
                    Path(resume_path).resolve().parent, \
                    "branch_cooldown: use a fresh --out_dir (trunk dir is " \
                    "read-only)"
                _start_cooldown(resumed_from)
                best = {"val_auc": 0.0, "gold_auc": 0.0}
                cool_best, cool_wait = 0.0, 0
                branch_from = resumed_from
                json.dump({"from_state": resume_path, "from_ep": resumed_from,
                           "t0": cool_box["t0"], "s0": cool_box["s0"],
                           "shape": cool_box["shape"], "len": cool_box["len"],
                           "end_ep": end_ep},
                          open(Path(args.out_dir) / "branch.json", "w"))
            else:
                log("branch: state already cooling -> continuing it")
        elif st.get("early_stopped"):
            log("resume: saved run had early-stopped; nothing to continue")
            start_ep = end_ep
        log(f"RESUME {resume_path}: ep {start_ep}/{end_ep} step {global_step}"
            f" | best val {best.get('val_auc', 0):.4f} @ ep "
            f"{best.get('epoch', -1)} | lr-check OK"
            + (f" | cooldown t0 {cool_box['t0']}" if cool_box["t0"] is not None
               else ""))
        del st
    for ep in range(start_ep, 10 ** 6):
        if ep >= end_ep:
            break
        ds.epoch = ep            # reaches workers because they re-spawn
        if whole and float(getattr(args, "slice_drop_end", -1)) >= 0:
            e0 = int(getattr(args, "drop_ramp_ep0", 0))
            e1 = int(getattr(args, "drop_ramp_ep1", -1))
            e1 = args.epochs - 1 if e1 < 0 else e1
            d0 = float(getattr(args, "slice_drop", 0.0))
            d1 = float(args.slice_drop_end)
            t = (0.0 if ep <= e0 else 1.0 if ep >= e1
                 else (ep - e0) / max(1, e1 - e0))
            ds.slice_drop = round(d0 + (d1 - d0) * t, 4)
        model.train()
        losses, t_ep = [], time.time()
        aux_stats_ep = []
        rank_ep = []            # exp-078: per-step ranking-loss values
        if chop_on:
            chop_cnt = np.zeros(12)
        if prot_on and ep % prot_refresh == 0:
            t_pr = time.time()
            ds.protect, prot_ent = compute_protect(
                ema, bank, tr_studies, prot_pos, dev,
                topk=int(getattr(args, "ep_prot_k", 3)),
                workers=args.workers)
            ds.protect_pos = prot_pos
            ds.p_prot = float(getattr(args, "ep_prot_p", 0.9))
            log(f"protect refresh ep {ep}: {len(ds.protect)} studies | "
                f"pos-att entropy {prot_ent:.3f} | "
                f"{(time.time() - t_pr) / 60:.1f} min")
        opt.zero_grad(set_to_none=True)
        t_data, _t0 = 0.0, time.time()          # data-stall meter (exp-045)
        _it = iter(dl)
        bi = -1
        while True:
            _tw = time.time()
            try:
                imgs, meta, mask, sidx = next(_it)
            except StopIteration:
                break
            t_data += time.time() - _tw
            bi += 1
            imgs, meta, mask = (imgs.to(dev, non_blocking=True), meta.to(dev),
                                mask.to(dev))
            meta_orig = meta.clone() if aux is not None else None  # pre-TA
            if ta is not None:                   # ONE op per sample (TA)
                x01 = imgs.float().div_(255.0)
                x01, meta, mask = ta(x01, meta, mask, ta_gen)
                imgs = x01.mul_(255.0)
            rows_sel = tr_idx[sidx.numpy()]
            tgt = y_soft[rows_sel].to(dev)
            w = batch_weights(tgt, label_mask[rows_sel].to(dev),
                              args.conf_weight)
            if w_cell is not None:            # before flip: SWAP_IDX applies
                w = w * w_cell[rows_sel].to(dev)
            u = rng.random()                  # always drawn (stream parity)
            if flip_p > 0 and u < flip_p:
                imgs, meta = flip_bag(imgs, meta)
                tgt = tgt[:, SWAP_IDX]
                w = w[:, SWAP_IDX].clone()
                w[:, TARGETS.index("MCL")] = 0.0
            if chop_on:
                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=AMP_DTYPE):
                    p_ema = torch.sigmoid(ema(imgs, meta, mask).float())
                pos = tgt > 0.5
                for b_i, c_i in pos.nonzero().tolist():
                    chop_pacc[c_i].append(float(p_ema[b_i, c_i]))
                cand = (pos & (p_ema < chop_hi) & (p_ema <= chop_qtau_t)
                        & (w > 0))
                if cand.any():
                    ramp = min(1.0, (ep + 1) / chop_ramp)
                    cap = chop_rho * ramp * chop_pos_total
                    for b_i, c_i in cand.nonzero().tolist():
                        if chop_cnt[c_i] < cap[c_i]:
                            w[b_i, c_i] = 0.0
                            chop_cnt[c_i] += 1
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                if aux is not None:
                    z, hpix = model(imgs, meta, mask, return_hpix=True)
                elif aux_txt is not None:
                    z, semb = model(imgs, meta, mask, return_study=True)
                elif aux_sev is not None:
                    z, ctx_f = model(imgs, meta, mask, return_ctx=True)
                else:
                    z = model(imgs, meta, mask)
                loss = (_soft_gce_bce(
                    z.float(), tgt, reduction="none",
                    pos_weight=pos_w) * w).sum() / w.sum()
                if rank_lam > 0:              # exp-078: AUC surrogate
                    conf = float(getattr(args, "rank_conf", 0.25))
                    zf = z.float()
                    b_pos = (tgt > 0.5 + conf) & (w > 0)
                    b_neg = (tgt < 0.5 - conf) & (w > 0)
                    b_m = b_pos | b_neg
                    if rq_fill > 0 and bool(b_m.any()):
                        qs = rq_s[:rq_fill]                    # [Q,T]
                        qy = rq_y[:rq_fill]
                        qm = rq_m[:rq_fill]
                        # sign +1 where (batch pos, queue neg), -1 where
                        # (batch neg, queue pos), 0 otherwise -> no pair
                        sg = ((b_pos.float()[:, None, :]
                               * (qm & (qy < 0.5)).float()[None, :, :])
                              - (b_neg.float()[:, None, :]
                                 * (qm & (qy > 0.5)).float()[None, :, :]))
                        npair = sg.abs().sum()
                        if float(npair) > 0:
                            d = zf[:, None, :] - qs[None, :, :]   # [B,Q,T]
                            mg = float(getattr(args, "rank_margin", 1.0))
                            h = torch.clamp(mg - sg * d, min=0.0) ** 2
                            lr_ = (h * sg.abs()).sum() / npair
                            loss = loss + rank_lam * lr_
                            rank_ep.append(float(lr_.detach()))
                    with torch.no_grad():     # push this batch into the FIFO
                        k = zf.shape[0]
                        idx_ = (torch.arange(k, device=dev) + rq_ptr) % rq_s.shape[0]
                        rq_s[idx_] = zf.detach()
                        rq_y[idx_] = (tgt > 0.5).float()
                        rq_m[idx_] = b_m
                        rq_ptr = int((rq_ptr + k) % rq_s.shape[0])
                        rq_fill = min(rq_s.shape[0], rq_fill + k)
                if aux is not None:
                    la, a_st = aux.loss(hpix, meta_orig, mask)
                    loss = loss + float(getattr(args, "aux_lambda", 0.1)) * la
                    aux_stats_ep.append(a_st)
                elif aux_txt is not None:
                    la, a_st = aux_txt.loss(
                        semb, txt_emb, txt_rows[rows_sel].to(dev))
                    loss = (loss + float(getattr(args, "aux_txt_lambda", 0.1))
                            * la)
                    aux_stats_ep.append(a_st)
                elif aux_sev is not None:
                    la, a_st = aux_sev.loss(ctx_f, sev_t[rows_sel].to(dev))
                    loss = (loss + float(getattr(args, "aux_sev_lambda", 0.1))
                            * la)
                    aux_stats_ep.append(a_st)
            losses.append(loss.item())
            scaler.scale(loss / accum).backward()
            if (bi + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(bb_params + head_params, 3.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema_update(ema, model, args.ema, global_step)
                global_step += 1
        # exp-077: a holdout jury REPLACES va_auc, so predicting the fold-0
        # val split would cost ~74 pct of the eval pass for a number that is
        # then discarded (and meaningless anyway under --full_data, where
        # those studies are inside train).
        va_st = ([] if holdout is not None
                 else [studies[i] for i in va_idx])
        gd_st = [studies[i] for i in gold_idx]
        hd_st = ([studies[i] for i in holdout[0]]
                 if holdout is not None else [])
        if whole:
            if va_st:
                pv = predict_full(ema, bank, va_st, dev,
                                  getattr(args, "max_slices", 256))
            if hd_st:
                pvh = predict_full(ema, bank, hd_st, dev,
                                   getattr(args, "max_slices", 256))
            pg = np.empty((0, N_TARGET), dtype=np.float64)
        else:
            eb = int(getattr(args, "eval_batch", 8))   # exp-059: hybrids OOM
            if va_st:                                  # at 8 bags x 32 slices
                pv = predict(ema, bank, va_st, dev, args.bag, args.eval_bags,
                             seed=12345, batch=eb, workers=args.workers)
            if hd_st:
                pvh = predict(ema, bank, hd_st, dev, args.bag,
                              args.eval_bags, seed=12345, batch=eb,
                              workers=args.workers)
            pg = np.empty((0, N_TARGET), dtype=np.float64)
        gold_auc, gold_per = float('nan'), [float('nan')] * N_TARGET
        if holdout is not None:                # exp-063: holdout selects
            va_auc, va_per = macro_auc(holdout[1], pvh, holdout[2])
            pv = pvh
        elif va_st:
            va_auc, va_per = macro_auc(y_eval_bin[va_idx], pv,
                                       y_eval_valid[va_idx])
        else:
            # FULL-DATA mode (empty val, e.g. --fold 99): the gold holdout
            # both scores AND selects the checkpoint (Raptor-style).
            pv = np.zeros((0, len(TARGETS)))
            va_auc, va_per = gold_auc, [np.nan] * len(TARGETS)
        gold2_msg = ""
        gold2_auc = None
        if gold2 is not None:
            g2_src, g2_y, g2_valid = gold2
            pg2 = np.stack([pg[k] if t == "g" else pv[k] for t, k in g2_src])
            gold2_auc, _ = macro_auc(g2_y, pg2, g2_valid)
            gold2_msg = f" | G2 {gold2_auc:.3f}"
        history.append({"ep": ep, "val": round(va_auc, 4),
                        "gold": round(gold_auc, 4),
                        "loss": round(float(np.mean(losses)), 4),
                        "val_per_target": [None if np.isnan(a)
                                           else round(float(a), 4)
                                           for a in va_per]})
        for kind in ("bb", "head"):                # exp-066 LR audit (I-098)
            history[-1][f"lr_{kind}"] = max(
                (float(g["lr"]) for g in opt.param_groups
                 if g.get("kind") == kind), default=0.0)
        history[-1]["phase"] = ("cooldown" if cool_box["t0"] is not None
                                else "trunk")
        history[-1]["lr_round"] = plat_round        # RLROP ladder round
        history[-1]['global_step'] = global_step
        if resumed_from is not None and ep == start_ep:
            history[-1]["resumed_from_ep"] = resumed_from
            if branch_from is not None:
                history[-1]["branch_from_ep"] = branch_from
        if whole and float(getattr(args, "slice_drop_end", -1)) >= 0:
            history[-1]["drop"] = ds.slice_drop
        if gold2_auc is not None:
            history[-1]["gold2"] = round(gold2_auc, 4)
        if adap_k > 0 and adap_stage_i < len(adap_stages) \
                and cool_box["t0"] is None:
            if va_auc > adap_best + float(getattr(args, "adap_min_delta",
                                                  1e-6)):
                adap_best, adap_last = va_auc, ep
            elif ep - adap_last >= adap_k:
                for nm in adap_stages[adap_stage_i]:
                    if nm not in adap_boxes:     # e.g. no bb_rest under LoRA
                        continue
                    if nm in adap_boxes:
                        adap_boxes[nm]["t0"] = global_step
                log(f"UNFREEZE stage {adap_stage_i + 1}: "
                    f"{adap_stages[adap_stage_i]} @ step {global_step}")
                adap_stage_i += 1
                adap_best, adap_last = va_auc, ep
        if wsd_on and cool_box["t0"] is None:       # exp-066 cooldown trigger
            cool_at = int(getattr(args, "cooldown_at_ep", -1))
            cool_pat = int(getattr(args, "cooldown_patience", 0))
            fire = cool_at >= 0 and ep + 1 >= cool_at
            if (not fire and cool_pat > 0
                    and (adap_k == 0 or adap_stage_i >= len(adap_stages))):
                if va_auc > cool_best + float(getattr(args,
                                                      "cooldown_min_delta",
                                                      0.002)):
                    cool_best, cool_wait = va_auc, 0
                else:
                    cool_wait += 1
                    fire = cool_wait >= cool_pat
                    if fire:
                        log(f"COOLDOWN trigger: EMA-val plateau {cool_pat} "
                            f"ep (best {cool_best:.4f}) @ ep {ep}")
            if fire:
                _start_cooldown(ep)
        if getattr(args, "sched", "cosine") == "plateau" \
                and cool_box["t0"] is None \
                and (adap_k == 0 or adap_stage_i >= len(adap_stages)):
            pat_k, del_k = plat_ladder[min(plat_round, len(plat_ladder) - 1)]
            imp_v = va_auc > plat_best + del_k
            imp_g = gold_auc > plat_best_gold + del_k
            m_ = 0.5 * (va_auc + gold_auc)
            imp_m = m_ > plat_best_mean + del_k
            imp = {"val": imp_v, "gold": imp_g, "either": imp_v or imp_g,
                   "mean": imp_m}[plat_metric]
            if imp_v:                 # bars move only on qualifying gains
                plat_best = va_auc    # (PyTorch RLROP 'abs' semantics)
            if imp_g:
                plat_best_gold = gold_auc
            if imp_m:
                plat_best_mean = m_
            if imp:
                plat_wait = 0
            else:
                plat_wait += 1
                if plat_wait >= pat_k:
                    if plateau_box["scale"] <= plat_min + 1e-12:
                        plat_done = True
                        log(f"RLROP END: {plat_metric} plateau {pat_k} ep at "
                            f"the LR floor {plat_min} (round {plat_round}) "
                            f"@ ep {ep}")
                    else:
                        new = max(plateau_box["scale"] * plat_factor, plat_min)
                        plateau_box["scale"] = new
                        plat_round += 1
                        log(f"RLROP round {plat_round}: {plat_metric} plateau "
                            f"{pat_k} ep (delta {del_k}) -> lr scale "
                            f"{new:.4f} @ ep {ep}")
                    plat_wait = 0
        if replay_on and ep + 1 in replay_drops:
            # Match the original controller's timing: change the box AFTER
            # evaluation, leaving the optimizer's current LR untouched until
            # its next scheduler.step(). This includes the one-step boundary lag.
            plateau_box['scale'] = max(plateau_box['scale'] * args.replay_factor,
                                       args.replay_floor)
            plat_round += 1
            log(f'REPLAY round {plat_round}: after ep {ep}, scale {plateau_box["scale"]:.6f}')
        chop_msg = ""
        if chop_on:
            cf = chop_cnt / chop_pos_total
            history[-1]["chop"] = [round(float(x), 4) for x in cf]
            chop_msg = (f" | chop {cf[list((2, 3, 10, 11))].mean():.3f}foc/"
                        f"{cf.mean():.3f}all")
            ramp_nx = min(1.0, (ep + 2) / chop_ramp)
            for c in range(12):
                a = np.asarray(chop_pacc[c], np.float64)
                chop_qtau[c] = (float(np.quantile(
                    a, min(0.5, chop_rho[c] * ramp_nx)))
                    if len(a) >= chop_min_n else 0.0)
            chop_pacc = [[] for _ in range(12)]
            chop_qtau_t = torch.from_numpy(chop_qtau).float().to(dev)
            history[-1]["chop_qtau"] = [round(float(x), 4) for x in chop_qtau]
        rank_msg = (f" | rank {sum(rank_ep) / len(rank_ep):.4f}"
                    f" ({len(rank_ep)} steps)") if rank_ep else ""
        aux_msg = ""
        if aux_stats_ep:
            keys = aux_stats_ep[0].keys()
            am = {k: float(np.mean([s[k] for s in aux_stats_ep])) for k in keys}
            aux_msg = " | aux " + " ".join(f"{k} {v:.3f}"
                                           for k, v in am.items())
        log(f"fold {fold} ep {ep}: loss {np.mean(losses):.4f} | val(EMA) "
            f"{va_auc:.3f} | GOLD {gold_auc:.3f}{gold2_msg} | "
            f"{(time.time() - t_ep) / 60:.1f} min"
            f" | data-wait {t_data:.0f}s" + chop_msg + rank_msg + aux_msg)
        _atomic_numpy(pv, Path(args.out_dir) / f"val_preds_f{fold}_ep{ep}.npy")
        _atomic_json(history, Path(args.out_dir) / f"history_f{fold}.json")
        sel = (gold_auc if str(getattr(args, "select_on", "val")) == "gold"
               else va_auc)                            # exp-073
        if topk_k > 0:                                 # exp-073 weight avg
            topk_pool.append((float(sel), ep,
                              {k: v.detach().cpu().clone()
                               for k, v in ema.state_dict().items()}))
            topk_pool.sort(key=lambda z: -z[0])
            del topk_pool[topk_k:]
        if sel > best["val_auc"]:
            best = {"val_auc": sel, "gold_auc": gold_auc, "epoch": ep,
                    "val_auc_raw": va_auc,
                    "gold_per_target": [round(float(a), 4) for a in gold_per],
                    "val_per_target": history[-1]["val_per_target"]}
            if gold2_auc is not None:
                best["gold2_auc"] = round(gold2_auc, 4)
            _atomic_save({"state_dict": {k: v.cpu() for k, v in
                                       ema.state_dict().items()},
                        "fold": fold, "targets": TARGETS, "arch": "slicebag-v1",
                        "backbone": str(getattr(args, "backbone", "")),
                        "full_finetune": model.full_finetune,
                        "d_model": args.d_model, "n_mix": args.n_mix,
                        "data_fingerprint": str(getattr(args, 'data_fingerprint', '')),
                        "bag": args.bag, "img": bank.img, "ema_eval": True,
                        "tri_gap": int(getattr(args, "tri_gap", 1)),
                        "decoder": getattr(args, "decoder", "mil"),
                        "pool": getattr(args, "pool", "cls_mean"),
                        "flip_p": flip_p, "llrd": llrd, "metrics": best,
                        "zoom_px": int(getattr(args, "zoom_px", 0)),
                        "ms_layers": list(ms_layers),
                        "aux_backbone": aux_bb,
                        "proj_adapter": int(model.proj_adapter),
                        "lora_r": int(model.lora_r),
                        "lora_alpha": float(getattr(args, "lora_alpha", 16)),
                        "lora_targets": str(getattr(model, "lora_targets",
                                                    "qv")),
                        "xcontrast_p": float(getattr(args, "xcontrast_p", 0)),
                        "quad_pool": int(model.quad_pool)},
                       Path(args.out_dir) / f"ckpt_fold{fold}.pt")
        patience = int(getattr(args, "patience", 0))
        stop_now = bool(
            patience > 0 and ep + 1 >= int(getattr(args, "min_ep", 24))
            and ep - best.get("epoch", ep) >= patience
            and (adap_k == 0 or adap_stage_i >= len(adap_stages))
            and cool_box["t0"] is None)          # cooldown is terminal
        if stop_now:
            log(f"EARLY STOP at ep {ep}: no val improvement since "
                f"ep {best.get('epoch', ep)} (patience {patience})")
        stop_now = stop_now or plat_done         # RLROP ladder exhausted
        if int(getattr(args, "save_state", 1)):  # exp-066 full-state save
            t_sv = time.time()
            _save_state(ep, stop_now)
            log(f"state saved ep {ep} ({(time.time() - t_sv):.0f}s)")
        if stop_now:
            break
    if topk_k > 0 and topk_pool:                        # exp-073: SWA ckpt
        eps = [e for _, e, _ in topk_pool]
        avg = {k: sum(sd[k].float() for _, _, sd in topk_pool) / len(topk_pool)
               for k in topk_pool[0][2]}
        base = torch.load(Path(args.out_dir) / f"ckpt_fold{fold}.pt",
                          map_location="cpu", weights_only=False)
        base["state_dict"] = avg
        base["topk_epochs"] = eps
        torch.save(base, Path(args.out_dir) / f"ckpt_fold{fold}_topk.pt")
        log(f"TOP-{topk_k} WEIGHT AVERAGE saved (epochs {sorted(eps)}) -> "
            f"ckpt_fold{fold}_topk.pt")
        best["topk_epochs"] = sorted(eps)
    best["history"] = history
    best["env"] = {"seed": seed, "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0), "flip_p": flip_p,
                   "llrd": llrd, "warmup_frac": warm_frac,
                   "head_only_ep": head_only_ep, "ema": args.ema,
                   "global_step": global_step, "end_ep": end_ep,
                   "resume_state": resume_path, "branch_from": branch_from}
    del model, ema
    gc.collect()
    torch.cuda.empty_cache()
    return best


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_finetune', action='store_true',
                    help='train ALL backbone params, including stem and downsample layers')
    ap.add_argument('--grad_checkpoint', action='store_true',
                    help='checkpoint backbone activations to reduce training VRAM')
    ap.add_argument('--data_fingerprint', default='', help='data/config digest checked on resume')
    ap.add_argument('--cosine_epochs', type=int, default=40)
    ap.add_argument('--cosine_floor', type=float, default=0.015)
    ap.add_argument('--replay_drop_epochs', default='10,19,25,29,35,37,39,41,43')
    ap.add_argument('--replay_factor', type=float, default=0.6)
    ap.add_argument('--replay_floor', type=float, default=0.015)
    ap.add_argument('--replay_warmup_epochs', type=int, default=80,
                    help='original epoch cap used to calculate replay warmup (not new cap)')
    ap.add_argument("--cache_dir", default="/content/allslice224")
    ap.add_argument("--csv_dir", default="/content")
    ap.add_argument("--labels_csv", default="labels_v07.csv")
    ap.add_argument("--eval_labels_csv", default="labels_v06.csv")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--backbone", default="facebook/dinov2-small")
    ap.add_argument("--unfreeze_last", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_mix", type=int, default=2)
    ap.add_argument("--bag", type=int, default=32)
    ap.add_argument("--eval_bags", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr_backbone", type=float, default=8e-6)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--conf_weight", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    # ---- P3 (diagnosis 2026-08-21 §4 recipe; this trainer has no pre-P3 run)
    ap.add_argument("--flip_p", type=float, default=0.0,
                    help="P(train-time L/R flip); 0 = off (legacy flip_bag "
                         "only mirrors relpos, see docstring)")
    ap.add_argument("--llrd", type=float, default=0.75,
                    help="layer-wise LR decay over backbone blocks")
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--head_only_ep", type=int, default=2,
                    help="epochs with backbone LR held at 0 first")
    ap.add_argument("--trivial_aug", action="store_true",
                    help="exp-034a: TrivialAugment-MRI (one random augs-"
                         "package op per sample, magnitude U(0,1))")
    ap.add_argument("--ta_p_bag", type=float, default=0.25,
                    help="probability the TA op is a bag op")
    ap.add_argument("--faa_policy", default="",
                    help="exp-034c: path to a faa_search_v2 policy JSON — "
                         "per sample one random searched sub-policy replaces "
                         "TA's uniform pixel op (bag branch kept at ta_p_bag)")
    # ---- exp-037 schedule arms
    ap.add_argument("--sched",
                    choices=["cosine", "cosine_restarts", "plateau", "wsd",
                             "cosine_hold", "plateau_replay"],
                    default="cosine")
    ap.add_argument("--cycle_ep", type=int, default=10,
                    help="cosine_restarts: cycle length in epochs (SGDR, "
                         "T_mult=1)")
    ap.add_argument("--cycle_decay", type=float, default=1.0,
                    help="cosine_restarts: LR peak multiplier per restart")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after N epochs without a val improvement "
                         "(0 = off). Val-based ONLY (gold stays untouched as "
                         "an instrument); floor at --min_ep")
    ap.add_argument("--min_ep", type=int, default=24,
                    help="never early-stop before this epoch (val saturation "
                         "guard, I-070)")
    # ---- exp-038 arm 1: AUX-GEO
    ap.add_argument("--aux_geo", action="store_true",
                    help="per-slice plane/fluid/fatsat/relpos auxiliary heads "
                         "on pixel-side features (pre-metadata-embedding); "
                         "heads are NOT saved in the checkpoint")
    ap.add_argument("--aux_lambda", type=float, default=0.1,
                    help="total weight of the AUX-GEO loss")
    # ---- exp-038 arm 3: AUX-TXT
    ap.add_argument("--aux_txt", default="",
                    help="path PREFIX of report embeddings (loads <p>.npy + "
                         "<p>_studies.json); enables ConVIRT-style joint "
                         "InfoNCE aux loss (heads NOT in the checkpoint)")
    ap.add_argument("--aux_txt_lambda", type=float, default=0.1)
    ap.add_argument("--aux_txt_tau", type=float, default=0.07)
    # ---- exp-038 arm 2: AUX-SEV
    ap.add_argument("--aux_sev", default="",
                    help="path to severity_v01.csv; enables CORN ordinal "
                         "auxiliary heads on per-target decoder features "
                         "(heads NOT in the checkpoint)")
    ap.add_argument("--aux_sev_lambda", type=float, default=0.1)
    # ---- exp-041: ConVIRT-proper (pretrain-then-finetune)
    ap.add_argument("--init_backbone", default="",
                    help="ckpt path: warm-start ONLY bb.* tensors from it "
                         "(heads/mixer/decoder start fresh)")
    # ---- exp-045c: continuation training (hard-tail curriculum)
    ap.add_argument("--init_full", default="",
                    help="ckpt path: warm-start the ENTIRE model (strict "
                         "load; EMA restarts from the same weights). For "
                         "continuation runs; optimizer/schedule start fresh")
    ap.add_argument("--init_lax", type=int, default=0,
                    help="with --init_full: skip shape-mismatched tensors "
                         "(fresh init) instead of strict load - for "
                         "architecture-widening continuations (exp-058b)")
    # ---- exp-047 arm A: POS-LLR (chop evidence-free positive draws)
    ap.add_argument("--chop_pos", action="store_true",
                    help="zero-weight positive cells the EMA model scores "
                         "below chop_tau on THIS dropped view (per-draw "
                         "corruption rejection; LL-R mirrored). Requires "
                         "--whole_study with slice_drop > 0")
    ap.add_argument("--chop_tau", type=float, default=0.5,
                    help="safety CEILING: never chop a positive the EMA "
                         "scores above this (v2 rank form; the threshold "
                         "itself is the per-target rho-quantile of the "
                         "previous epoch's positive EMA predictions)")
    ap.add_argument("--chop_min_n", type=int, default=50,
                    help="min positive samples before a target's quantile "
                         "threshold activates")
    ap.add_argument("--chop_ramp_ep", type=int, default=4,
                    help="epochs to ramp the chop budget 0 -> rho_max")
    ap.add_argument("--chop_rho_focal", type=float, default=0.15,
                    help="per-epoch chop cap, fraction of train positives: "
                         "menisci/Contusion/Fracture (drop^(k+2) math)")
    ap.add_argument("--chop_rho_mid", type=float, default=0.05,
                    help="cap for ACL/MCL/Baker's")
    ap.add_argument("--chop_rho_dif", type=float, default=0.01,
                    help="cap for OA x3/Effusion/Synovitis")
    # ---- exp-047 arm B: EP-DROP (evidence-preserving slice dropout)
    ap.add_argument("--ep_drop_protect", action="store_true",
                    help="if the Bernoulli drop removed ALL top-k EMA-"
                         "attended slices of a positive target, force-keep "
                         "one at random w.p. ep_prot_p (KeepAugment for MIL; "
                         "floors total evidence wipeout, keeps STKIM "
                         "pressure). Requires --whole_study, slice_drop > 0; "
                         "mutually exclusive with --chop_pos")
    ap.add_argument("--ep_prot_p", type=float, default=0.9)
    ap.add_argument("--ep_prot_k", type=int, default=3)
    ap.add_argument("--ep_prot_refresh", type=int, default=5,
                    help="recompute the EMA attention cache every N epochs")
    # ---- exp-048: dropout-ramp curriculum (single-run easy-early/hard-late)
    ap.add_argument("--slice_drop_end", type=float, default=-1.0,
                    help=">=0 enables the ramp: slice_drop interpolates from "
                         "--slice_drop to this value between drop_ramp_ep0 "
                         "and drop_ramp_ep1 (hold before/after). The clean "
                         "one-run form of the exp-045c curriculum")
    ap.add_argument("--drop_ramp_ep0", type=int, default=0)
    ap.add_argument("--drop_ramp_ep1", type=int, default=-1,
                    help="-1 = last epoch")
    # ---- exp-051: input-geometry variant from the SAME cache
    ap.add_argument("--saff_p", type=float, default=0.0,
                    help="per-study prob of the in-plane affine aug (train only)")
    ap.add_argument("--saff_rot", type=float, default=10.0, help="max rotation, degrees")
    ap.add_argument("--saff_scale", type=float, default=0.12, help="zoom range (1+s)^U[-1,1]")
    ap.add_argument("--saff_shift", type=float, default=0.08, help="max shift, frac of half-width")
    ap.add_argument("--dres_p", type=float, default=0.0,
                    help="per-study prob of down-up resampling aug (train only)")
    ap.add_argument("--dres_min", type=float, default=0.6,
                    help="min down-up scale (1 = no change)")
    ap.add_argument("--recon_p", type=float, default=0.0,
                    help="per-study prob of noise + recon-kernel aug (train only)")
    ap.add_argument("--recon_noise", type=float, default=0.035,
                    help="max Rician noise SD on [0, 1] intensities")
    ap.add_argument("--recon_sharp", type=float, default=1.0,
                    help="max unsharp-mask amount")
    ap.add_argument("--recon_smooth", type=float, default=0.6,
                    help="max smoothing amount (negative unsharp)")
    ap.add_argument("--soft_gce", type=float, default=0.0,
                    help="SOFT_GCE mix weight lam: (1-lam)*BCE + lam*GCE (0 = off)")
    ap.add_argument("--gce_q", type=float, default=0.7,
                    help="GCE exponent q in (0, 1]")
    ap.add_argument("--zoom_px", type=int, default=0,
                    help="center-crop the cached slice to this many px and "
                         "resize back (e.g. 168 = central 112 mm of the "
                         "150 mm cache at 0.5 mm/px effective). Applies "
                         "identically at train and eval; 0 = off")
    # ---- exp-057: adaptive progressive unfreeze (user design)
    ap.add_argument("--adaptive_unfreeze", type=int, default=0,
                    help="heads-only until val stalls this many epochs, then "
                         "unfreeze the top half of the trainable blocks; the "
                         "rest on the next stall. OPEN-ENDED: pair with a "
                         "generous --epochs cap + --patience/--min_ep for "
                         "convergence-based stopping. 0 = off")
    ap.add_argument("--adap_span_ep", type=int, default=15,
                    help="epochs over which each fired group cosines from "
                         "peak to the floor (then holds)")
    ap.add_argument("--adap_min_lr", type=float, default=0.05,
                    help="LR floor as a fraction of each group's peak")
    ap.add_argument("--adap_min_delta", type=float, default=1e-6,
                    help="val must improve by MORE than this to reset the "
                         "unfreeze patience (guards against micro-creep "
                         "postponing the trigger indefinitely)")
    # ---- exp-058: multi-scale detached features (user design)
    ap.add_argument("--plateau_factor", type=float, default=0.5,
                    help="--sched plateau (RLROP): LR multiplier on val "
                         "plateau")
    ap.add_argument("--plateau_patience", type=int, default=3,
                    help="RLROP: epochs of no val gain before reducing LR")
    ap.add_argument("--plateau_delta", type=float, default=0.002,
                    help="RLROP: min val gain to count as improvement")
    ap.add_argument("--plateau_min", type=float, default=0.02,
                    help="RLROP: LR-scale floor (fraction of base); a "
                         "plateau at the floor ends the run")
    ap.add_argument("--plateau_ladder", default="",
                    help="RLROP per-round 'patience:min_delta' list, e.g. "
                         "'2:0.01,3:0.005,3:0.003'; round k uses entry k "
                         "(last repeats); empty = plateau_patience/delta")
    ap.add_argument("--plateau_metric", default="val",
                    choices=["val", "gold", "either", "mean"],
                    help="RLROP improvement instrument: EMA val AUC (n~870, "
                         "sd ~0.002), gold-58 4-bag (window noise ~0.011), "
                         "either, or their mean")
    ap.add_argument("--adap_stages", type=int, default=2,
                    help="adaptive_unfreeze stage count: 2 = classic halves; "
                         "1 = decoder-first (ALL trainable bb params fire "
                         "together on the first min-delta stall - pairs "
                         "with --lora_r for decoder-first LoRA)")
    ap.add_argument("--quad_pool", type=int, default=0,
                    help="exp-065 QUAD (jury winner): append 2x2 quadrant "
                         "patch-mean features (4*hidden dims) to the decoder "
                         "input - spatial/compartment identity. Pairs with "
                         "--proj_adapter 1 + --init_lax for a function-"
                         "preserving warm start")
    ap.add_argument("--holdout_csv", default="",
                    help="exp-063: CSV (StudyInstanceUID + 12 targets, NaN "
                         "ok) of labeled JURY studies excluded from train; "
                         "their AUC selects the ckpt (use with --fold 99; "
                         "gold-58 stays report-only). Excludes gold2_csv")
    ap.add_argument("--lora_r", type=int, default=0,
                    help="exp-064: LoRA rank on attention (HF q+v, timm "
                         "qkv). Requires --unfreeze_last 0; base weights "
                         "frozen; LoRA LR = --lr_backbone (set --llrd 1.0). "
                         "0 = off. Ckpts store base.*/A/B naming - merge "
                         "before kernel use")
    ap.add_argument("--lora_alpha", type=float, default=16,
                    help="LoRA scaling numerator (scale = alpha/r)")
    ap.add_argument("--hard_labels", action="store_true",
                    help="exp-074: threshold the TRAINING targets at 0.5. "
                         "Pairs with the default soft run to measure what "
                         "our extractor's calibration is actually worth - "
                         "the delta is the diagnostic, not the score")
    ap.add_argument("--full_data", action="store_true",
                    help="exp-073 (I-107): train on EVERY non-gold study "
                         "(4,349 instead of ~3,479, +25 pct). The val fold is "
                         "then inside train, so requires --select_on gold")
    ap.add_argument("--tri_gap", type=int, default=1,
                    help="exp-079: through-plane extent of a 2.5D token. The "
                         "side channels become rows r-G and r+G instead of "
                         "r-1/r+1, so the token spans 2G slice-gaps at "
                         "IDENTICAL compute. 1 = current behaviour")
    ap.add_argument("--rank_lambda", type=float, default=0.0,
                    help="exp-078: weight of a per-target pairwise ranking "
                         "surrogate added to the soft BCE. The metric is "
                         "macro ROC AUC (pure ranking); BCE optimises it only "
                         "indirectly. 0 = current behaviour")
    ap.add_argument("--rank_queue", type=int, default=1024,
                    help="exp-078: FIFO of recent detached logits that the "
                         "batch is ranked against (batch is 2 studies, far "
                         "too few pairs on its own)")
    ap.add_argument("--rank_margin", type=float, default=1.0,
                    help="exp-078: squared-hinge margin in logit units")
    ap.add_argument("--rank_conf", type=float, default=0.25,
                    help="exp-078: a soft label only defines a pos/neg for "
                         "the ranking term if it is this far from 0.5")
    ap.add_argument("--select_on", default="val", choices=["val", "gold"],
                    help="which metric picks the best checkpoint; the public "
                         "top recipes select on the gold set")
    ap.add_argument("--pos_weight", action="store_true",
                    help="exp-073: class-balanced BCE, pos_weight = "
                         "clip((1-prev)/prev, 1, 10) per target, on soft "
                         "targets (from the 0.936 public recipe)")
    ap.add_argument("--topk", type=int, default=0,
                    help="exp-073: average the weights of the best K epochs "
                         "and save ckpt_fold{f}_topk.pt (free at inference)")
    ap.add_argument("--bag_mode", default="random",
                    choices=["random", "window"],
                    help="exp-072 slice geometry: 'random' = uniform draw "
                         "over every slice of the study (current); 'window' "
                         "= a CONTIGUOUS centred window per series, which a "
                         "competitor measured at +0.018 over equal-spaced")
    ap.add_argument("--bag_jitter", type=float, default=0.5,
                    help="--bag_mode window: 0 = always exactly centred, "
                         "1 = window start uniform over the whole series")
    ap.add_argument("--xcontrast_p", type=float, default=0.0,
                    help="exp-067 CROSS-CONTRAST tokens: probability that a "
                         "token's channel 0 is the OTHER contrast class "
                         "(T1/PD-like vs T2FS/STIR-like) at the same "
                         "anatomical level in the same plane, instead of the "
                         "anchor's previous neighbour. 1.0 = every token; "
                         "0 = off. Slices with no partner fall back to the "
                         "3-neighbour triplet")
    ap.add_argument("--lora_targets", default="qv", choices=["qv", "all"],
                    help="exp-066: 'qv' = attention q,v (HF) / fused qkv "
                         "(timm); 'all' = q,k,v,out-proj,fc1,fc2 with "
                         "per-projection factors (timm qkv split 3-way)")
    # ---- exp-066: exact full-state resume + WSD trunk / branch cooldowns
    ap.add_argument("--resume_state", default="",
                    help="state_last.pt / state_epN.pt to continue EXACTLY "
                         "(weights, EMA, optimizer, schedule, RNG); 'auto' = "
                         "<out_dir>/state_last.pt if present else fresh")
    ap.add_argument("--branch_cooldown", action="store_true",
                    help="with --resume_state: start a cooldown-to-zero "
                         "from the trunk's current LR into a FRESH out_dir "
                         "(trunk state is read-only)")
    ap.add_argument("--save_state", type=int, default=1,
                    help="write <out_dir>/state_last.pt every epoch (0=off)")
    ap.add_argument("--keep_state_every", type=int, default=0,
                    help="also keep state_ep{N}.pt every K epochs (branch "
                         "points); 0 = only state_last.pt")
    ap.add_argument("--cooldown_ep", type=int, default=0,
                    help="cooldown length in epochs; 0 = cooldown_frac * "
                         "trunk epochs, floor cooldown_min_ep")
    ap.add_argument("--cooldown_frac", type=float, default=0.2)
    ap.add_argument("--cooldown_min_ep", type=int, default=2)
    ap.add_argument("--cooldown_shape", default="linear",
                    choices=["linear", "cosine", "sqrt"],
                    help="decay shape from the frozen scale to exactly 0")
    ap.add_argument("--cooldown_at_ep", type=int, default=-1,
                    help="--sched wsd: start the cooldown after this many "
                         "trunk epochs (-1 = never inside the trunk)")
    ap.add_argument("--cooldown_patience", type=int, default=0,
                    help="--sched wsd: start the cooldown after this many "
                         "epochs without an EMA-val gain > cooldown_min_delta")
    ap.add_argument("--cooldown_min_delta", type=float, default=0.002)
    ap.add_argument("--warm_steps", type=int, default=0,
                    help="plateau/wsd: warmup length in optimizer steps "
                         "(per group, from its own start); 0 = warmup_frac "
                         "of --epochs")
    ap.add_argument("--eval_batch", type=int, default=8,
                    help="bags per eval forward (8 x bag-32 = 256 images "
                         "OOMs attention-heavy hybrids; drop to 1-2 there)")
    ap.add_argument("--feat_dir", default="",
                    help="exp-061: dir of feats_part{k}.npy pooled-feature "
                         "banks (row-aligned with slices_part{k}.npy). Use "
                         "with --backbone feat:<hidden> (feat rows are "
                         "2*hidden = [cls, patch-mean]); decoder-only")
    ap.add_argument("--proj_adapter", type=int, default=0,
                    help="1 = keep base proj at its tap-free width (warm-"
                         "loadable) and route ms_layers/aux features through "
                         "a parallel ZERO-INIT LayerNorm+Linear branch: a "
                         "lax warm start then equals the source model at "
                         "step 0 and taps can only add on top")
    ap.add_argument("--ms_layers", default="",
                    help="comma-separated block indices (0-based) whose "
                         "pooled cls+mean features join the decoder input, "
                         "DETACHED (no gradient). e.g. '3,7' with the final "
                         "layer always included. HF backbones only")
    # ---- exp-060: frozen dual-backbone fusion (user design)
    ap.add_argument("--aux_backbone", default="",
                    help="timm:<name> second tower, FROZEN forever, forward-"
                         "only; its pooled cls+mean join the proj input "
                         "(stationary by construction - the Meta-intended "
                         "frozen use of DINOv3). Shares the main tower's "
                         "input normalization: pair matching-stat towers")
    # ---- exp-040: 336px bank support
    ap.add_argument("--n_parts", type=int, default=3,
                    help="SliceBank part count (allslice224=3, allslice336=5)")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps (effective batch = "
                         "batch x accum); default 1 = legacy behaviour")
    # ---- exp-042: ACMIL stochastic top-k attention masking
    ap.add_argument("--acmil_p", type=float, default=0.0,
                    help="P(mask the top-k attended slices per target) during "
                         "training; 0 = off")
    ap.add_argument("--acmil_k", type=int, default=1)
    # ---- exp-043: patch-token cross-attention decoder (zero-gated residual)
    ap.add_argument("--decoder", choices=["mil", "xattn"], default="mil")
    # ---- exp-044: encoder pooling
    ap.add_argument("--pool", choices=["cls_mean", "cls_mean_focal"],
                    default="cls_mean")
    # ---- exp-045: whole-study training (no bag subsampling)
    ap.add_argument("--whole_study", action="store_true",
                    help="every slice of the study per sample (batch must be "
                         "1; use --accum for effective batch); eval = one "
                         "deterministic full-study forward")
    ap.add_argument("--slice_drop", type=float, default=0.0,
                    help="whole-study train-time slice dropout (the explicit "
                         "form of bagging's implicit regularization)")
    ap.add_argument("--max_slices", type=int, default=256)
    # ---- gold_v2 dual-instrument reporting (agent-jury expansion, 2026-08-26)
    ap.add_argument("--gold2_csv", default="",
                    help="path to gold_v2.csv (sparse cells = NaN). Reports a "
                         "second FOLD-AWARE gold AUC: gold-58 studies plus "
                         "only the NEW gold_v2 studies in THIS fold's val "
                         "split (new studies elsewhere were trained on via "
                         "v07 -> excluded). Zero extra forwards: predictions "
                         "reuse pg (gold-58) and pv (val).")
    # ---- LLM-agreement sample weights (weights_v01)
    ap.add_argument("--weight_csv", default="",
                    help="path to a per-cell weight csv (StudyInstanceUID + "
                         "12 target columns, values in (0,1]); multiplied "
                         "into the BCE cell weights. Missing studies/cells "
                         "default to 1.0")
    return ap


def main():
    args = build_parser().parse_args()
    cd = Path(args.csv_dir)
    folds_df = pd.read_csv(cd / "folds.csv")
    studies = list(folds_df["StudyInstanceUID"])
    folds_map = dict(zip(folds_df["StudyInstanceUID"], folds_df["fold"]))
    lab = pd.read_csv(cd / args.labels_csv).set_index("StudyInstanceUID")
    lab_ev = pd.read_csv(cd / args.eval_labels_csv).set_index("StudyInstanceUID")
    # P3.1 keep NaN - run_fold masks unknown cells
    soft = lab.reindex(studies)[TARGETS].values.astype(np.float32)
    soft_eval = lab_ev.reindex(studies)[TARGETS].values.astype(np.float32)
    train_df = pd.read_csv(cd / "train.csv")
    gold_df = train_df[train_df[TARGETS].notna().all(axis=1)]
    gold_map = gold_df.set_index("StudyInstanceUID")[TARGETS]
    gold_idx = np.array([i for i, s in enumerate(studies) if s in gold_map.index])
    gold_y = gold_map.reindex([studies[i] for i in gold_idx]).values.astype(int)
    bank = SliceBank(args.cache_dir, n_parts=getattr(args, "n_parts", 3),
                     feat_dir=getattr(args, "feat_dir", "") or None)
    log(f"bank: {bank.n_slices_total} usable slices, "
        f"{len(bank.by_study)} studies, {bank.img}px")
    dev = torch.device("cuda")
    r = run_fold(args.fold, studies, bank, soft, soft_eval, gold_y, gold_idx,
                 folds_map, args, dev)
    mp = Path(args.out_dir) / "metrics.json"
    res = json.load(open(mp)) if mp.is_file() else {}
    res[str(args.fold)] = r
    json.dump(res, open(mp, "w"), indent=1)
    log(f"DONE fold {args.fold}: val {r['val_auc']:.4f} gold {r['gold_auc']:.4f} "
        f"ep {r['epoch']}")


if __name__ == "__main__":
    main()
