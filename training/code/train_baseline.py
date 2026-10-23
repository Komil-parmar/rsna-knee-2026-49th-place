"""exp-001 baseline trainer: DINOv2-small slot-attention model on labels-v04.

Re-implements the public 0.832 architecture (notebooks_public/NOTES.md) on our
assets: 512^2 preprocessed npz shards, scanner-fingerprint folds, labels-v04
soft targets, precomputed slot table. Gold-58 studies are NEVER trained on and
are evaluated every epoch (the only honest metric; fold-val vs v04 is a noisy
proxy).

Colab (A100, after `python colab_pull_shards.py --dest /content/rsna_data`):
    files needed next to this script: folds.csv, labels_v04.csv,
    series_slots.csv, train.csv
    !pip install -q transformers timm
    !python train_baseline.py --prep_dir /content/rsna_data/prep --fold 0
    !python train_baseline.py --prep_dir /content/rsna_data/prep --fold all
Outputs: ckpt_fold{k}.pt + metrics.json in --out_dir (persist those to Drive).
"""
import argparse
import copy
import gc
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
# left<->right flip aug swaps medial<->lateral targets (images are left-knee
# normalized; a consistent flip turns them into a valid right-knee sample)
SWAP = {"Medial Meniscus": "Lateral Meniscus",
        "Lateral Meniscus": "Medial Meniscus",
        "Medial OA": "Lateral OA", "Lateral OA": "Medial OA"}
SWAP_IDX = [TARGETS.index(SWAP.get(t, t)) for t in TARGETS]
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS",
         "COR_T1", "SAG_T1"]
SLOT_PLANE = ["Sagittal", "Coronal", "Axial", "Sagittal", "Coronal",
              "Sagittal"]
N_SLOT, GROUP, N_GROUP = len(SLOTS), 3, 3
CACHE_SLICES = GROUP * N_GROUP
T0 = time.time()
# A100 -> bf16; Kaggle T4/P100 -> fp16 + GradScaler
AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.is_bf16_supported() else torch.float16)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- cache
_worker_args = {}


def _init_worker(prep_dir, slot_rows, img, crop_px):
    _worker_args.update(prep=Path(prep_dir), slots=slot_rows, img=img,
                        crop_px=crop_px)


def _build_study(study):
    prep, img = _worker_args["prep"], _worker_args["img"]
    crop_px = _worker_args["crop_px"]  # 130mm equiv: 416 of the 512 (I-027)
    out = np.zeros((N_SLOT, CACHE_SLICES, img, img), np.uint8)
    mask = np.zeros(N_SLOT, np.float32)
    try:
        with np.load(prep / f"{study}.npz") as z:
            for si, slot in enumerate(SLOTS):
                uid = _worker_args["slots"].get((study, slot))
                if uid is None or uid not in z.files:
                    continue
                vol = z[uid]
                if crop_px and crop_px < vol.shape[1]:
                    o = (vol.shape[1] - crop_px) // 2
                    vol = vol[:, o:o + crop_px, o:o + crop_px]
                n = vol.shape[0]
                lo, hi = int(0.20 * (n - 1)), int(0.80 * (n - 1))
                idx = (np.unique(np.linspace(lo, hi, CACHE_SLICES).astype(int))
                       if hi > lo else np.array([n // 2]))
                while len(idx) < CACHE_SLICES:
                    idx = np.append(idx, idx[-1])
                for j, k in enumerate(idx[:CACHE_SLICES]):
                    out[si, j] = cv2.resize(vol[int(k)], (img, img),
                                            interpolation=cv2.INTER_AREA)
                mask[si] = 1.0
    except Exception as exc:
        return study, None, None, str(exc)[:120]
    return study, out, mask, None


def build_cache(studies, prep_dir, slot_rows, img, workers, cache_path,
                crop_px=0):
    if (cache_path and Path(cache_path + ".npy").is_file()
            and not Path(cache_path + "_studies.json").is_file()):
        log("cache incomplete (no studies.json) - rebuilding")
        for suffix in (".npy", "_mask.npy"):
            Path(cache_path + suffix).unlink(missing_ok=True)
    if cache_path and Path(cache_path + ".npy").is_file():
        log(f"loading cache {cache_path}")
        cache = np.load(cache_path + ".npy")
        mask = np.load(cache_path + "_mask.npy")
        order = json.load(open(cache_path + "_studies.json"))
        if order != list(studies):
            log("cache study-list mismatch - rebuilding")
            for suffix in (".npy", "_mask.npy", "_studies.json"):
                Path(cache_path + suffix).unlink(missing_ok=True)
            return build_cache(studies, prep_dir, slot_rows, img, workers,
                               cache_path, crop_px)
        return cache, mask
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, img, img), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f"building cache {cache.shape} = {cache.nbytes / 2**30:.1f} GB RAM "
        f"(crop_px={crop_px})")
    sidx = {s: i for i, s in enumerate(studies)}
    errs = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(prep_dir, slot_rows, img, crop_px)) as pool:
        for n, (study, out, m, err) in enumerate(
                pool.map(_build_study, studies, chunksize=16)):
            if err:
                errs += 1
                continue
            cache[sidx[study]] = out
            mask[sidx[study]] = m
            if (n + 1) % 500 == 0:
                log(f"  cache {n + 1}/{len(studies)}")
    log(f"cache done ({errs} study errors)")
    if cache_path:
        np.save(cache_path + ".npy", cache)
        np.save(cache_path + "_mask.npy", mask)
        json.dump(list(studies), open(cache_path + "_studies.json", "w"))
    return cache, mask


# ---------------------------------------------------------------- model
class SlotHead(nn.Module):
    """Per-target attention over slot embeddings (port of the 0.832 head)."""

    def __init__(self, dim, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                  nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(N_SLOT, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(len(TARGETS), hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, len(TARGETS))
        self.hidden = hidden
        prior = torch.zeros(len(TARGETS), N_SLOT)
        preferred = {"ACL": (0, 3, 5), "MCL": (1, 4),
                     "Medial Meniscus": (0, 1, 3, 4),
                     "Lateral Meniscus": (0, 1, 3, 4),
                     "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
                     "PF OA": (0, 2, 5), "Effusion": (0, 2),
                     "Synovitis": (0, 2), "Baker's": (0,),
                     "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5)}
        for t, ss in preferred.items():
            prior[TARGETS.index(t), list(ss)] = 0.55
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = (torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
               + self.slot_prior.unsqueeze(0))
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class Model(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(dim)
        self.register_buffer("mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        k = max(1, patch.shape[1] // 8)
        focal = patch.topk(k, dim=1).values.mean(1)
        feat = torch.cat([out[:, 0], patch.mean(1), focal],
                         dim=1).reshape(B, S, -1)
        return self.head(feat, mask)


def build_model(backbone="facebook/dinov2-small", unfreeze_last=6):
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(backbone)
    for p in bb.parameters():
        p.requires_grad = False
    for blk in bb.encoder.layer[-unfreeze_last:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in bb.layernorm.parameters():
        p.requires_grad = True
    return Model(bb, bb.config.hidden_size * 3)


# ---------------------------------------------------------------- train
def flip_batch(imgs, mask):
    """Consistent left->right flip: hflip cor/ax, reverse sag slice order."""
    out = imgs.clone()
    for si, plane in enumerate(SLOT_PLANE):
        if plane in ("Coronal", "Axial"):
            out[:, si] = torch.flip(out[:, si], dims=[-1])
        else:
            out[:, si] = torch.flip(out[:, si], dims=[1])
    return out


def take_group(rows, g):
    return rows[:, :, g * GROUP:(g + 1) * GROUP]


# I-025 (LB-validated +0.006): focal findings live in ~2 slices; a mean over
# slice windows dilutes their evidence. Pool those targets with max instead.
POOL_MAX = ["Fracture", "Contusion", "Medial Meniscus", "Lateral Meniscus",
            "Baker's"]
POOL_MAX_IDX = [TARGETS.index(t) for t in POOL_MAX]


@torch.no_grad()
def predict(model, cache, mask, idx, dev, batch=24, target_pool=True):
    model.eval()
    out = []
    for b in range(0, len(idx), batch):
        sel = idx[b:b + batch]
        rows = torch.from_numpy(cache[sel]).to(dev)
        m = torch.from_numpy(mask[sel]).to(dev)
        probs = []
        for g in range(N_GROUP):
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                z = model(take_group(rows, g), m).float()
            probs.append(torch.sigmoid(z))
        probs = torch.stack(probs)                      # [G, B, T]
        pooled = probs.mean(0)
        if target_pool:
            pooled[:, POOL_MAX_IDX] = probs[:, :, POOL_MAX_IDX].max(0).values
        out.append(pooled.cpu().numpy())
    return np.concatenate(out)


def macro_auc(y, p, valid=None):
    """Macro ROC AUC over columns. `valid` (bool, same shape as y) restricts
    column j to rows where valid[:, j] (P3.1: unknown label cells are NOT
    scored). Columns with <2 classes after masking are NaN and excluded."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    p = np.asarray(p)
    if valid is None:
        valid = np.ones(y.shape, bool)
    aucs = []
    for j in range(y.shape[1]):
        v = valid[:, j]
        yj = y[v, j].astype(int)
        aucs.append(roc_auc_score(yj, p[v, j]) if len(np.unique(yj)) > 1
                    else np.nan)
    return float(np.nanmean(aucs)), aucs


def eval_targets(soft_eval):
    """Reference soft labels -> (binary targets, validity mask). NaN cells and
    exact 0.5 ties are UNKNOWN (excluded), positives are STRICTLY > 0.5."""
    raw = np.asarray(soft_eval, dtype=np.float32)
    valid = np.isfinite(raw) & (raw != 0.5)
    y_bin = (raw > 0.5).astype(int)
    return y_bin, valid


def batch_weights(tgt, lmask, conf_weight):
    """Per-cell BCE weights: 0 for unknown (NaN) label cells, optionally
    scaled by soft-label confidence 0.3 + 0.7|2p-1| (I-0xx conf_weight)."""
    w = lmask.to(tgt.dtype)
    if conf_weight:
        w = w * (0.3 + 0.7 * (2 * tgt - 1).abs())
    return w


def maybe_flip(imgs, m, tgt, w, rng, flip_p):
    """Training-time left<->right flip. ALWAYS draws one rng number so the
    per-step RNG stream is identical whether or not the flip is enabled
    (paired comparison against pre-P3 runs at equal seed). flip_p defaults
    to 0 everywhere: the legacy flip_batch reverses the 3-slice WINDOW, not
    the sagittal stack, i.e. the model saw the same anatomical depth with
    medial/lateral labels swapped on 2 of 3 groups (diagnosis 2026-08-21 §1)."""
    u = rng.random()
    if flip_p <= 0 or u >= flip_p:
        return imgs, tgt, w, False
    imgs = flip_batch(imgs, m)
    tgt = tgt[:, SWAP_IDX]
    w = w[:, SWAP_IDX].clone()          # weights/masks travel with targets
    # MCL is medial-sided with no LCL counterpart target: its label is wrong
    # on mirrored knees, so drop it from the loss.
    w[:, TARGETS.index("MCL")] = 0.0
    return imgs, tgt, w, True


def ema_decay_at(decay, step):
    """timm-style warmup so the shadow is not dominated by init weights for
    the first ~1/(1-decay) steps: d = min(decay, (1+step)/(10+step))."""
    return min(decay, (1.0 + step) / (10.0 + step))


@torch.no_grad()
def ema_update(ema, model, decay, step=None):
    d = decay if step is None else ema_decay_at(decay, step)
    for pe, pm in zip(ema.parameters(), model.parameters()):
        if pm.requires_grad:
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)


def make_lr_lambda(total, warm_frac=0.0, head_only_steps=0, is_backbone=False):
    """LambdaLR factor: [head-only hold (backbone only)] -> linear warmup ->
    cosine to 0. With warm_frac=0 and head_only_steps=0 this is exactly the
    closed form of CosineAnnealingLR(T_max=total) (pre-P3 behaviour)."""
    hold = int(head_only_steps) if is_backbone else 0
    span = max(total - hold, 1)
    warm = int(round(warm_frac * span))

    def f(step):
        if step < hold:
            return 0.0
        t = step - hold
        wf = min(1.0, (t + 1) / warm) if warm > 0 else 1.0
        return wf * 0.5 * (1.0 + math.cos(math.pi * min(t, span) / span))
    return f


def build_param_groups(model, lr_bb, lr_head, wd, llrd=1.0,
                       backbone_attr="backbone"):
    """Layer-wise LR decay over the backbone's transformer blocks:
    block i of n gets lr_bb * llrd**(n-1-i); final layernorm gets lr_bb.
    Every trainable parameter lands in exactly one group (asserted)."""
    bb = getattr(model, backbone_attr)
    blocks = list(bb.encoder.layer)
    n = len(blocks)
    groups, seen = [], set()
    for i, blk in enumerate(blocks):
        ps = [p for p in blk.parameters() if p.requires_grad]
        if ps:
            groups.append({"params": ps, "lr": lr_bb * (llrd ** (n - 1 - i)),
                           "weight_decay": wd, "kind": "bb",
                           "name": f"block{i}"})
            seen.update(id(p) for p in ps)
    rest_bb = [p for p in bb.parameters()
               if p.requires_grad and id(p) not in seen]
    if rest_bb:                                   # final layernorm, embeddings
        groups.append({"params": rest_bb, "lr": lr_bb, "weight_decay": wd,
                       "kind": "bb", "name": "bb_rest"})
        seen.update(id(p) for p in rest_bb)
    head = [p for p in model.parameters()
            if p.requires_grad and id(p) not in seen]
    groups.append({"params": head, "lr": lr_head, "weight_decay": wd,
                   "kind": "head", "name": "head"})
    seen.update(id(p) for p in head)
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(seen) == n_train, (len(seen), n_train)
    return groups


def run_fold(fold, studies, cache, mask, soft, gold_y, gold_idx, folds_map,
             args, dev, soft_eval=None):
    # P1.1 determinism: torch init + dropout seeded; numpy rng below.
    seed = int(getattr(args, "seed", 1000))
    torch.manual_seed(seed * 100 + fold)
    torch.cuda.manual_seed_all(seed * 100 + fold)
    env = {"seed": seed, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0)}
    # P1.2 decision metric: val is ALWAYS scored against a fixed reference
    # label set (v06), never against the (possibly self-referential) training
    # labels. Falls back to the training labels if no reference is passed.
    # P3.1: both arrays MAY carry NaN (= unknown cell). Unknown training cells
    # get zero loss weight; unknown/tied eval cells are excluded from the AUC.
    soft = np.asarray(soft, dtype=np.float32)
    if soft_eval is None:
        soft_eval = soft
    label_mask_np = np.isfinite(soft)
    soft = np.nan_to_num(soft, nan=0.5)
    y_eval_bin, y_eval_valid = eval_targets(soft_eval)
    n_unk = int((~label_mask_np).sum())
    n_tie = int((~y_eval_valid).sum())
    if n_unk or n_tie:
        log(f"P3.1 label mask: {n_unk} unknown train cells (zero weight), "
            f"{n_tie} unknown/tied eval cells (excluded from val AUC)")
    label_mask = torch.from_numpy(label_mask_np)
    flip_p = float(getattr(args, "flip_p", 0.0))
    ema_eval = bool(getattr(args, "ema_eval", False))
    llrd = float(getattr(args, "llrd", 1.0))
    warm_frac = float(getattr(args, "warmup_frac", 0.0))
    head_only_ep = int(getattr(args, "head_only_ep", 0))
    log(f"P3 flags: flip_p {flip_p} | ema_eval {ema_eval} | llrd {llrd} | "
        f"warmup {warm_frac} | head_only_ep {head_only_ep}")
    if float(getattr(args, "lambda_h", 0.0)) > 0 and args.lambda_a <= 0:
        raise ValueError("lambda_h requires lambda_a > 0 (Arazo: entropy "
                         "term diverges without the balance term)")
    is_val = np.array([folds_map[s] == fold for s in studies])
    is_gold = np.zeros(len(studies), bool)
    is_gold[gold_idx] = True
    tr_idx = np.where(~is_val & ~is_gold)[0]
    va_idx = np.where(is_val & ~is_gold)[0]
    log(f"fold {fold}: train {len(tr_idx)}, val {len(va_idx)}, "
        f"gold {len(gold_idx)} | seed {seed} | {env['gpu']}")

    model = build_model(args.backbone, args.unfreeze_last).to(dev)
    # P3.4 LLRD param groups (llrd=1.0 -> the old two-group optimizer)
    groups = build_param_groups(model, args.lr_backbone, args.lr_head,
                                args.wd, llrd)
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    steps_per_ep = len(tr_idx) // args.batch + 1
    steps = args.epochs * steps_per_ep
    # P3.3 warmup + head-only hold + cosine (== CosineAnnealingLR when both 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [
        make_lr_lambda(steps, warm_frac, head_only_ep * steps_per_ep,
                       is_backbone=(g["kind"] == "bb")) for g in groups])
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)
    y_soft = torch.from_numpy(soft).float()
    y_dyn = y_soft.clone()                       # dynamic pseudo-labels
    dyn_on = args.dynamic_pl > 0
    prior = y_soft[tr_idx].mean(0).to(dev)       # per-target prevalence
    pl_mask = torch.ones(len(TARGETS))
    tgt_list = getattr(args, "dynamic_pl_targets", "")
    if tgt_list:
        pl_mask = torch.zeros(len(TARGETS))
        for t in tgt_list.split("|"):
            pl_mask[TARGETS.index(t.strip())] = 1.0

    # P1.3-P1.5 anchors: immutable hard-consensus labels from --anchor_csv
    # (per-target 1/0/NaN), target-balanced sampling, ADDED to the batch (the
    # ordinary stream is untouched). Legacy substitutive conf-pool kept only
    # for reproducing pre-P1 runs when no anchor_csv is given.
    anchor_csv = getattr(args, "anchor_csv", "")
    anchor_rows = np.array([], int)
    anchor_pos, anchor_neg = {}, {}
    legacy_pool = np.array([], int)
    y_anchor = None
    defined_np = None
    if anchor_csv:
        adf = pd.read_csv(anchor_csv).set_index("StudyInstanceUID")
        avals = adf.reindex(studies)[TARGETS].values.astype(np.float32)
        tr_mask_rows = np.zeros(len(studies), bool)
        tr_mask_rows[tr_idx] = True
        defined_np = (~np.isnan(avals)) & tr_mask_rows[:, None]
        amin = int(getattr(args, "anchor_min_pos", 15))
        for tj, t in enumerate(TARGETS):
            anchor_pos[tj] = np.where(defined_np[:, tj] & (avals[:, tj] == 1))[0]
            anchor_neg[tj] = np.where(defined_np[:, tj] & (avals[:, tj] == 0))[0]
            if dyn_on and len(anchor_pos[tj]) < amin:
                raise ValueError(f"anchor pool too small for {t}: "
                                 f"{len(anchor_pos[tj])} < {amin} positives")
        anchor_rows = np.where(defined_np.any(1))[0]
        y_anchor = y_soft.clone()
        dv = torch.from_numpy(np.nan_to_num(avals, nan=0.0))
        dm = torch.from_numpy(defined_np)
        y_anchor[dm] = dv[dm]
        y_dyn = y_anchor.clone()
        log("anchors: " + " ".join(
            f"{t[:4]}+{len(anchor_pos[j])}" for j, t in enumerate(TARGETS))
            + f" | rows {len(anchor_rows)}")
    else:
        conf0 = (2 * y_soft - 1).abs().mean(1)
        legacy_pool = np.array([i for i in tr_idx
                                if conf0[i] >= args.anchor_conf])
    upd_idx = np.setdiff1d(tr_idx, anchor_rows)  # rows harvest may rewrite
    y_base = y_anchor if y_anchor is not None else y_soft
    base_prev = y_base[upd_idx].numpy().mean(0)  # for drift alarm
    log(f"dynamic_pl={args.dynamic_pl} on {int(pl_mask.sum())} targets; "
        f"anchors={'csv:' + str(len(anchor_rows)) if anchor_csv else 'legacy:' + str(len(legacy_pool))}")

    # P1.6 EMA shadow model = pseudo-label harvesting teacher; P3.2 the same
    # shadow is the EVAL + CHECKPOINT model when --ema_eval (I-051 fix a).
    ema_model = None
    if dyn_on or ema_eval:
        ema_model = copy.deepcopy(model).eval()
        for p_ in ema_model.parameters():
            p_.requires_grad_(False)
    ema_decay = float(getattr(args, "ema", 0.999))
    global_step = 0

    faa_policy = None
    fp = getattr(args, "faa_policy", "")
    if fp:
        from mri_augs import apply_subpolicy
        pol = json.load(open(fp))["sub_policies"]
        faa_policy = [[(op, float(pp), float(ll)) for op, pp, ll in sub]
                      for sub in pol]
        log(f"FAA policy loaded: {len(faa_policy)} sub-policies from {fp}")
    dump_dir = Path(args.out_dir) / "audit"
    dump_dir.mkdir(parents=True, exist_ok=True)
    best = {"gold_auc": 0.0, "val_auc": 0.0}
    history = []
    rho, rho_frozen, agree0, agree_bad = 0.0, False, None, 0
    anchor_cycle = 0
    rng = np.random.default_rng(seed + fold)
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr_idx)
        losses = []
        for b in range(0, len(order), args.batch):
            sel = order[b:b + args.batch].copy()
            if anchor_csv:
                # P1.4/P1.5: k anchors APPENDED (ordinary stream untouched),
                # cycling targets so every target contributes hard positives
                extra_rows = []
                for _ in range(int(getattr(args, "anchor_k", 2))):
                    tj = anchor_cycle % len(TARGETS)
                    anchor_cycle += 1
                    pool = (anchor_pos[tj]
                            if (rng.random() < 0.5 or not len(anchor_neg[tj]))
                            else anchor_neg[tj])
                    extra_rows.append(int(rng.choice(pool)))
                sel = np.concatenate([sel, np.array(extra_rows)])
            elif dyn_on and len(legacy_pool) and len(sel) >= 4:
                sel[-2:] = rng.choice(legacy_pool, 2)   # legacy (pre-P1)
            rows = torch.from_numpy(cache[sel]).to(dev)
            m = torch.from_numpy(mask[sel]).to(dev)
            tgt = y_dyn[sel].to(dev)
            if faa_policy is not None:
                x9 = rows.float() / 255.0               # [B,S,9,H,W]
                for i in range(x9.shape[0]):
                    sub = faa_policy[int(rng.integers(len(faa_policy)))]
                    g_ = torch.Generator(device="cpu").manual_seed(
                        int(rng.integers(2 ** 31)))
                    x9[i:i + 1] = apply_subpolicy(x9[i:i + 1], sub, g_)
                rows = x9 * 255.0
            g = int(rng.integers(N_GROUP))
            imgs = take_group(rows, g)
            # P3.1 unknown cells -> weight 0; conf_weight trusts confident
            # soft labels more than near-0.5 ones. P3.0 flip is OFF unless
            # --flip_p > 0 (weights/masks are permuted with the targets).
            w = batch_weights(tgt, label_mask[sel].to(dev), args.conf_weight)
            imgs, tgt, w, _ = maybe_flip(imgs, m, tgt, w, rng, flip_p)
            mix_a = float(getattr(args, "mixup_alpha", 0.0))
            if mix_a > 0:
                # P1.7 BCE-compatible mixup: mix images, soft targets and
                # weights with one lambda; slot masks combine as min so a
                # slot missing in either partner is dropped, not half-blank.
                lam = float(rng.beta(mix_a, mix_a))
                perm = torch.from_numpy(rng.permutation(len(sel))).to(dev)
                imgs = imgs.float()
                imgs = lam * imgs + (1 - lam) * imgs[perm]
                tgt = lam * tgt + (1 - lam) * tgt[perm]
                w = lam * w + (1 - lam) * w[perm]
                m = torch.minimum(m, m[perm])
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                z = model(imgs, m)
                loss = (F.binary_cross_entropy_with_logits(
                    z.float(), tgt, reduction="none") * w).sum() / w.sum()
                if args.lambda_a > 0:
                    h = torch.sigmoid(z.float()).mean(0).clamp(1e-4, 1 - 1e-4)
                    pr = prior.clamp(1e-4, 1 - 1e-4)
                    r_a = (pr * (pr / h).log()
                           + (1 - pr) * ((1 - pr) / (1 - h)).log()).mean()
                    loss = loss + args.lambda_a * r_a
                lh = float(getattr(args, "lambda_h", 0.0))
                if lh > 0:
                    px = torch.sigmoid(z.float()).clamp(1e-4, 1 - 1e-4)
                    r_h = -(px * px.log()
                            + (1 - px) * (1 - px).log()).mean()
                    loss = loss + lh * r_h
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema_model is not None:
                ema_update(ema_model, model, ema_decay, global_step)
            global_step += 1
            losses.append(loss.item())
        eval_model = ema_model if ema_eval else model
        pv = predict(eval_model, cache, mask, va_idx, dev)
        va_auc, va_per = macro_auc(y_eval_bin[va_idx], pv,
                                   y_eval_valid[va_idx])
        pg = predict(eval_model, cache, mask, gold_idx, dev)
        gold_auc, gold_per = macro_auc(gold_y, pg)
        # P1.9 ranking-relevant alarm: mean margin of DISCORDANT gold pairs
        # (how confidently wrong the ranking is) — the AUC analogue of r_t
        margins = []
        for j in range(len(TARGETS)):
            p1 = pg[gold_y[:, j] == 1, j]
            p0 = pg[gold_y[:, j] == 0, j]
            if len(p1) and len(p0):
                d = p0[None, :] - p1[:, None]
                if (d > 0).any():
                    margins.append(float(d[d > 0].mean()))
        rank_err_margin = float(np.mean(margins)) if margins else 0.0
        extra = f" | mrg {rank_err_margin:.3f}"
        ep_rec = {"ep": ep, "val": round(va_auc, 4),
                  "gold": round(gold_auc, 4),
                  "rank_err_margin": round(rank_err_margin, 4),
                  "val_per_target": [None if np.isnan(a) else round(float(a), 4)
                                     for a in va_per]}   # P3: per-target val
        np.save(dump_dir / f"val_preds_f{fold}_ep{ep}.npy", pv)  # P1.10
        if dyn_on and ep + 1 >= args.pl_start_ep:
            if not rho_frozen:
                rho = args.dynamic_pl * min(
                    1.0, (ep + 1 - args.pl_start_ep + 1)
                    / max(args.epochs - args.pl_start_ep, 1))
            # P1.6 harvest: EMA teacher, SAME pooling as eval
            teacher = ema_model if ema_model is not None else model
            pt = predict(teacher, cache, mask, upd_idx, dev)
            pg_t = predict(teacher, cache, mask, gold_idx, dev)
            pl_agree, _ = macro_auc(gold_y, pg_t)
            if agree0 is None:
                agree0 = pl_agree
            # P1.11 auto rho-freeze: harvested-label quality dropping
            if pl_agree < agree0 - 0.010:
                agree_bad += 1
                if agree_bad >= 2 and not rho_frozen:
                    rho_frozen = True
                    log(f"RHO_FROZEN at {rho:.2f} (pl_gold_agreement "
                        f"{pl_agree:.3f} < start {agree0:.3f} - 0.010)")
            else:
                agree_bad = 0
            # rare-target collapse detector: per-target prevalence drift
            drift = (np.abs(pt.mean(0) - base_prev)
                     / np.maximum(base_prev, 1e-3))
            for j in np.where((drift > 0.30) & (pl_mask.numpy() > 0))[0]:
                pl_mask[j] = 0.0
                log(f"PL DISABLED for {TARGETS[j]} "
                    f"(prevalence drift {drift[j]:.0%})")
            rho_vec = rho * pl_mask                     # [12] per-target
            y_dyn[upd_idx] = ((1 - rho_vec) * y_base[upd_idx]
                              + rho_vec * torch.from_numpy(pt).float())
            if len(anchor_rows):
                # P1.3 immutability: anchors can NEVER drift — hard abort
                assert torch.equal(y_dyn[anchor_rows],
                                   y_anchor[anchor_rows]), \
                    "anchor labels mutated"
                samp = (anchor_rows if len(anchor_rows) <= 240 else
                        rng.choice(anchor_rows, 240, replace=False))
                pa = predict(model, cache, mask, samp, dev)
                ya = y_anchor[samp].numpy()
                a_aucs = []
                from sklearn.metrics import roc_auc_score
                for j in range(len(TARGETS)):
                    dfn = defined_np[samp, j]
                    if dfn.any() and len(set(ya[dfn, j].astype(int))) > 1:
                        a_aucs.append(roc_auc_score(ya[dfn, j].astype(int),
                                                    pa[dfn, j]))
                if a_aucs:
                    ep_rec["anchor_auc"] = round(float(np.mean(a_aucs)), 4)
            extra += f" | rho {rho:.2f} agr {pl_agree:.3f}"
            ep_rec.update(rho=round(rho, 3),
                          pl_gold_agreement=round(pl_agree, 4),
                          rho_frozen=rho_frozen,
                          max_drift=round(float(drift.max()), 3))
            if ep % 4 == 0:
                np.save(dump_dir / f"pl_snap_f{fold}_ep{ep}.npy",
                        y_dyn[tr_idx].numpy())
        history.append(ep_rec)
        log(f"fold {fold} ep {ep}: loss {np.mean(losses):.4f} | "
            f"val {va_auc:.3f} | GOLD {gold_auc:.3f}{extra}")
        # P1.2 best epoch by VAL (vs fixed reference labels), never by gold
        if va_auc > best["val_auc"]:
            best = {"gold_auc": gold_auc, "val_auc": va_auc, "epoch": ep,
                    "gold_per_target": [round(float(a), 4) for a in gold_per],
                    "val_per_target": ep_rec["val_per_target"]}
            # save immediately (out_dir may be Drive): survives disconnects
            torch.save({"state_dict": {k: v.cpu() for k, v in
                                       eval_model.state_dict().items()},
                        "fold": fold, "targets": TARGETS, "slots": SLOTS,
                        "img": args.img, "group": GROUP, "n_group": N_GROUP,
                        "ema_eval": ema_eval, "flip_p": flip_p, "llrd": llrd,
                        "metrics": best},
                       Path(args.out_dir) / f"ckpt_fold{fold}.pt")
    env.update(flip_p=flip_p, ema_eval=ema_eval, llrd=llrd,
               warmup_frac=warm_frac, head_only_ep=head_only_ep)
    best["env"] = env
    best["history"] = history
    del model
    if ema_model is not None:
        del ema_model
    gc.collect()
    torch.cuda.empty_cache()
    return best


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", default="/content/rsna_data/prep")
    ap.add_argument("--csv_dir", default=".")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--cache_path", default=None,
                    help="default: /content/cache<img> (per-size, no reuse bug)")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--backbone", default="facebook/dinov2-small")
    ap.add_argument("--unfreeze_last", type=int, default=6)
    ap.add_argument("--conf_weight", action="store_true",
                    help="weight BCE by soft-label confidence |2p-1|")
    ap.add_argument("--dynamic_pl", type=float, default=0.0,
                    help="max blend weight of model predictions into the "
                         "pseudo-labels (Arazo-style dynamic PL); 0 = off")
    ap.add_argument("--pl_start_ep", type=int, default=3)
    ap.add_argument("--dynamic_pl_targets", default="",
                    help="'|'-separated target names to apply dynamic PL to; "
                         "empty = all targets")
    ap.add_argument("--anchor_conf", type=float, default=0.8,
                    help="legacy conf-pool threshold (pre-P1 reproduction)")
    ap.add_argument("--anchor_csv", default="",
                    help="per-target hard anchor labels (1/0/NaN); enables "
                         "immutable target-balanced additive anchors")
    ap.add_argument("--anchor_k", type=int, default=2,
                    help="anchor rows APPENDED per batch")
    ap.add_argument("--anchor_min_pos", type=int, default=15,
                    help="refuse to run dynamic PL if any target has fewer "
                         "positive anchors")
    ap.add_argument("--lambda_a", type=float, default=0.3,
                    help="per-target prevalence prior-matching weight")
    ap.add_argument("--lambda_h", type=float, default=0.0,
                    help="per-target binary entropy weight (requires "
                         "lambda_a > 0)")
    ap.add_argument("--mixup_alpha", type=float, default=0.0,
                    help="BCE-compatible mixup Beta(a,a); 0 = off")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay of the PL harvesting teacher")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--faa_policy", default="",
                    help="faa_policy_v1.json: random sub-policy per study "
                         "applied to the 9-slice row before group selection")
    ap.add_argument("--eval_labels_csv", default="",
                    help="fixed reference labels for val scoring + best-epoch "
                         "selection (use labels_v06.csv); empty = training "
                         "labels")
    ap.add_argument("--crop_mm", type=float, default=160.0,
                    help="effective FOV; 130 center-crops the 160mm/512px "
                         "source to 416px before resize (I-027)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fold", default="0",
                    help="'all', a single fold '0'..'4', or a list '1,2,3,4'")
    # ---- P3 (diagnosis 2026-08-21): defaults reproduce the pre-P3 control
    # except the flip, which is OFF (it was a window reversal, see maybe_flip)
    ap.add_argument("--flip_p", type=float, default=0.0,
                    help="P(train-time L/R flip); 0 = off (default). The "
                         "legacy flip is geometrically wrong on sagittal "
                         "slots - keep 0 unless reproducing old runs (0.5)")
    ap.add_argument("--ema_eval", action="store_true",
                    help="evaluate + checkpoint the EMA shadow weights")
    ap.add_argument("--llrd", type=float, default=1.0,
                    help="layer-wise LR decay over backbone blocks (0.75 "
                         "typical); 1.0 = single backbone LR")
    ap.add_argument("--warmup_frac", type=float, default=0.0,
                    help="linear LR warmup fraction (0.05 typical)")
    ap.add_argument("--head_only_ep", type=int, default=0,
                    help="epochs with backbone LR held at 0 before warmup")
    return ap


def main():
    args = build_parser().parse_args()
    args.crop_px = (0 if args.crop_mm >= 160 else
                    int(round(512 * args.crop_mm / 160.0)))
    if args.cache_path is None:
        args.cache_path = f"/content/cache{args.img}c{args.crop_px}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    cd = Path(args.csv_dir)
    folds_df = pd.read_csv(cd / "folds.csv")
    labels = pd.read_csv(cd / "labels_v04.csv").set_index("StudyInstanceUID")
    slots_df = pd.read_csv(cd / "series_slots.csv")
    train_df = pd.read_csv(cd / "train.csv")

    studies = [s for s in folds_df["StudyInstanceUID"]
               if (Path(args.prep_dir) / f"{s}.npz").is_file()]
    log(f"{len(studies)}/{len(folds_df)} studies present in prep_dir")
    folds_map = dict(zip(folds_df["StudyInstanceUID"], folds_df["fold"]))
    slot_rows = {(r.StudyInstanceUID, r.slot): r.SeriesInstanceUID
                 for r in slots_df.itertuples()}
    # P3.1: keep NaN (unknown cells / missing studies) - run_fold masks them
    soft = labels.reindex(studies)[TARGETS].values.astype(np.float32)
    soft_eval = None
    if getattr(args, "eval_labels_csv", ""):
        ev = pd.read_csv(cd / args.eval_labels_csv).set_index(
            "StudyInstanceUID")
        soft_eval = ev.reindex(studies)[TARGETS].values.astype(np.float32)

    gold_df = train_df[train_df[TARGETS].notna().all(axis=1)]
    gold_map = gold_df.set_index("StudyInstanceUID")[TARGETS]
    gold_idx = np.array([i for i, s in enumerate(studies)
                         if s in gold_map.index])
    gold_y = gold_map.reindex([studies[i] for i in gold_idx]).values.astype(
        int)

    cache, mask = build_cache(studies, args.prep_dir, slot_rows, args.img,
                              args.workers, args.cache_path, args.crop_px)

    folds = (range(5) if args.fold == "all"
             else [int(x) for x in str(args.fold).split(",")])
    metrics_path = Path(args.out_dir) / "metrics.json"
    results = (json.load(open(metrics_path)) if metrics_path.is_file()
               else {})
    for f in folds:
        results[str(f)] = run_fold(f, studies, cache, mask, soft, gold_y,
                                   gold_idx, folds_map, args, dev,
                                   soft_eval=soft_eval)
        json.dump(results, open(metrics_path, "w"), indent=1)  # incremental
    log("SUMMARY: " + " | ".join(
        f"fold{f}: gold {r.get('gold_auc', 0):.3f} "
        f"val {r.get('val_auc', 0):.3f}"
        for f, r in sorted(results.items())))


if __name__ == "__main__":
    main()
