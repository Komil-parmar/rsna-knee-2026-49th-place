"""K-fold OOF trainer wrapper (campaign v2, 2026-09-23).

Validation fold k = every study with folds.csv fold == k, gold-58 included as ordinary
rows (their REPORT labels train/select nothing outside their own fold). Training = all
other folds. After training, the selected checkpoint predicts fold k whole-study (cap256)
-> an out-of-fold bank. Gold-58 AUC is computed post hoc from those OOF rows only; it
never selects anything. Selection = EMA four-bag macro AUC vs --eval-labels (report v06).

Crash-safe: rerunning the identical command resumes from <out>/state_last.pt.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FAMILIES = {
    "convnext": {"backbone": "timm:convnext_tiny.fb_in22k_ft_in1k", "unfreeze_last": 18},
    "coatnet": {"backbone": "timm:coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k", "unfreeze_last": 24},
    # campaign v2 trainer (block adapter); unfreeze_last is moot under full_finetune
    "effv2s": {"backbone": "timm:tf_efficientnetv2_s.in21k_ft_in1k", "unfreeze_last": 8, "trainer": "v2"},
    "resnet34": {"backbone": "timm:resnet34.a1_in1k", "unfreeze_last": 8, "trainer": "v2"},
    "convnext_s": {"backbone": "timm:convnext_small.fb_in22k_ft_in1k", "unfreeze_last": 36},
    "convnextv2": {"backbone": "timm:convnextv2_tiny.fcmae_ft_in22k_in1k", "unfreeze_last": 18},
    "convnextv2b": {"backbone": "timm:convnextv2_base.fcmae_ft_in22k_in1k", "unfreeze_last": 36},   # harness CH2 (scale)
    # HF DINOv2 (production DINO recipe comes via --overrides); weights from the offline HF cache
    # exp-154: SigLIP (web image-text) ViTs — the Qwen3.5-VL tower class; trainer v7 wrapper
    # exp-176: new 336-cache families for ensemble diversity
    "convnext384": {"backbone": "timm:convnext_tiny.in12k_ft_in1k_384", "unfreeze_last": 18},
    "effv2m": {"backbone": "timm:tf_efficientnetv2_m.in21k_ft_in1k", "unfreeze_last": 8, "trainer": "v2"},
    "coatnet384": {"backbone": "timm:coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "unfreeze_last": 24},
    "siglipb": {"backbone": "timm:vit_base_patch16_siglip_gap_384.webli", "unfreeze_last": 8, "trainer": "v7"},
    "siglipl": {"backbone": "timm:vit_large_patch16_siglip_gap_384.webli", "unfreeze_last": 16, "trainer": "v7"},
    "dinov2s": {"backbone": "facebook/dinov2-small", "unfreeze_last": 8,
                "pre": "/workspace/rsna-knee/hf/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors"},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def resolved_args(trainer_path, a):
    tree = ast.parse(Path(trainer_path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_parser")
    scope = {"argparse": argparse}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(trainer_path), "exec"), scope)
    args = scope["build_parser"]().parse_args([])
    fam = FAMILIES[a.family]
    fixed = {"backbone": fam["backbone"], "unfreeze_last": fam["unfreeze_last"],
             "full_finetune": True, "grad_checkpoint": False, "fold": a.fold,
             "labels_csv": Path(a.labels).name, "eval_labels_csv": Path(a.eval_labels).name,
             "seed": a.seed, "epochs": a.epochs, "n_parts": a.n_parts, "d_model": 384, "n_mix": 2,
             "bag": 32, "eval_bags": 4, "batch": a.batch, "accum": a.accum, "eval_batch": 2,
             "workers": a.workers, "lr_backbone": 1e-4, "lr_head": 3e-4, "wd": .02, "ema": .999,
             "llrd": 1.0, "head_only_ep": 2, "conf_weight": True, "trivial_aug": True, "ta_p_bag": .25,
             "flip_p": 0.0, "sched": "cosine_hold", "cosine_epochs": 40, "cosine_floor": .015,
             "warmup_frac": .05, "min_ep": a.epochs, "patience": 0, "select_on": "val",
             "plateau_metric": "val", "full_data": False, "gold2_csv": "", "holdout_csv": "",
             "weight_csv": "", "init_backbone": "", "init_full": "", "resume_state": "auto",
             "save_state": 1, "keep_state_every": 0, "topk": 0, "whole_study": False, "tri_gap": 1,
             "xcontrast_p": 0.0, "zoom_px": 0, "hard_labels": False, "aux_geo": False,
             "aux_txt": "", "aux_sev": "", "feat_dir": ""}
    fixed.update(json.loads(a.overrides or "{}"))   # registered per-experiment deltas
    for k, v in fixed.items():
        assert hasattr(args, k), f"unknown trainer option {k}"
        setattr(args, k, v)
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)          # has code/, data/, cache224/, pretrained/
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--labels", required=True)                    # training targets CSV
    p.add_argument("--eval-labels", default="labels_v06.csv")   # selector targets (report)
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--overrides", default="")
    p.add_argument("--cache", default="cache224")               # or cache336 (130 mm, 5 parts)
    p.add_argument("--n-parts", type=int, default=3)
    p.add_argument("--trainer", default="", help="override trainer version (v1/v2/v3)")
    p.add_argument("--exclude-folds", default="", help="comma list of folds dropped from train AND val (sealed outer fold)")
    p.add_argument("--plane", choices=["", "sag", "cor", "ax"], default="",
                   help="plane specialist: bags/inference use only this plane's slices (fallback: all)")
    p.add_argument("--full-data", action="store_true",
                   help="exp-149: train on EVERY study (val fold only monitored); ckpt = final epoch (trainer vNL)")
    p.add_argument("--exclude-gold", action="store_true",
                   help="drop gold-58 from train AND val; score them with the selected ckpt at the end")
    a = p.parse_args()

    import pandas as pd
    root = a.root.resolve()
    code, data = root / "code", root / "data"
    out = root / "runs" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        trainer_path = code / ("train_slicebag_campaign_%s.py" % ((a.trainer or FAMILIES[a.family].get("trainer", "v1"))
                                                                  + ("L" if a.full_data else "")))
        folds = pd.read_csv(data / "folds.csv").set_index("StudyInstanceUID")
        ids = folds.index.to_numpy(dtype=str)
        lab = pd.read_csv(data / "labels" / a.labels).set_index("StudyInstanceUID").reindex(ids)[TARGETS]
        ev = pd.read_csv(data / "labels" / a.eval_labels).set_index("StudyInstanceUID").reindex(ids)[TARGETS]
        soft = lab.to_numpy(np.float32)
        soft_eval = ev.to_numpy(np.float32)
        assert np.isfinite(soft).mean() > .9 and np.nanmin(soft) >= 0 and np.nanmax(soft) <= 1, "bad train labels"
        resolved = resolved_args(trainer_path, a)
        resolved.cache_dir, resolved.csv_dir, resolved.out_dir = str(root / a.cache), str(data), str(out)
        if a.full_data:                               # exp-149: all studies train, final epoch kept
            assert not (a.exclude_gold or a.exclude_folds or a.plane)
            resolved.full_data, resolved.select_on = True, "last"
        pre = Path(FAMILIES[a.family].get("pre", root / "pretrained" / a.family / "model.safetensors"))
        os.environ.setdefault("HF_HOME", "/workspace/rsna-knee/hf")
        identity = {"protocol": "oof5_v1" + ("_xgold" if a.exclude_gold else "") + (f"_xfolds{a.exclude_folds}" if a.exclude_folds else "") + (f"_plane_{a.plane}" if a.plane else "") + ("_full" if a.full_data else ""), "family": a.family, "fold": a.fold, "seed": a.seed,
                    "labels_sha256": sha256(data / "labels" / a.labels),
                    "eval_labels_sha256": sha256(data / "labels" / a.eval_labels),
                    "folds_sha256": sha256(data / "folds.csv"), "trainer_sha256": sha256(trainer_path),
                    "pretrained_sha256": sha256(pre),
                    "config": {k: v for k, v in vars(resolved).items()
                               if k not in ("out_dir", "csv_dir", "cache_dir", "resume_state", "data_fingerprint")}}
        fp = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        resolved.data_fingerprint = fp
        man = out / "run_manifest.json"
        if man.exists():
            assert json.loads(man.read_text())["fingerprint"] == fp, "identity changed; refusing to resume"
        else:
            atomic_json(man, {"fingerprint": fp, "identity": identity, "started_unix": started,
                              "wrapper_sha256_at_start": sha256(__file__)})
        os.environ.update(TIMM_NO_PRETRAIN="0", RSNA_CAMPAIGN_PRETRAINED_FILE=str(pre),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        sys.path.insert(0, str(code))
        spec = importlib.util.spec_from_file_location("campaign_trainer_v1", trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = trainer
        spec.loader.exec_module(trainer)
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible GPU"
        torch.set_num_threads(2)
        random.seed(a.seed)
        np.random.seed(a.seed)
        bank = trainer.SliceBank(resolved.cache_dir, n_parts=a.n_parts, drop_zero=True)
        official_all = pd.read_csv(data / "train.csv").set_index("StudyInstanceUID").reindex(ids)[TARGETS]
        is_gold = official_all.notna().all(axis=1).to_numpy()
        keep = ~is_gold if a.exclude_gold else np.ones(len(ids), bool)
        if a.exclude_folds:                            # sealed outer fold(s): invisible to this run
            xf = [int(x) for x in a.exclude_folds.split(",")]
            assert a.fold not in xf
            keep &= ~folds["fold"].isin(xf).to_numpy()
        studies = ids[keep].tolist()
        if a.plane:                               # plane specialist (tab col 2 = plane_id)
            pid = {"sag": 0, "cor": 1, "ax": 2}[a.plane]
            n_fb = 0
            for s_, tab in bank.by_study.items():
                sel = tab[:, 2] == pid
                if sel.sum() >= 3:
                    bank.by_study[s_] = tab[sel]
                else:
                    n_fb += 1
            print(f"PLANE {a.plane}: {n_fb} studies fall back to all planes", flush=True)
        soft, soft_eval = soft[keep], soft_eval[keep]
        assert all(s in bank.by_study for s in studies), "incomplete bank"
        fmap = {s: int(folds.loc[s, "fold"]) for s in studies}
        result = trainer.run_fold(a.fold, studies, bank, soft, soft_eval, np.empty((0, 12), np.int64),
                                  np.empty(0, np.int64), fmap, resolved, torch.device("cuda"))
        ck_path = out / f"ckpt_fold{a.fold}.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        _kw = {"hier": int(getattr(resolved, "hier", 0))} if getattr(resolved, "hier", 0) else {}
        if getattr(resolved, "vol3d", 0):                  # exp-161: trainer v8 volume branch
            _kw["vol3d"] = int(resolved.vol3d)
        for _k in ("ms_xattn", "tdec", "film", "gated_mil"):   # exp-179 v9 / exp-191 v11 options
            if getattr(resolved, _k, 0):
                _kw[_k] = int(getattr(resolved, _k))
        if getattr(resolved, "inst_head", 0):
            _kw["inst_head"] = float(resolved.inst_head)
        model = trainer.SliceBagModel(resolved.backbone, unfreeze_last=0, d_model=resolved.d_model,
                                      n_mix=resolved.n_mix, pool=resolved.pool,
                                      quad_pool=int(getattr(resolved, "quad_pool", 0)),
                                      use_xattn=getattr(resolved, "decoder", "mil") == "xattn",
                                      **_kw).to("cuda").eval()
        model.load_state_dict(ck["state_dict"], strict=True)
        model.zoom_px = int(getattr(resolved, "zoom_px", 0) or 0)   # exp-152: forward-time centre zoom
        va_ids = ids[(folds["fold"].to_numpy() == a.fold) & keep]
        pred = trainer.predict_full(model, bank, va_ids, torch.device("cuda"), max_slices=256, workers=a.workers)
        assert pred.shape == (len(va_ids), 12) and np.isfinite(pred).all()
        atomic_npz(out / "oof_pred.npz", ids=va_ids, pred=pred.astype(np.float32))
        y, valid = trainer.eval_targets(ev.reindex(va_ids).to_numpy(np.float32))
        val_auc, val_per = trainer.macro_auc(y, pred, valid)
        official = pd.read_csv(data / "train.csv").set_index("StudyInstanceUID").reindex(va_ids)[TARGETS]
        gmask = official.notna().all(axis=1).to_numpy()
        g_auc, g_per = (trainer.macro_auc(official.to_numpy()[gmask].astype(int), pred[gmask])
                        if gmask.sum() >= 5 else (float("nan"), []))
        if a.exclude_gold:
            g_ids = ids[is_gold]
            gp = trainer.predict_full(model, bank, g_ids, torch.device("cuda"), max_slices=256, workers=a.workers)
            atomic_npz(out / "gold_pred.npz", ids=g_ids, pred=gp.astype(np.float32))
            g_auc, g_per = trainer.macro_auc(official_all.to_numpy()[is_gold].astype(int), gp)
            gmask = np.zeros(len(va_ids), bool)
        res = {"status": "complete", "run": a.run_name, "full_data": bool(a.full_data), "family": a.family, "fold": a.fold,
               "seed": a.seed, "labels": a.labels, "selected_epoch": int(ck["metrics"]["epoch"]),
               "ckpt_sha256": sha256(ck_path), "oof_val_auc_report": val_auc,
               "oof_val_per_target": dict(zip(TARGETS, val_per)), "n_gold_scored": 58 if a.exclude_gold else int(gmask.sum()),
               "gold_auc": g_auc, "gold_per_target": dict(zip(TARGETS, g_per)) if len(g_per) else {}, "seconds": time.time() - started,
               "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        atomic_json(out / "result.json", res)
        print(json.dumps({k: res[k] for k in ("run", "selected_epoch", "oof_val_auc_report",
                                              "gold_auc", "seconds")}), flush=True)
    except BaseException as exc:
        atomic_json(out / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                           "seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
