#!/bin/bash
# Stack smoke test ON THE BOX: the real wrapper + real trainer + real cache, on a study subset, 3 epochs
# (2 head-only + 1 full fine-tune, so backbone backward and VRAM are exercised). ~5-8 min per recipe.
#
# usage: bash smoke_xp.sh [RECIPE=a57|a57dino3|a57mix|a57elr|a57aux|a57anti|a57ttst|f0d|s4c|b8d|a4c][GPU=0] [LABELS=labels_v13D_r7.csv] [N_TRAIN=400] [N_VAL=150]
#
# How: a throw-away root /workspace/rsna-knee/smoke/ whose code/, cache336/, pretrained/, data/labels/,
# data/train.csv are symlinks to work/, and whose data/folds.csv keeps N_VAL fold-0 + N_TRAIN fold-1..4 studies
# (the wrapper trains/validates on folds.csv rows only; SliceBank still loads the full cache). Nothing in
# work/runs is touched. Expected in smoke/runs/<name>/train.log:
#   a57: "THIN-SERIES ON: ... rows 819058 -> 717833"  (proves the cache is complete and identical to A/C)
#   both: "FULL FINETUNE", "STUDY AFFINE AUG ON", "DOWN-UP RESAMPLE ON", "RECON AUG ON", "TrivialAugment-MRI ON"
#   f0d: "SOFT_GCE ON"
#   a57dino3: a57 lines; run_manifest family convnext_dino3, backbone timm:convnext_tiny, pretrained_sha256 a7637753...
#   a57mix: a57 lines + "MIXUP ON (XP-0563)" and per epoch "MIXUP ep k: n/m micro-batches mixed" (n/m ~ .4-.5)
#   a57elr: a57 lines + "ELR ON (XP-0565)" and per epoch "ELR ep k: mean term ... | buffer mean |t-.5| ..." (finite, < 0)
#   a57ttst: a57 lines + "TTST ON (XP-0589): pseudo_trio_f0.csv ... | mode mix | train rows 400 labelled -> 550 (150 val-fold ..."
#   a57aux: a57 lines + "AUX-REP ON (XP-0586): aux_v2.csv sha256 ... | 38 fields ..." and per epoch "| aux rep_bce .. rep_cov .."
#   s4c: no acq-aug lines (exp184 recipe has none); family convnext, trainer v9 (FiLM/tdec2/inst-head/ms-xattn)
#   b8d: acq-aug lines + "GROUP-MIX ON: 4 target-group study mixers"; no THIN-SERIES line (no thinning)
#   a4c: acq-aug lines + "HIER DEEP SUPERVISION: aux BCE on the hier head logits, weight 0.25"; family convnext
#   epoch lines "fold 0 ep k: ... | data-wait ~0s"; then result.json status complete + oof_pred.npz [N_VAL, 12].
# Timing extrapolation: full epoch ~= smoke ep-2 minutes x 3526 / N_TRAIN (old boxes: 7.6 min/epoch, 40 ep = 5.1 h).
set -euo pipefail
RECIPE=${1:-a57}; GPU=${2:-0}; LABELS=${3:-labels_v13D_r7.csv}; NTR=${4:-400}; NVA=${5:-150}
W=/workspace/rsna-knee/work; S=/workspace/rsna-knee/smoke; PY=/workspace/rsna-knee/venv/bin/python
mkdir -p "$S/data" "$S/runs"
for d in code cache336 pretrained; do ln -sfn "$W/$d" "$S/$d"; done
ln -sfn "$W/data/labels" "$S/data/labels"; ln -sfn "$W/data/train.csv" "$S/data/train.csv"
"$PY" - "$W/data/folds.csv" "$S/data/folds.csv" "$NTR" "$NVA" <<'EOF'
import sys, pandas as pd
f = pd.read_csv(sys.argv[1]); ntr, nva = int(sys.argv[3]), int(sys.argv[4])
va = f[f.fold == 0].sample(nva, random_state=0)
tr = f[f.fold != 0].sample(ntr, random_state=0)
pd.concat([va, tr]).to_csv(sys.argv[2], index=False)
print("smoke folds:", len(va), "val /", len(tr), "train")
EOF
ACQ='"decoder": "xattn", "wd": 0.05, "saff_p": 0.5, "saff_rot": 10.0, "saff_scale": 0.12, "saff_shift": 0.08, "dres_p": 0.5, "dres_min": 0.6, "recon_p": 0.5, "recon_noise": 0.035, "recon_sharp": 1.0, "recon_smooth": 0.6'
FAM=convnextv2; NW=12; WRAP=train_oof224.py
case "$RECIPE" in
  a57) TRAINER=airaa57ba94073;  OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32}" ;;
  a57mix) TRAINER=a57mix;     OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"mixup_alpha\": 0.2, \"mixup_p\": 0.5}" ;;
  a57elr) TRAINER=a57elr;     OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"elr_lambda\": 3.0, \"elr_beta\": 0.7}" ;;
  a57aux) TRAINER=a57aux;     OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"aux_csv\": \"aux_v2.csv\", \"aux_lambda\": 0.1}" ;;
  # XP-0587: cold a57 + the anti term (the run recipe a57antiw is warm; this exercises the CUDA/autocast path)
  a57anti) TRAINER=a57anti;   OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"anti_csv\": \"anti_teacher_v14.csv\", \"anti_gamma\": 0.1}" ;;
  # XP-0589: cold a57 + the pseudo val fold (mix; the run recipe a57ttst is warm); the smoke folds.csv holds 150 of
  # the 881 fold-0 studies, the CSV's other fold-0 rows are ignored (not in this run)
  a57ttst) TRAINER=a57ttst;   OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"pseudo_csv\": \"pseudo_trio_f0.csv\", \"pseudo_mode\": \"mix\"}" ;;
  a57dino3) TRAINER=airaa57ba94073; FAM=convnext_dino3; WRAP=train_oof224_dino3.py
       OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"lr_backbone\": ${BB_LR:-1e-4}, \"lr_head\": ${HEAD_LR:-3e-4}}" ;;
  f0d) TRAINER=mlef0d1753e037f; OVR="{$ACQ, \"soft_gce\": 0.2, \"gce_q\": 0.7}" ;;
  s4c) TRAINER=v9; FAM=convnext; NW=10; OVR='{"decoder": "xattn", "film": 1, "tdec": 2, "inst_head": 0.25, "ms_xattn": 1}' ;;
  b8d) TRAINER=aira8bd0d5e41f; OVR="{$ACQ, \"group_mix\": 1}" ;;
  a4c) TRAINER=aira4a0c41e2eb; FAM=convnext; OVR='{"decoder": "xattn", "dres_min": 0.6, "dres_p": 0.5, "film": 1, "hier": 1, "hier_aux": 0.25, "inst_head": 0.25, "ms_xattn": 1, "recon_noise": 0.035, "recon_p": 0.5, "recon_sharp": 1.0, "recon_smooth": 0.6, "saff_p": 0.5, "saff_rot": 10.0, "saff_scale": 0.12, "saff_shift": 0.08, "tdec": 2, "wd": 0.1}' ;;
  *) echo "unknown recipe $RECIPE (a57|a57dino3|a57mix|a57elr|a57aux|a57anti|a57ttst|f0d|s4c|b8d|a4c)"; exit 1 ;;
esac
NAME=smoke_${RECIPE}_$(date +%H%M%S)
cd "$S"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" -u "$W/code/$WRAP" --root "$S" --run-name "$NAME" \
  --fold 0 --seed 1000 --family "$FAM" --labels "$LABELS" --epochs 3 --cache cache336 --n-parts 5 \
  --batch 2 --accum 2 --workers "$NW" --overrides "$OVR" --trainer "$TRAINER" 2>&1 | tee "$S/runs/$NAME.log"
echo "wall $(( $(date +%s) - T0 )) s"
R=$S/runs/$NAME
grep -E "THIN-SERIES|FULL FINETUNE|AUG ON|RESAMPLE ON|SOFT_GCE|MIXUP|ELR|AUX-REP|ANTI-ENSEMBLE|TTST ON|GROUP-MIX|HIER DEEP|TrivialAugment|ep [0-9]+:" "$R/train.log" "$S/runs/$NAME.log" 2>/dev/null | sort -u | head -20 || true
"$PY" - "$R" "$NVA" <<'EOF'
import json, sys, numpy as np
r = json.load(open(sys.argv[1] + "/result.json"))
z = np.load(sys.argv[1] + "/oof_pred.npz")
assert r["status"] == "complete" and z["pred"].shape == (int(sys.argv[2]), 12) and np.isfinite(z["pred"]).all()
print("SMOKE OK", {k: r[k] for k in ("selected_epoch", "oof_val_auc_report", "seconds", "max_mem_gb")})
EOF
