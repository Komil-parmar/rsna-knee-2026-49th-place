#!/bin/bash
# Launch ONE training run exactly as the old dispatcher_v2 did (same python, same wrapper, same args), detached.
#
# usage (ON THE BOX):
#   bash run_xp.sh RECIPE FOLD SEED LABELS_CSV GPU [RUN_NAME] [EXTRA_WRAPPER_ARGS...]
#     RECIPE      a57  = AIRA r02 node a57ba94073024a87 (acq-aug + thin >60-slice series), best single family
#                 a57mix = XP-0563 MIX-LOW: a57 + study-level mixup (trainer a57mix = patches/a57mix_xp0563.py
#                        on the a57 trainer; overrides add "mixup_alpha": 0.2, "mixup_p": 0.5, i.e. the trainer's
#                        --mixup-alpha 0.2 --mixup-p 0.5; the wrapper only forwards trainer options via --overrides)
#                 a57aux = XP-0586 REPORT-AUX: a57 + training-only linear head on the per-target decoder context ->
#                        38 report fields (data/labels/aux_v2.csv, value/max soft BCE, -1 masked), aux_lambda .1
#                        (trainer a57aux = patches/a57aux_xp0586.py); a57auxw = the same, warm (a57w recipe, INIT=)
#                 a57antiw = XP-0587 ANTI-ENSEMBLE, warm (a57w recipe, INIT=): + training-only penalty
#                        anti_gamma * mean_j corr(z_j, teacher_j)^2 over a 256-pair window vs the v14 trio OOF logits
#                        (data/labels/anti_teacher_v14.csv; trainer a57anti = patches/a57anti_xp0587.py);
#                        gamma from env ANTI_G (default 0.1), NODE a57ba94073antiw<G*100, 3 digits> e.g. antiw010
#                 a57ttst = XP-0589 TTST SIMULATION, warm (INIT= the SAME-FOLD a57 ckpt, i.e. trained WITHOUT fold FOLD:
#                        runs/a3_a57ba94073_v14_s1000_f<k>/ckpt_fold<k>.pt; basename must be ckpt_fold<FOLD>.pt):
#                        4 ep (cosine 4, min_ep 4, head_only 0), lr_backbone 3e-5 / lr_head 1e-4, + the VAL-fold studies
#                        trained with OOF-bank predictions as soft targets (data/labels/pseudo_<PSRC>_f<k>.csv; trainer
#                        a57ttst = patches/a57ttst_xp0589.py). Selection/OOF still vs labels_v06. Env:
#                        PSRC (default trio): trio = mean of XP-0573 + XP-0573_s3000 + XP-0576 + XP-0579 (full v14 ens.)
#                                             self = XP-0573 only (a57-v14 s1000 = the INIT model's own bank; self-PL)
#                                             cot  = XP-0576 + XP-0579 (4a0c + 8bd0, no a57; co-training PL)
#                                             none = REPLAY CONTROL: same warm short schedule, train folds only, no
#                                                    pseudo rows (pseudo_csv unset -> exactly the a57 trainer path)
#                        PMODE (default mix): mix = train folds (LABELS) + pseudo val fold; val_only = pseudo val fold
#                                             only (PSRC=none requires mix)
#                        NODE a57ba94073ttst<PSRC><mix|val>, e.g. ttsttriomix, ttstselfval, ttstcotmix; control ttstnone
#                 a57dino3 = a57 exactly (trainer airaa57ba94073, acq-aug, thin 60:32, xattn) but backbone family
#                        convnext_dino3 = timm convnext_tiny initialised from DINOv3 LVD-1689M weights
#                        (pretrained/convnext_dino3/model.safetensors, sha a7637753..., shipped in the payload; DINOv3
#                        licence: keep LICENSE_DINOv3.md with any released weights), launched through the
#                        insert-only wrapper copy code/train_oof224_dino3.py (adds that one FAMILIES entry).
#                        LRs from env BB_LR (default 1e-4) / HEAD_LR (default 3e-4), e-notation only (e.g. 5e-5);
#                        NODE a57ba94073d3b<BB>h<HEAD>, digits only: 1e-4 -> 1e4, 2.5e-4 -> 25e4, e.g. d3b1e4h3e4
#                 a57elr = XP-0565 ELR: a57 + early-learning regularisation (trainer a57elr = patches/a57elr_xp0565.py
#                        on the a57 trainer; overrides add "elr_lambda": 3.0, "elr_beta": 0.7)
#                 f0d  = MLEvolve node f0d1753e037fcd27 (acq-aug "recipe C" + SOFT_GCE/density-power loss, NO thinning)
#                 s4c  = exp184 stack4 ConvNeXt-T 336 (family convnext, trainer v9, FiLM+tdec2+inst .25+ms_xattn, workers 10)
#                        = the S60/S62 efficiency-kernel family (XP-0007 DISTILL-EFF student), ~2.9 h/fold on old box A
#                 b8d  = AIRA r02 node 8bd0d5e41f1843f7 (S78 member): acq-aug + 4 target-group study mixers (group_mix 1),
#                        ConvNeXtV2-T, trainer aira8bd0d5e41f = patch(v1); no thinning. ~5.5 h/fold old box A, ~5 h 5090
#                 a4c  = AIRA r02 node 4a0c41e2eb4b4ff7 (S78 member): v9 stack4 ConvNeXt-T + hier head (hier 1, hier_aux .25)
#                        + acq-aug, wd .1, trainer aira4a0c41e2eb = patch(v9). ~3.8 h/fold old box C, ~3.5 h 5090
#     FOLD        0..4 (validation fold of data/folds.csv; training = the other 4 folds)
#     SEED        1000 / 2000 / 3000 ...
#     LABELS_CSV  TRAIN targets, a file name inside work/data/labels/ (e.g. labels_v11_exp108_r7.csv,
#                 labels_v13D_r7.csv). Epoch SELECTION always uses --eval-labels labels_v06.csv (wrapper default,
#                 never changed here), so swapping this file changes only what the model is trained on.
#     GPU         physical GPU index (becomes CUDA_VISIBLE_DEVICES; the wrapper asserts exactly one visible GPU)
#     RUN_NAME    default a3_<node>_<labeltag>_s<seed>_f<fold>, e.g. a3_a57ba94073_v13D_s1000_f0, a3_f0d1753e_v11_s2000_f0
#                 ("a57ba94073" in the name makes the delivery kernel REFUSE to serve it without a thin rule)
# examples:
#   bash run_xp.sh a57 0 1000 labels_v13D_r7.csv 0            # XP-0004 fold 0
#   bash run_xp.sh f0d 0 2000 labels_v11_exp108_r7.csv 5      # XP-0005 seed 2000 (exact pending job of 2026-10-02)
#   bash run_xp.sh a57mix 0 1000 labels_v11_exp108_r7.csv 1  # XP-0563 fold 0 -> a3_a57ba94073mix_v11_s1000_f0
#   INIT=$W/runs/a3_a57ba94073_v14_s1000_f0/ckpt_fold0.pt PSRC=trio PMODE=mix bash run_xp.sh a57ttst 0 1000 labels_v14_r7.csv 0
#                                                               # XP-0589 fold 0 -> a3_a57ba94073ttsttriomix_v14_s1000_f0
#   INIT=... PSRC=none bash run_xp.sh a57ttst 0 1000 labels_v14_r7.csv 1   # replay control -> a3_a57ba94073ttstnone_v14_s1000_f0
#   bash run_xp.sh a57elr 0 1000 labels_v11_exp108_r7.csv 2  # XP-0565 fold 0 -> a3_a57ba94073elr_v11_s1000_f0
#   BB_LR=5e-5 bash run_xp.sh a57dino3 0 1000 labels_v14_r7.csv 3   # -> a3_a57ba94073d3b5e5h3e4_v14_s1000_f0
#   bash run_xp.sh s4c 0 1000 labels_s78T_r9.csv 7            # XP-0007 fold 0 (S62 needs folds 0-2)
#   bash run_xp.sh b8d 0 1000 labels_v13D_r7.csv 3            # -> a3_8bd0d5e41f_v13D_s1000_f0
#   bash run_xp.sh a4c 0 1000 labels_v13D_r7.csv 4            # -> a3_4a0c41e2eb_v13D_s1000_f0
#
# Output: work/runs/<RUN_NAME>/ {train.log, run_manifest.json, history_f<k>.json, val_preds_f<k>_ep*.npy,
#   state_last.pt (resume), ckpt_fold<k>.pt (selected EMA epoch), oof_pred.npz (ids, pred: whole-study cap-256
#   preds of the val fold), result.json (oof_val_auc_report vs v06, per target, gold_auc, selected_epoch, seconds)}
#   + launch.json / pid written by this script.
# Resume: re-run the IDENTICAL command; the wrapper resumes from state_last.pt (refuses if any identity field changed).
set -euo pipefail
[ $# -ge 5 ] || { sed -n 2,70p "$0"; exit 1; }
RECIPE=$1; FOLD=$2; SEED=$3; LABELS=$4; GPU=$5; shift 5
NAME=${1:-}; [ $# -ge 1 ] && shift
W=/workspace/rsna-knee/work
PY=/workspace/rsna-knee/venv/bin/python

ACQ='"decoder": "xattn", "wd": 0.05, "saff_p": 0.5, "saff_rot": 10.0, "saff_scale": 0.12, "saff_shift": 0.08, "dres_p": 0.5, "dres_min": 0.6, "recon_p": 0.5, "recon_noise": 0.035, "recon_sharp": 1.0, "recon_smooth": 0.6'
FAM=convnextv2; NW=12
ANTI_G=${ANTI_G:-0.1}      # XP-0587 a57antiw gamma; ANTI_GT = 100*gamma, 3 digits (0.1 -> 010) for the NODE name
[[ "$ANTI_G" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "ANTI_G must be a non-negative number"; exit 1; }
BB_LR=${BB_LR:-1e-4}; HEAD_LR=${HEAD_LR:-3e-4}   # a57dino3 learning rates (wrapper defaults 1e-4 / 3e-4)
for v in "$BB_LR" "$HEAD_LR"; do [[ "$v" =~ ^[0-9]+(\.[0-9]+)?e-[0-9]+$ ]] || { echo "BB_LR/HEAD_LR must look like 1e-4 / 2.5e-4"; exit 1; }; done
LR_T=b${BB_LR//[.-]/}h${HEAD_LR//[.-]/}   # 1e-4,3e-4 -> b1e4h3e4
WRAP=train_oof224.py       # a57dino3 swaps in the insert-only copy train_oof224_dino3.py
PMODE=${PMODE:-mix}        # XP-0589 a57ttst: mix | val_only
[[ "$PMODE" =~ ^(mix|val_only)$ ]] || { echo "PMODE must be mix or val_only"; exit 1; }
PMODE_T=${PMODE%%_*}       # mix -> mix, val_only -> val (NODE name)
PSRC=${PSRC:-trio}         # XP-0589 a57ttst pseudo source: trio | self | cot | none (replay control)
[[ "$PSRC" =~ ^(trio|self|cot|none)$ ]] || { echo "PSRC must be trio, self, cot or none"; exit 1; }
if [ "$PSRC" = none ]; then
  [ "$PMODE" = mix ] || { echo "PSRC=none (replay control, no pseudo rows) needs PMODE=mix"; exit 1; }
  PS_OVR=""; PS_TAG=none
else
  PS_OVR=", \"pseudo_csv\": \"pseudo_${PSRC}_f$FOLD.csv\", \"pseudo_mode\": \"$PMODE\""; PS_TAG=$PSRC$PMODE_T
fi
ANTI_GT=$(printf '%03d' "$(awk -v g="$ANTI_G" 'BEGIN{printf "%d", g*100 + 0.5}')")
case "$RECIPE" in
  a57) TRAINER=a57;  OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32}" ;;
  # XP-0564 CONF-OFF: a57 exactly, with the wrapper's default conf_weight (0.3 + 0.7|2p-1| per cell) turned off
  a57nc) TRAINER=a57; OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"conf_weight\": false}" ;;
  # warm 16-ep variants (XP-0582 design; INIT=<abs path of a prev-round same-fold ckpt> required)
  a57w) TRAINER=a57; OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"epochs\": 16, \"cosine_epochs\": 16, \"min_ep\": 16, \"head_only_ep\": 0, \"init_full\": \"$INIT\"}" ;;
  a57ncw) TRAINER=a57; OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"epochs\": 16, \"cosine_epochs\": 16, \"min_ep\": 16, \"head_only_ep\": 0, \"init_full\": \"$INIT\", \"conf_weight\": false}" ;;
  # XP-0586 REPORT-AUX (patches/a57aux_xp0586.py): + linear head on ctx -> 38 report fields of data/labels/aux_v2.csv
  a57aux) TRAINER=a57aux;     OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"aux_csv\": \"aux_v2.csv\", \"aux_lambda\": 0.1}" ;;
  a57auxw) TRAINER=a57aux;    OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"epochs\": 16, \"cosine_epochs\": 16, \"min_ep\": 16, \"head_only_ep\": 0, \"init_full\": \"$INIT\", \"aux_csv\": \"aux_v2.csv\", \"aux_lambda\": 0.1}" ;;
  # XP-0587 ANTI-ENSEMBLE (patches/a57anti_xp0587.py): warm a57w + decorrelation from the v14 trio teacher, gamma ANTI_G
  a57antiw) TRAINER=a57anti;  OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"epochs\": 16, \"cosine_epochs\": 16, \"min_ep\": 16, \"head_only_ep\": 0, \"init_full\": \"$INIT\", \"anti_csv\": \"anti_teacher_v14.csv\", \"anti_gamma\": $ANTI_G}" ;;
  # XP-0589 TTST (patches/a57ttst_xp0589.py): warm a57 (same-fold INIT), 4 ep at 0.3x LR, + val fold with ensemble pseudo targets
  a57ttst) TRAINER=a57ttst;   OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"epochs\": 4, \"cosine_epochs\": 4, \"min_ep\": 4, \"head_only_ep\": 0, \"lr_backbone\": 3e-05, \"lr_head\": 0.0001, \"init_full\": \"${INIT:-}\"$PS_OVR}" ;;
  a57mix) TRAINER=a57mix;     OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"mixup_alpha\": 0.2, \"mixup_p\": 0.5}" ;;
  # XP-0565 ELR: a57 + early-learning regularisation (patches/a57elr_xp0565.py), lambda 3, beta .7 (Liu et al. 2020 CIFAR-10)
  a57elr) TRAINER=a57elr;     OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"elr_lambda\": 3.0, \"elr_beta\": 0.7}" ;;
  # XP-0565 dose arm: same, lambda 1 (ELR term ~3x smaller relative to BCE)
  a57elr1) TRAINER=a57elr;    OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"elr_lambda\": 1.0, \"elr_beta\": 0.7}" ;;
  # a57dino3: a57 with the DINOv3 ConvNeXt-T init (family convnext_dino3, wrapper copy); LRs from BB_LR / HEAD_LR
  a57dino3) TRAINER=a57; FAM=convnext_dino3; WRAP=train_oof224_dino3.py
       OVR="{$ACQ, \"thin_thr\": 60, \"thin_target\": 32, \"lr_backbone\": $BB_LR, \"lr_head\": $HEAD_LR}" ;;
  f0d) TRAINER=f0d; OVR="{$ACQ, \"soft_gce\": 0.2, \"gce_q\": 0.7}" ;;
  # exp184 args verbatim (dispatch/done/p05_0*_exp184_stack4_cnx336_f*.json); NO acq-aug package, wd stays .02
  s4c) TRAINER=v9; FAM=convnext; NW=10
       OVR='{"decoder": "xattn", "film": 1, "tdec": 2, "inst_head": 0.25, "ms_xattn": 1}' ;;
  # S78 family: AIRA r02 node 8bd0d5e41f1843f7 = SPEC overrides verbatim (acq-aug + 4 target-group mixers, NO thinning)
  b8d) TRAINER=8bd0; OVR="{$ACQ, \"group_mix\": 1}" ;;
  # S78 family: AIRA r02 node 4a0c41e2eb4b4ff7 = SPEC overrides verbatim (v9 stack4 cnx-T + hier head, hier_aux .25,
  # acq-aug, wd .1); workers 12 like every AIRA job (exp184/s4c used 10)
  a4c) TRAINER=4a0c; FAM=convnext; OVR='{"decoder": "xattn", "dres_min": 0.6, "dres_p": 0.5, "film": 1, "hier": 1, "hier_aux": 0.25, "inst_head": 0.25, "ms_xattn": 1, "recon_noise": 0.035, "recon_p": 0.5, "recon_sharp": 1.0, "recon_smooth": 0.6, "saff_p": 0.5, "saff_rot": 10.0, "saff_scale": 0.12, "saff_shift": 0.08, "tdec": 2, "wd": 0.1}' ;;
  *) echo "unknown recipe $RECIPE (a57|a57nc|a57w|a57ncw|a57aux|a57auxw|a57antiw|a57ttst|a57mix|a57elr|a57elr1|a57dino3|f0d|s4c|b8d|a4c)"; exit 1 ;;
esac
[ -f "$W/code/train_slicebag_campaign_$TRAINER.py" ] || { echo "trainer file missing"; exit 1; }
[ -f "$W/code/$WRAP" ] || { echo "wrapper $W/code/$WRAP missing (rebuild the payload)"; exit 1; }
case "$RECIPE" in a57dino3) echo "a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f  $W/pretrained/convnext_dino3/model.safetensors"   | sha256sum -c --quiet - || { echo "pretrained/convnext_dino3/model.safetensors missing or not the DINOv3 file"; exit 1; };; esac
[ -f "$W/data/labels/$LABELS" ] || { echo "labels $W/data/labels/$LABELS missing"; exit 1; }
case "$RECIPE" in a57antiw) [ -f "$W/data/labels/anti_teacher_v14.csv" ] || { echo "anti csv $W/data/labels/anti_teacher_v14.csv missing"; exit 1; };; esac
case "$RECIPE" in a57ttst) [ "$PSRC" = none ] || [ -f "$W/data/labels/pseudo_${PSRC}_f$FOLD.csv" ] || { echo "pseudo csv $W/data/labels/pseudo_${PSRC}_f$FOLD.csv missing"; exit 1; }
  [ -f "${INIT:-}" ] || { echo "a57ttst needs INIT=<abs path of a same-fold a57 ckpt_fold$FOLD.pt>"; exit 1; }
  [ "$(basename "$INIT")" = "ckpt_fold$FOLD.pt" ] || { echo "INIT must be a ckpt_fold$FOLD.pt (val fold $FOLD = never trained on it)"; exit 1; };; esac
case "$RECIPE" in a57aux|a57auxw) [ -f "$W/data/labels/aux_v2.csv" ] || { echo "aux csv $W/data/labels/aux_v2.csv missing (rebuild payload after build_aux_v2.py)"; exit 1; };; esac
[ -f "$W/pretrained/$FAM/model.safetensors" ] || { echo "pretrained $FAM missing (run 01_pretrained.py)"; exit 1; }
for k in 0 1 2 3 4; do [ -f "$W/cache336/slices_part$k.npy" ] || { echo "cache336 part $k missing"; exit 1; }; done
TAG=$(echo "$LABELS" | sed -E 's/^labels_//; s/\.csv$//; s/_exp108_r7$//; s/_r7$//')
case "$RECIPE" in a57) NODE=a57ba94073;; a57nc) NODE=a57ba94073nc;; a57w) NODE=a57ba94073w;; a57ncw) NODE=a57ba94073ncw;; a57aux) NODE=a57ba94073aux;; a57auxw) NODE=a57ba94073auxw;; a57antiw) NODE=a57ba94073antiw$ANTI_GT;; a57ttst) NODE=a57ba94073ttst$PS_TAG;; a57mix) NODE=a57ba94073mix;; a57elr) NODE=a57ba94073elr;; a57dino3) NODE=a57ba94073d3$LR_T;; a57elr1) NODE=a57ba94073elr1;; f0d) NODE=f0d1753e;; s4c) NODE=s4c184;;
  b8d) NODE=8bd0d5e41f;; a4c) NODE=4a0c41e2eb;; esac
NAME=${NAME:-a3_${NODE}_${TAG}_s${SEED}_f${FOLD}}
OUT=$W/runs/$NAME
ARGS=(--fold "$FOLD" --seed "$SEED" --family "$FAM" --labels "$LABELS" --epochs 40 --cache cache336
      --n-parts 5 --batch 2 --accum 2 --workers "$NW" --overrides "$OVR" --trainer "$TRAINER" "$@")

[ -f "$OUT/result.json" ] && { echo "$NAME already complete"; exit 0; }
if [ -f "$OUT/pid" ] && kill -0 "$(cat "$OUT/pid")" 2>/dev/null; then echo "$NAME already running pid $(cat "$OUT/pid")"; exit 1; fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)"; then
  echo "WARN: GPU $GPU already has a process (old boxes ran 1 job/GPU; ~16-20 GB each)"; [ "${FORCE:-0}" = 1 ] || exit 1
fi
mkdir -p "$OUT"
LJ=$(printf '%s\n' "${ARGS[@]}" | "$PY" -c 'import json,sys;print(json.dumps([l.rstrip("\n") for l in sys.stdin]))')
if [ -f "$OUT/launch.json" ]; then
  "$PY" -c 'import json,sys;a=json.load(open(sys.argv[1]))["args"];b=json.loads(sys.argv[2]);sys.exit(0 if a==b else 1)' "$OUT/launch.json" "$LJ" \
    || { echo "args differ from the first launch of $NAME (resume would be refused). Use a new RUN_NAME."; exit 1; }
else
  echo "{\"name\": \"$NAME\", \"recipe\": \"$RECIPE\", \"gpu\": $GPU, \"args\": $LJ, \"launched\": \"$(date -u +%FT%TZ)\"}" > "$OUT/launch.json"
fi
cd "$W"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 setsid nohup "$PY" -u "$W/code/$WRAP" --root "$W" --run-name "$NAME" \
  "${ARGS[@]}" >> "$OUT/train.log" 2>&1 < /dev/null &
echo $! > "$OUT/pid"
echo "started $NAME on gpu$GPU pid $(cat "$OUT/pid")  log: $OUT/train.log"
echo "$(date -u +%T) start $NAME gpu$GPU pid$(cat "$OUT/pid")" >> "$W/runs/a3_launch.log"
