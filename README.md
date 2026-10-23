# RSNA Knee Abnormality Detection 2026, 49th place (silver, solo)

Code, labels and write-up of my solution to the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition on Kaggle: twelve findings per knee MRI study, macro ROC AUC, 4,407 training studies with free-text reports and only 58 expert-labelled, hidden test of ~1,300 studies scored in a nine-hour offline notebook.

| | public | private |
|---|---|---|
| pick A: own models + three public members, per-finding rank blend | 0.962 | 0.951 (rank 49 of 6,425) |
| pick B: own models only, no external data or weights | 0.958 | 0.949 |

**Read the write-up first**: [`writeup/WRITEUP.md`](writeup/WRITEUP.md) (also posted on Kaggle: [49th place (silver, solo): whole-study 2.5D reader + report labels + rank blend](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/49th-place-silver-solo-whole-study-2-5d-reader)). It explains what moved the score, how the labels were built, the model, the blend, the instruments and everything that did not work. This README only says where the code is and how it runs.

## Released artefacts

- **Training labels** (soft probabilities per study, derived from the reports and from out-of-fold image predictions; the 58 expert studies were never trained on): Kaggle dataset [`komilparmar/rsna-knee-2026-labels-49th-place`](https://www.kaggle.com/datasets/komilparmar/rsna-knee-2026-labels-49th-place) with `labels_v04`, `labels_v06` (text readers), `labels_v14` (final) and the scanner-grouped folds.
- **Checkpoints** (Kaggle Models, Apache-2.0): [`rsna-camp-v14-a57`](https://www.kaggle.com/models/komilparmar/rsna-camp-v14-a57) (ConvNeXtV2-Tiny, 5 folds) and [`rsna-camp-v14-4a0c`](https://www.kaggle.com/models/komilparmar/rsna-camp-v14-4a0c) (ConvNeXt-Tiny, 5 folds) are the own members of pick A; [`rsna-camp-v14-a57s2000`](https://www.kaggle.com/models/komilparmar/rsna-camp-v14-a57s2000) and [`rsna-camp-v14-8bd0`](https://www.kaggle.com/models/komilparmar/rsna-camp-v14-8bd0) complete the 20-model own-only pick B.
- **Competition data is not included** and cannot be redistributed; everything here reads it from `/kaggle/input` on Kaggle or from caches you build with the preprocessing scripts.

## Layout

| folder | what it holds |
|---|---|
| `writeup/` | the write-up and its figures |
| `preprocessing/` | Kaggle-side conversion of the 570 GB DICOM tree: `kaggle_prep_full/` builds and pushes the eight 512 px shard kernels (decode, sort by position, left-knee convention, 160 mm field of view); `build_allslice336.py` builds the all-slice 336 px / 130 mm training cache (five CPU kernels); `make_folds.py` builds the scanner-fingerprint GroupKFold split; `meta_analysis.py` is the DICOM-header EDA |
| `labels/` | the report readers (`label_v02.py`, `label_v03.py`: descriptive factored extraction with vLLM + Qwen), reader ensembling (`make_labels_v04.py`, `build_labels_v05.py`), the image-teacher rounds (`build_v13d.py`, `build_round.py`: 0.3 x text + 0.7 x out-of-fold ensemble with OA decoupling, anchored to v06) and `LABEL_LEDGER.md`, the version-by-version record with gold-58 scores |
| `training/code/` | the whole-study slice-bag reader: `train_oof224.py` (K-fold wrapper and the fixed recipe), `train_slicebag_campaign_v1.py` (base trainer), `train_slicebag_campaign_a57.py` / `_4a0c.py` / `_8bd0.py` (the three final recipes), `train_baseline.py`, `lowres_utils.py`, `mri_augs.py` + `augs/` (augmentation library). The `_a57mix/_a57elr/_a57aux/_a57anti/_a57ttst/_a57cut/_a57pu` trainers are the negative results of section 8 of the write-up |
| `training/bootstrap/` | bring-up of a training box: `00_env.sh`, `01_pretrained.py` (ImageNet weights), `02_stage_cache336.py` (pulls the cache from the Kaggle kernel outputs), `smoke_xp.sh`, `run_xp.sh` (launches one fold of one recipe), `xp_status.sh` |
| `inference/` | `own_models_kernel/` is the Kaggle notebook that scores the own models (whole-study reads, 20 checkpoints); `build_pb.py` turns it into pick B (slice cap, two slice grids, the three rules); `build_blend4.py` builds pick A on top of it (public members, per-finding rank blend, exclusions, rules, `--invert` for probes); `probe_readout.py` is the column-inversion read-out; `check_trial.py` / `cmp_trial.py` gate a trial run before submission; `run_and_submit.py`, `publish_ckpts.py` are the Kaggle plumbing |
| `analysis/` | `lb_snap.py` (leaderboard snapshots and rank-in-block reads), `addone_fold.py` / `gate_addone.py` (leak-matched add-one gate for a candidate member), `bank_to_fold_npz.py`, `private_readouts.md` (per-finding private AUCs from the probes), the figure scripts |

## Reproducing the pipeline

1. **Preprocess on Kaggle.** `python preprocessing/kaggle_prep_full/build_and_push.py` pushes eight CPU kernels that write per-study 512 px uint8 arrays (all 4,407 studies, ~121 GB). Then `python preprocessing/build_allslice336.py` pushes the five cache kernels that produce `slices_part{0..4}.npy` + `index_part{k}.csv` at 336 px (819,078 slices). `python preprocessing/make_folds.py` writes `folds.csv` (83 scanner groups, 5 folds; the 58 expert studies are flagged and never trained on).
2. **Labels.** Run the readers in `labels/` on the reports (Colab A100 with the vLLM pins in `LABEL_LEDGER.md`), average them with `make_labels_v04.py` / `build_labels_v05.py` to get the text labels, or skip this and download `labels_v06` and `labels_v14` from the dataset above. After the first five-fold training, `build_v13d.py` / `build_round.py` build the next round from the out-of-fold banks.
3. **Train.** On a Linux box with one or more GPUs (every run uses one GPU, ~5 h per fold on an RTX 5090): `bash training/bootstrap/00_env.sh`, `python training/bootstrap/01_pretrained.py`, `python training/bootstrap/02_stage_cache336.py --dest <work>/cache336 --workers 8`, `bash training/bootstrap/smoke_xp.sh a57 0 labels_v14_r7.csv` (must end with `SMOKE OK`), then `bash training/bootstrap/run_xp.sh a57 <fold> 1000 labels_v14_r7.csv <gpu>` for folds 0 to 4, and the same with recipes `a4c` and `b8d`. The paths in these scripts assume `/workspace/rsna-knee/work` as the work root; change `W` and `PY` at the top of `run_xp.sh` if yours differs. Each run writes `result.json`, `oof_pred.npz` and `ckpt_fold<k>.pt`.
4. **Inference.** Publish the checkpoints as a Kaggle Model (`inference/publish_ckpts.py`), then build the notebook: `python inference/build_pb.py --tag pb3 --cap 160 --phases 0,0.5 --habit 0.30 --oa-dec 0.15` for the own-only entry, or `python inference/build_blend4.py --tag k12 --s6f --r7-model r6 --folds 4 --rap224 0.35 --r224-mirror --habit 0.3 --oa-dec 0.15 --drop "Lateral OA:raptor,s6;Fracture:s6"` for the blend. `build_blend4.py` expects the public notebooks it patches under `inference/public/` (pull them with `kaggle kernels pull`: heliosli's blend notebook, nartaa's `rsna-knee-s6fall-solo`, the Raptor 224 px notebook) and the public weights attached as datasets; those are their authors' work and are not redistributed here. Push the generated kernel with `kaggle kernels push`, gate the trial with `check_trial.py`, submit with `run_and_submit.py`.
5. **Read a finding off the public board.** `python inference/probe_readout.py` builds the column-inversion probes; `analysis/private_readouts.md` has the formula and the numbers.

## Credits

- nartaa: the Raptor CoAtNet weights and the S6 all-data ConvNeXt, with the discussion that explains them.
- heliosli: the R6 compact ConvNeXt-S reader and the per-finding blend notebook.
- pilkwang: the public LLM label set used as one of the four text readers.
- tonylica: the launch notebook whose slot design and laterality handling I started from.
- The hosts and the annotating radiologists.

## Licence

Apache License 2.0 for the code in this repository (see `LICENSE`). Public components keep their own licences; the competition data stays under the competition's rules.
