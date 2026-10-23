# Label versions ledger

Rule (CLAUDE.md §6): never train on unversioned labels; every version logged here with its gold-58 agreement.

| Version | Date | Method | Gold-58 macro AUC | Status |
|---------|------|--------|-------------------|--------|
| public: pilkwang regex v1 | 2026-08-08 (eval) | rules/regex over reports | 0.812 | reference |
| public: pilkwang LLM v2 | 2026-08-08 (eval) | API LLM over reports, soft quantized {.08/.28/.68/.82/.94} | **0.870** ← bar to beat | reference |
| public: barun2104 / jezrealmomoh | 2026-08-08 (eval) | n/a | ⚠ circular (gold copied into files) — unusable for quality eval | rejected |
| labels-v01 | 2026-08-08 | `label_v01.py`: rubric-encoded prompt, 5-level verdicts, Qwen2.5-32B-AWQ, T=0 | **0.809** — FAILED vs bar. Diagnosis: verdict quantization + hard rubric thresholding collapses ranking (grade-2 MCL sprain → 0.03 ≈ pristine ligament); worst: Synovitis .633, Effusion .714, Lateral OA .733, MCL .782 | rejected |
| labels-v02 | 2026-08-08 | `label_v02.py`: graded 0-100 probability, per-target severity ladders | **0.810** — no better than v01; MCL/Synovitis/Effusion up, OA/Contusion/Baker's down (58-point noise + task still too hard) | rejected |
| labels-v03 | 2026-08-08 | `label_v03.py`: DESCRIPTIVE factored extraction à la pilkwang (m ∈ NM/NORM/MINOR/PRESENT × severity 0–3, rubric folded into the PRESENT/MINOR boundary; score map in code: .03/.15/.40/.62/.80/.95); Qwen3-32B-AWQ | **0.839 solo** (ACL .953, MCL .975 — factored extraction fixed the ligaments; weak: Contusion .692, Synovitis .698) | accepted as component |
| **labels-v04 = prob-mean(v03, v02, pilkwang v2)** | 2026-08-11 | equal-weight probability mean over all 4,407 studies (parse-fail rows NaN-masked); built by `scripts/make_labels_v04.py` → `data/labels/labels_v04.csv` | **0.895** (rank-mean variant 0.893; components 0.869–0.873) | **LOCKED — the training label set** |

Full-run components: v02 full = 0.872, v03 full = 0.869 on gold-58 (both ran higher than their smokes — vLLM greedy decoding is not batch-stable; treat ±0.02 as run noise).

Both labelers (v02, v03) weak on the same targets as pilkwang (Synovitis ~.7, Contusion ~.75) — report-intrinsic difficulty, not prompt failure; these need the image model.

**Method learnings** (from reading pilkwang's `api_labeler.py`):
- Their 0.870 set = regex-rules primary + hosted-API LLM as *second reader* only where rules were silent; output schema-enforced (verdict YES/NO/UNK × severity 0–3), scores mapped in code → the quantized {.08,.28,.68,.82,.94}.
- "Judge only what the report states, do not infer"; explicitly-normal (NO) ≠ unmentioned (UNK). Descriptive extraction beats outcome-prediction prompting.
- ⚠ Prompt iteration is evaluated on only 58 gold points — per-target swings of ±0.08 are noise. Don't hill-climb details; change design, or ensemble.

**Colab env pins** (Colab CUDA 12 + new transformers both break latest vLLM): `vllm==0.10.1.1 transformers==4.55.2`.

| labels-v05 | 2026-08-18 | 45 Sonnet-5 subagents (Claude Max, workflow fan-out; ~100 reports each; descriptive factored m/s/e + lang; JSONL part-files; gold-58 gate 0.874 passed; 77 missing repaired, 1 corrupted-uid dup fixed) | **0.8738 solo** — equal to v02/pilkwang, not better; strong Medial OA .92/Baker's .94/Contusion .81, weak Fracture .80/Synovitis .68 | component |
| **labels-v06 = mean(v02, v03, pilkwang, v05)** | 2026-08-18 | 4-reader prob-mean (`scripts/build_labels_v05.py`) → `data/labels/labels_v06.csv` | **0.8992** vs v04 0.895 — +0.004, *within* gold-58 noise (±0.012), adopted on structural grounds: a 4th decorrelated equal-quality reader can barely hurt; Contusion .82→.85, Fracture .91→.90, Synovitis .76 | **ADOPTED for training** |

| **labels-v07 = 0.5·v06 + 0.5·VPT-OOF** | 2026-08-20 | Cross-fitted OOF distillation (our protocol; not in the Arazo paper — I-043): leak-free out-of-fold predictions of the 5-fold VPT ensemble (exp-012, gold 0.8486) prob-mean-blended into v06; built in-session (`/content/oof_gate.py`), stored `Drive/rsna-knee/assets/labels_v07.csv` + `exp-012/oof_preds.npy` | **0.9200** vs v06 0.8991 (+0.021; w=0.25 gives 0.9185 — robust plateau, not a knife-edge). Image arm fixes report blind spots: Effusion .89→.97, Synovitis .76→.84, Fracture .90→.94, PF OA .89→.91; ACL 1.00→.96 the only real cost | **CANDIDATE** — adopted for exp-015 fold-0 A/B; ⚠ w chosen on gold-58: treat per-target detail as noisy (I-042), and the training A/B (not the agreement number) is the adoption gate |

| **anchors_v01** (not training labels — immutable anchor set) | 2026-08-20 | PLAN-arazo P2 Tier A: per-target hard 1/0/NaN consensus of 5 teachers (4 LLM readers + VPT-OOF), tiered ladder T1(all4≥.85+OOF≥.6) → T3(all4≥.60 "all say PRESENT") → T4(top-K); negatives all≤.10-.20+OOF≤.3; gold-58 excluded by construction; `scripts/make_anchors_v01.py` → `data/labels/anchors_v01.csv` (3,694 studies) | purity on gold-58 hits: 1.00 on 9 measurable targets; **Contusion 0.77** (reader-driven over-calling; OOF gate tightened to .65; #1 Tier-B human-audit priority). Pos counts: ACL 105, MM 229, MedOA 118, Contusion 168, Synovitis 66(T3), MCL 15(T4), Fracture 15(T3) — all ≥15 | ANCHOR SET for exp-019+ (never trained on as labels; loss anchors only) |

Language mix (v05 lang tags over 4,407): en 1735, es 681, tr 546, hr 408, el 321, de 263, bg 220, nl 152 — Croatian/Greek/Bulgarian are bigger than our earlier marker-scan suggested; useful for per-language error analysis.

Notes:
- Weakest public targets (pilkwang v2): Synovitis .694, Contusion .768, Lateral OA .789, Effusion .830 — all severity-threshold targets; rubric encoding targets exactly these.
- Eval protocol: `eda/eval_public_labels.py` pattern — soft scores vs gold-58, per-target + macro AUC. Gold-58 is validation-only everywhere (never train on it).
- v01 hard-label rule: score ≥ 0.65 (borderline → negative per rubric).

| **labels-v08-text — REJECTED** | 2026-08-21 | exp-026: 5-fold cross-fitted XLM-RoBERTa-large + LaBSE trained on the 4-reader consensus (`anchors_v02_text` hard anchors at w=1.0 + v06-soft at 0.30), the first *trained* (non-prompted) reader we have built; `src/labels/jointmatch_text.py` | **0.8847 gold-58 macro AUC vs v06's 0.8991 — FAILS the bar.** Decisive detail: on gold cells where the readers DISAGREE it scores **0.4808 (chance)** vs the readers' 0.6350; where they AGREE it ties them (0.9129 vs 0.9146). Best blend 0.25*text + 0.75*v06 = 0.9049, inside the +/-0.012 noise | **REJECTED — do not train on it.** I-057: distilling the reader consensus cannot exceed it; better labels need information from OUTSIDE the ensemble (images, per v07's 0.9200; or human review of the I-051 disagreement queue) |

| **labels-v09 (candidate)** | 2026-08-23 | exp-035 anchored two-family OOF fusion (research/jointmatch-v2-plan.md): q0=v06 immutable anchor; corrections only where slice-bag OOF (free from val_preds dumps, self-checked alignment) and slot/VPT OOF AGREE (|d|<0.25); global rho grid {.25,.4,.5}, smallest-within-0.002 rule; never recursive; built by `scripts/build_labels_v09.py` on the user's session | pending — gate A: >= 0.9250 required (v07 = 0.9200) | PENDING — notebook scripts/colab_exp035_labels_v09.ipynb |
| labels-v09 — BUILT, gate passed, retrain TIED | 2026-08-23 | exp-035 result: agreement 0.9266 (rho 0.4; grid .25/.4/.5 -> .9204/.9266/.9285; v07 = 0.9200); fold-0 retrain val 0.8709 / gold 0.9019 vs control 0.8706/0.9009 = TIE at T=0.0052 | **0.9266** | ARCHIVED — v07 stays the production label set; I-075: label gains beyond v06 do not convert to model gains (2nd measured instance) |
| **gold_v2 (eval-only)** | 2026-08-26 | 111-cell disagreement queue adjudicated by a 28-agent blind jury workflow (3 lenses/cell, 96% unanimous, 4 resolver cells, 1 BLANK): 44x1 / 66x0; jury-vs-readers 0.842 on RR cells (jury adds signal on ~16% of them), 0.986 on RI cells | gold_v2.csv = 118 studies / 806 cells | **CAVEAT: RI cells (n=72) encode REPORT-truth by construction — a report-reading jury cannot arbitrate report-vs-image conflicts; ladder-validate before trusting, and eval must be FOLD-AWARE (the 60 new studies are training studies — only a ckpt's own val-fold cells are clean for it) |

| **weights_v01 (not labels — per-cell sample weights)** | 2026-08-27 | `scripts/build_weights_v01.py`: w = 0.3 + 0.7*(1 - 2*std) over the 4 v06 readers (v02/v03/pilkwang/v05), <2 readers -> 1.0 | mean 0.829, 19.9% cells < 0.7; per-target lows Effusion .746 / PF OA .799; corr 0.689 vs disagree_v01 __d | staged for exp-046 (pre-registered); trainer flag --weight_csv |

| **gold_v3_new (eval-only, REPORT-truth)** | 2026-08-27 | 250 fold-0-val studies (170 lang-stratified random + 80 rare-target-enriched, arm-flagged), 3-lens blind Fable jury + resolver (wf_4bd2a027-d00): 96.3% cell unanimity (111/3000 disputed), 78 BLANK cells, >=35 positives every target | v06-vs-jury 0.9894 (rand arm) vs v06-vs-gold58 0.899 -> the jury rows are REPORT-truth (they cannot arbitrate report-vs-image, same caveat as gold_v2 RI cells); gold-58 stays the image-truth anchor | data/labels/gold_v3_new.csv; ladder validation pending (zero-GPU via saved val_preds) |

| weights_v01 — TESTED, TIE | 2026-08-28 | exp-046 fold-0 production run: gold 0.9053 vs control 0.9050 (+0.0003, sub-T); val -0.005 (conf-weight double-counting) | line closed per pre-registration |
