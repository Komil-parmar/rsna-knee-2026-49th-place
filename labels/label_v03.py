"""labels-v03: descriptive factored extraction (pilkwang-style) + code-side map.

v01 (0.809) and v02 (0.810) asked the model to PREDICT image-review outcomes -
extraction and threshold-reasoning in one shot. The 0.870 public labeler
instead asks WHAT THE REPORT SAYS (verdict x severity) and maps to scores in
code - an easier, more reliable task. v03 adopts that factorization with a
finer state space and the rubric folded into the PRESENT/MINOR boundary:

  per finding: m in {NM (not mentioned), NORM (explicitly normal),
               MINOR (sub-threshold changes), PRESENT (finding asserted)},
               s in 0-3 (severity, only for PRESENT), e = short quote.
  score map:   NORM .03 < NM .15 < MINOR .40 < PRESENT .62/.80/.95 by s.

Also evaluates rank-average ensembles with any extra score CSVs you pass, so
each run reports solo AND ensemble AUC vs gold-58.

Colab (A100), same session/pins as before; upload report_labels_v2.csv too:
    !python label_v03.py --train_csv train.csv --out labels_v03_smoke.csv \
        --limit 200 --ensemble_with labels_v02_smoke.csv,report_labels_v2.csv
    !python label_v03.py --train_csv train.csv --out labels_v03.csv \
        --ensemble_with report_labels_v2.csv
Default model: Qwen3-32B-AWQ (better than Qwen2.5-32B; fits A100-40GB).
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa",
        "lateral_oa", "pf_oa", "effusion", "synovitis", "bakers",
        "contusion", "fracture"]
M_SCORE = {"NORM": 0.03, "NM": 0.15, "MINOR": 0.40}
S_SCORE = {0: 0.62, 1: 0.62, 2: 0.80, 3: 0.95}

SYSTEM = """You are an expert musculoskeletal radiologist. Read ONE knee-MRI \
radiology report in its ORIGINAL language (Spanish, English, Dutch, French, \
Turkish, German, Italian, Greek, or others). For each of the 12 findings \
below, record what THE REPORT STATES. Do not infer a finding from a related \
one; judge only what is written.

For each finding output:
  "m" - one of:
     "NM"      the report never mentions the structure/finding
     "NORM"    the report explicitly calls it normal, intact, or preserved
     "MINOR"   the report mentions only sub-threshold changes (see each
               finding's MINOR list)
     "PRESENT" the report asserts the finding at or above threshold
  "s" - severity 0-3, only meaningful when m="PRESENT":
     1 mild / small / low-grade / partial without grade
     2 stated without qualification / moderate
     3 severe / large / extensive / complete / high-grade / displaced
  "e" - short verbatim quote (<=12 words, original language) or ""

FINDINGS (PRESENT threshold | MINOR examples):
1. acl - ACL tear: partial or complete tear, rupture, discontinuity | MINOR: \
mucoid degeneration, intrasubstance signal, sprain/strain without tear, \
thickening, ganglion.
2. mcl - MCL tear: grade 2-3 sprain/tear, torn or disrupted fibers | MINOR: \
grade 1 sprain, periligamentous edema only, chronic thickening/remote injury.
3. medial_meniscus - MEDIAL meniscus tear: tear/rupture of any type (radial, \
horizontal, root, bucket-handle, flap), signal reaching articular surface, \
truncation, maceration, displaced fragment | MINOR: intrasubstance or mucoid \
degeneration not reaching the surface, mild irregularity, extrusion alone.
4. lateral_meniscus - same, LATERAL meniscus only.
5. medial_oa - MEDIAL tibiofemoral cartilage loss: high-grade (grade 3-4, \
full-thickness, bone-on-bone, severe narrowing) chondropathy/osteoarthritis \
| MINOR: grade 1-2 chondropathy, thinning, fibrillation, small focal defect, \
osteophytes alone.
6. lateral_oa - same, LATERAL tibiofemoral compartment.
7. pf_oa - same, PATELLOFEMORAL compartment (patella/trochlea).
8. effusion - joint effusion, moderate or large | MINOR: small, trace, \
minimal, mild, or physiologic fluid.
9. synovitis - synovitis / synovial thickening or proliferation / pannus / \
hoffitis asserted | MINOR: "possible synovitis", nonspecific synovial \
enhancement remark.
10. bakers - Baker/popliteal cyst, moderate or large | MINOR: small cyst.
11. contusion - bone contusion / bone bruise / traumatic bone-marrow edema \
(no fracture line) | MINOR: nonspecific or degenerative subchondral marrow \
signal next to cartilage loss.
12. fracture - ACUTE fracture: cortical break, fracture line, depression, \
acute osteochondral/insufficiency fracture | MINOR: old healed fracture, \
remote deformity, "no displaced fracture" wording with equivocal acuity.

Rules:
- Structure called normal/intact/preserved -> NORM (not NM).
- Never named at all -> NM (not NORM).
- Medial and lateral are distinct; answering one says nothing about the other.
- Negated findings ("sin derrame", "geen ruptuur") -> NORM for that finding.
- If both a MINOR and a PRESENT-level statement exist, use PRESENT.

Output STRICT JSON only, exactly the 12 keys in this order:
{"acl": {"m": "...", "s": 0, "e": "..."}, "mcl": {...}, "medial_meniscus": \
{...}, "lateral_meniscus": {...}, "medial_oa": {...}, "lateral_oa": {...}, \
"pf_oa": {...}, "effusion": {...}, "synovitis": {...}, "bakers": {...}, \
"contusion": {...}, "fracture": {...}}"""


def build_prompt(report, tokenizer, think):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"REPORT:\n{report[:6000]}"}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=think)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)


def to_score(m, s):
    if m == "PRESENT":
        return S_SCORE.get(int(s) if s in (0, 1, 2, 3) else 0, 0.62)
    return M_SCORE.get(m, 0.5)


def parse_response(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in KEYS:
        item = d.get(k)
        if not isinstance(item, dict):
            return None
        mv = str(item.get("m", "")).strip().upper()
        if mv not in ("NM", "NORM", "MINOR", "PRESENT"):
            return None
        try:
            sv = int(item.get("s", 0))
        except (TypeError, ValueError):
            sv = 0
        out[k] = {"m": mv, "s": max(0, min(3, sv)),
                  "score": to_score(mv, sv),
                  "evidence": str(item.get("e", ""))[:120]}
    return out


def extract_scores(df):
    """Score frame from ours ({k}__score) or pilkwang (raw TARGETS) schema."""
    out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"]})
    for t, k in zip(TARGETS, KEYS):
        if f"{k}__score" in df.columns:
            out[t] = pd.to_numeric(df[f"{k}__score"], errors="coerce")
        elif t in df.columns:
            out[t] = pd.to_numeric(df[t], errors="coerce")
    return out.set_index("StudyInstanceUID")


def eval_scores(scores, gold, name):
    from sklearn.metrics import roc_auc_score
    common = gold.index.intersection(scores.index)
    aucs = {}
    for t in TARGETS:
        if t not in scores.columns:
            continue
        y = gold.loc[common, t].astype(int)
        p = scores.loc[common, t]
        ok = p.notna()
        if y[ok].nunique() > 1:
            aucs[t] = roc_auc_score(y[ok], p[ok])
    macro = float(np.mean(list(aucs.values())))
    print(f"\n=== {name} vs gold-{len(common)}: macro AUC {macro:.3f} ===")
    for t, a in aucs.items():
        print(f"  {t:18s} {a:.3f}")
    return macro


def rank_avg(frames):
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    acc = None
    for f in frames:
        r = f.loc[idx, TARGETS].rank(pct=True)
        acc = r if acc is None else acc + r
    return acc / len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="train.csv")
    ap.add_argument("--out", default="labels_v03.csv")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--max_model_len", type=int, default=10240)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", action="store_true",
                    help="enable Qwen3 thinking mode (slower, maybe better)")
    ap.add_argument("--ensemble_with", default="",
                    help="comma-separated score CSVs to rank-ensemble with")
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    gold = train[train[TARGETS].notna().all(axis=1)].set_index(
        "StudyInstanceUID")
    reports = train[["StudyInstanceUID", "Report"]].copy()
    if args.limit:
        keep = pd.concat([reports.head(args.limit),
                          reports[reports["StudyInstanceUID"].isin(
                              gold.index)]])
        reports = keep.drop_duplicates("StudyInstanceUID")
    print(f"labeling {len(reports)} reports with {args.model} "
          f"(think={args.think})")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92, enable_prefix_caching=True)
    if args.think:
        sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=3200)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=800)

    prompts = [build_prompt(r, tok, args.think)
               for r in reports["Report"].fillna("")]
    outputs = llm.generate(prompts, sp)

    rows, failed = [], []
    for (_, rec), out in zip(reports.iterrows(), outputs):
        parsed = parse_response(out.outputs[0].text)
        if parsed is None:
            failed.append(rec["StudyInstanceUID"])
            parsed = {k: {"m": "PARSE_FAIL", "s": 0, "score": 0.5,
                          "evidence": ""} for k in KEYS}
        row = {"StudyInstanceUID": rec["StudyInstanceUID"]}
        for k in KEYS:
            row[f"{k}__m"] = parsed[k]["m"]
            row[f"{k}__s"] = parsed[k]["s"]
            row[f"{k}__score"] = parsed[k]["score"]
            row[f"{k}__evidence"] = parsed[k]["evidence"]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"parse failures: {len(failed)}"
          + (f" e.g. {failed[:3]}" if failed else ""))
    for t, k in zip(TARGETS, KEYS):
        df[t] = (df[f"{k}__score"] >= 0.6).astype(int)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")

    mine = extract_scores(df)
    eval_scores(mine, gold, "labels-v03 SOLO (v01 .809, v02 .810, bar .870)")
    extras = [p.strip() for p in args.ensemble_with.split(",") if p.strip()]
    frames = {"v03": mine}
    for p in extras:
        try:
            frames[p.split("/")[-1]] = extract_scores(pd.read_csv(p))
        except Exception as exc:
            print(f"skip {p}: {exc}")
    names = list(frames)
    if len(names) > 1:
        for other in names[1:]:
            eval_scores(rank_avg([frames["v03"], frames[other]]), gold,
                        f"ENSEMBLE v03 + {other}")
        if len(names) > 2:
            eval_scores(rank_avg(list(frames.values())), gold,
                        "ENSEMBLE all " + "+".join(names))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
