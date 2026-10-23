"""labels-v02: graded 0-100 probability extraction (fixes v01's rank collapse).

v01 (macro 0.809 vs gold-58) failed because 5-level verdicts + strict
thresholding collapse ranking: a grade-2 MCL sprain scored 0.03, same as a
pristine ligament, though it is far likelier to be label-positive. v02 asks for
a 0-100 probability with an explicit per-target severity LADDER, so
sub-threshold findings get intermediate ranks. Bar: pilkwang 0.870.

Colab (A100):
    !pip uninstall -q -y vllm torch torchvision torchaudio xformers
    !pip install -q vllm==0.10.1.1 transformers==4.55.2
    !python label_v02.py --train_csv train.csv --out labels_v02_smoke.csv --limit 200
    !python label_v02.py --train_csv train.csv --out labels_v02.csv
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

SYSTEM = """You are a subspecialty musculoskeletal radiologist. You will read \
one knee-MRI radiology report (possibly in Spanish, English, Dutch, French, \
Turkish, German, Italian, or another language - read it natively) and \
estimate, for 12 findings, the PROBABILITY (integer 0-100) that expert image \
reviewers - who re-read the IMAGES with strict criteria, independent of the \
report - would call the finding POSITIVE.

Because reviewers apply severity thresholds, a finding that is mentioned but \
below threshold gets an INTERMEDIATE probability (not near-zero): reports and \
image review disagree most near thresholds. Use the FULL 0-100 range; grade by \
the severity language. Ladders:

1. acl (positive = high-grade partial/complete tear, discontinuity or >50% \
fibers): 0-5 normal or not mentioned; 10-30 mucoid degeneration / \
intrasubstance signal / "sprain"; 40-70 partial tear of unspecified grade; \
75-100 high-grade partial, complete tear, discontinuity, or chronic complete \
tear / prior rupture.
2. mcl (positive = high-grade partial or complete ACUTE tear): 0-5 normal; \
15-35 grade 1 sprain / periligamentous edema only; 40-70 grade 2 or \
unspecified sprain-tear; 80-100 grade 3, complete, or clearly acute \
high-grade tear.
3. medial_meniscus (positive = tear: signal definitely reaching an articular \
surface, truncation, maceration, displaced/bucket-handle/root/radial/flap \
tear): 0-5 normal; 10-30 intrasubstance or mucoid signal NOT reaching \
surface; 35-60 equivocal surfacing signal, "possible tear"; 70-100 definite \
tear or morphologic change. MEDIAL meniscus only.
4. lateral_meniscus: same ladder, LATERAL meniscus only.
5. medial_oa (positive = high-grade cartilage loss >50% thickness over >=1 cm \
in MEDIAL tibiofemoral compartment): 0-5 none; 10-35 low-grade chondropathy \
(grade I-II, thinning, fibrillation); 40-70 grade III or small/focal \
full-thickness defect; 75-100 extensive full-thickness loss, grade IV, \
bone-on-bone, severe compartment narrowing.
6. lateral_oa: same ladder, LATERAL tibiofemoral compartment.
7. pf_oa: same ladder, PATELLOFEMORAL compartment (patella/trochlea).
8. effusion (positive = moderate or large): 0-10 not mentioned; 15-35 \
small/trace/minimal; 50-70 unqualified "effusion" or mild-moderate; 75-100 \
moderate, large, or tense effusion.
9. synovitis (positive = synovial inflammation AND thickening): 0-15 not \
mentioned and no supporting signs; 25-45 moderate/large effusion present but \
synovium not described (they often co-occur); 60-100 explicit synovitis, \
synovial thickening/proliferation, pannus, or hoffitis.
10. bakers (positive = moderate or large popliteal cyst): 0-5 none; 15-35 \
small cyst; 55-70 unqualified Baker cyst; 75-100 moderate/large.
11. contusion (positive = traumatic bone-marrow edema WITHOUT fracture line): \
0-5 none; 15-35 nonspecific or degenerative subchondral marrow signal \
(adjacent to cartilage loss); 70-100 traumatic contusion / bone bruise / \
impaction marrow edema, incl. pivot-shift pattern.
12. fracture (positive = ACUTE cortical break or fracture line): 0-5 none; \
15-35 old/healed fracture or remote deformity; 60-100 acute fracture line, \
cortical break, depression, or insufficiency fracture with a line.

Output STRICT JSON only, exactly 12 keys:
{"acl": {"p": <0-100>, "e": "<short verbatim quote, original language, or \
empty>"}, "mcl": {...}, "medial_meniscus": {...}, "lateral_meniscus": {...}, \
"medial_oa": {...}, "lateral_oa": {...}, "pf_oa": {...}, "effusion": {...}, \
"synovitis": {...}, "bakers": {...}, "contusion": {...}, "fracture": {...}}"""


def build_prompt(report, tokenizer):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"REPORT:\n{report[:6000]}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)


def parse_response(text):
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
        if not isinstance(item, dict) or "p" not in item:
            return None
        try:
            p = float(item["p"])
        except (TypeError, ValueError):
            return None
        out[k] = {"score": min(max(p, 0.0), 100.0) / 100.0,
                  "evidence": str(item.get("e", ""))[:120]}
    return out


def score_vs_gold(df, train):
    from sklearn.metrics import roc_auc_score
    gold = train[train[TARGETS].notna().all(axis=1)]
    merged = gold.merge(df, on="StudyInstanceUID", suffixes=("_gold", ""))
    aucs = {}
    for t, k in zip(TARGETS, KEYS):
        y = merged[f"{t}_gold"].astype(int)
        if y.nunique() > 1:
            aucs[t] = roc_auc_score(y, merged[f"{k}__score"])
    macro = float(np.mean(list(aucs.values())))
    print(f"\n=== labels-v02 vs gold-{len(merged)}: macro AUC {macro:.3f} "
          f"(bar: pilkwang 0.870, v01 0.809) ===")
    for t, a in aucs.items():
        print(f"  {t:18s} {a:.3f}")
    return macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="train.csv")
    ap.add_argument("--out", default="labels_v02.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    reports = train[["StudyInstanceUID", "Report"]].copy()
    if args.limit:
        gold_ids = train[train[TARGETS].notna().all(axis=1)][
            "StudyInstanceUID"]
        keep = pd.concat([reports.head(args.limit),
                          reports[reports["StudyInstanceUID"].isin(gold_ids)]])
        reports = keep.drop_duplicates("StudyInstanceUID")
    print(f"labeling {len(reports)} reports with {args.model}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92)
    sp = SamplingParams(temperature=0.0, max_tokens=900)

    prompts = [build_prompt(r, tok) for r in reports["Report"].fillna("")]
    outputs = llm.generate(prompts, sp)

    rows, failed = [], []
    for (_, rec), out in zip(reports.iterrows(), outputs):
        parsed = parse_response(out.outputs[0].text)
        if parsed is None:
            failed.append(rec["StudyInstanceUID"])
            parsed = {k: {"score": 0.5, "evidence": "PARSE_FAIL"}
                      for k in KEYS}
        row = {"StudyInstanceUID": rec["StudyInstanceUID"]}
        for k in KEYS:
            row[f"{k}__score"] = parsed[k]["score"]
            row[f"{k}__evidence"] = parsed[k]["evidence"]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"parse failures: {len(failed)}"
          + (f" e.g. {failed[:3]}" if failed else ""))

    for t, k in zip(TARGETS, KEYS):  # hard labels: >= 0.65 (borderline -> neg)
        df[t] = (df[f"{k}__score"] >= 0.65).astype(int)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")
    score_vs_gold(df, train)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
