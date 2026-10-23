"""Data-driven figures for the competition write-up (notes/writeup/figures/).
Sources: team/exchange/private_scores_2026-10-23.txt (all scored submissions), data/train.csv (58 expert-labelled studies),
data/labels/labels_v06.csv (text-only labels), team/a3/labels/labels_v14_r7.csv (final labels), src/labels/README.md,
INSIGHTS.md I-182, notes/PRIVATE_READOUTS_2026-10-23.md. Run: py -3.14 notes/writeup/make_figures.py
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notes/writeup/figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#1f2937", "#2563eb", "#ea580c", "#16a34a", "#9ca3af", "#dc2626", "#7c3aed"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
DPI = 200
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def hanley_se(a, n1, n0):
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    return np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


# ------------------------------------------------------------------ submissions table
def load_subs():
    rows = []
    for line in (ROOT / "team/exchange/private_scores_2026-10-23.txt").read_text(encoding="utf-8", errors="replace").splitlines()[2:]:
        m = re.match(r"\s*(\d+) (\S+ \S+)\s+([\d.]+)\s+([\d.]+)\s+(.*)", line)
        if m:
            rows.append(dict(ref=int(m.group(1)), date=pd.Timestamp(m.group(2)), public=float(m.group(3)),
                             private=float(m.group(4)), desc=m.group(5)))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def tag_track(desc):
    pub_words = ["dinosaur", "chain", "public", "rad arm", "public-assets", "Raptor", "heliosli", "stack (", "B1", "H1", "F1",
                 "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "K1", "nartaa", "S6", "R6", "R7", "CENSUS", "Synovitis <-"]
    if desc.startswith(("PB1", "PB2", "PB3")):
        return "own"
    return "pub" if any(w in desc for w in pub_words) else "own"


# ------------------------------------------------------------------ fig 1: what the reports know
def fig1_thesis():
    tr = pd.read_csv(ROOT / "data/train.csv")
    g = tr.dropna(subset=T).set_index("StudyInstanceUID")[T]
    v06 = pd.read_csv(ROOT / "data/labels/labels_v06.csv").set_index("StudyInstanceUID").loc[g.index]
    v14 = pd.read_csv(ROOT / "team/a3/labels/labels_v14_r7.csv").set_index("StudyInstanceUID").loc[g.index]
    a06 = [auc(g[t], v06[t]) for t in T]
    a14 = [auc(g[t], v14[t]) for t in T]
    order = np.argsort(a06)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    y = np.arange(len(T))
    for i, k in enumerate(order):
        ax.plot([a06[k], a14[k]], [i, i], color=GREY, lw=2, zorder=1)
    ax.scatter([a06[k] for k in order], y, s=46, color=ORANGE, zorder=3, label="text-only labels (four LLM report readers, v06)")
    ax.scatter([a14[k] for k in order], y, s=46, color=GREEN, zorder=3, label="final training labels (text + image-teacher rounds, v14)")
    ax.set_yticks(y, [T[k] for k in order], fontsize=9)
    for i, k in enumerate(order):
        ax.text(min(a06[k], a14[k]) - 0.012, i, f"{a06[k]:.2f}", va="center", ha="right", fontsize=7.6, color=ORANGE)
        ax.text(max(a06[k], a14[k]) + 0.012, i, f"{a14[k]:.2f}", va="center", ha="left", fontsize=7.6, color=GREEN)
    ax.set_xlim(0.6, 1.03)
    ax.set_xlabel("label AUC against the 58 expert-labelled studies")
    ax.set_title("What the reports know: text-only labels are weakest exactly where the experts read the images",
                 fontsize=10.2, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 0.42))
    ax.text(0.605, 11.6, f"macro over 12 findings: text-only {np.mean(a06):.3f}, final {np.mean(a14):.3f}.\nThe 58 studies were never trained on.",
            fontsize=8, color=INK, va="top")
    save(fig, "fig3_thesis.png")
    return a06, a14


# ------------------------------------------------------------------ fig 2: waterfall by change
WATERFALL = [
    # (label, public, private, group)   one chain of public bests, each step = the change named
    ("first own model\n1 fold, 224 px, DINOv2", 0.898, 0.883, "model"),
    ("recipe: augmentation,\nslow cosine, EMA, pooling", 0.918, 0.903, "model"),
    ("336 px / 130 mm input", 0.926, 0.907, "model"),
    ("whole-study inference", 0.928, 0.910, "model"),
    ("ConvNeXt backbone,\n5 folds, 2 families", 0.938, 0.922, "model"),
    ("image-teacher labels v10", 0.944, 0.930, "labels"),
    ("labels v11, more members", 0.948, 0.934, "labels"),
    ("cross-attention decoder", 0.949, 0.936, "model"),
    ("OOF-selected ensemble", 0.950, 0.938, "blend"),
    ("stack4 ConvNeXt-336", 0.952, 0.940, "model"),
    ("acquisition augs; OA\ndecoupling of predictions", 0.954, 0.942, "model"),
    ("labels v13D", 0.956, 0.945, "labels"),
    ("labels v14", 0.957, 0.947, "labels"),
    ("public Raptor + S6\nblend added at 0.5", 0.958, 0.946, "blend"),
    ("Synovitis composite rule", 0.959, 0.947, "rules"),
    ("four-member rank blend\n(own, Raptor, S6, R7)", 0.960, 0.948, "blend"),
    ("Lateral OA exclusion\n(public read-out) + mirror arm", 0.961, 0.949, "rules"),
    ("nartaa S6-all swap", 0.962, 0.951, "blend"),
]
GCOL = {"model": BLUE, "labels": GREEN, "blend": ORANGE, "rules": PURPLE}


def fig2_waterfall():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(WATERFALL))
    pub = [w[1] for w in WATERFALL]
    pri = [w[2] for w in WATERFALL]
    base = 0.88
    for i, (lab, p, q, g) in enumerate(WATERFALL):
        lo = pub[i - 1] if i else base
        ax.bar(i, p - lo, bottom=lo, width=0.62, color=GCOL[g], alpha=0.9)
        ax.text(i, p + 0.0012, f"{p:.3f}", ha="center", fontsize=7.4, color=INK)
    ax.plot(x, pri, color=INK, lw=1.2, ls="--", marker="o", ms=3.5, label="private score of the same submission")
    for i, q in enumerate(pri):
        ax.text(i, q - 0.0035, f"{q:.3f}", ha="center", fontsize=6.8, color=INK, alpha=.8)
    ax.set_xticks(x, [w[0] for w in WATERFALL], rotation=38, ha="right", fontsize=7.4)
    ax.set_ylim(0.88, 0.968)
    ax.set_ylabel("macro ROC AUC")
    for g, c, lab in (("model", BLUE, "model / input"), ("labels", GREEN, "labels"), ("blend", ORANGE, "ensembling"), ("rules", PURPLE, "definition rules / exclusions")):
        ax.bar([0], [0], bottom=[0.88], color=c, label=lab)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("What moved the score: each bar is one change on top of the previous best (public leaderboard)",
                 fontsize=10.2, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "fig1_waterfall.png")
    lines = ["| step | change | public | private |", "|---|---|---|---|"]
    for i, (lab, p, q, g) in enumerate(WATERFALL, 1):
        lines.append(f"| {i} | {lab.replace(chr(10), ' ')} | {p:.3f} | {q:.3f} |")
    (ROOT / "notes/writeup/table2_waterfall.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ fig 4: label ladder
def fig4_labels():
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    tr = pd.read_csv(ROOT / "data/train.csv")
    g = tr.dropna(subset=T).set_index("StudyInstanceUID")[T]

    def macro(path):
        d = pd.read_csv(ROOT / path).set_index("StudyInstanceUID").loc[g.index]
        return float(np.mean([auc(g[t], d[t]) for t in T]))
    m = {k: macro(p) for k, p in (("v06", "data/labels/labels_v06.csv"), ("v07", "data/labels/labels_v07.csv"),
                                   ("v10", "data/labels/labels_v10_exp103_r7.csv"), ("v11", "data/labels/labels_v11_exp108_r7.csv"),
                                   ("v13D", "team/a3/labels/labels_v13D_r7.csv"), ("v14", "team/a3/labels/labels_v14_r7.csv"))}
    print("label macro AUC from files:", {k: round(v, 4) for k, v in m.items()})
    versions = [("public regex rules", .812, "reference"), ("public LLM reader", .870, "reference"),
                ("v01 rubric verdicts (5 levels)", .809, "prompt"), ("v02 probability ladders", .810, "prompt"),
                ("v03 descriptive factored extraction", .839, "prompt"),
                ("v04 mean of 3 readers", .895, "readers"), ("v06 mean of 4 readers", m["v06"], "readers"),
                ("v07 0.5 text + 0.5 image OOF", m["v07"], "teacher"), ("v10 0.3 text + 0.7 teacher OOF", m["v10"], "teacher"),
                ("v11 second round", m["v11"], "teacher"), ("v13D decoupled teacher", m["v13D"], "teacher"),
                ("v14 final round", m["v14"], "teacher")]
    cols = {"reference": GREY, "prompt": ORANGE, "readers": BLUE, "teacher": GREEN}
    names = [v[0] for v in versions]
    vals = [v[1] for v in versions]
    ax.barh(names[::-1], vals[::-1], color=[cols[v[2]] for v in versions][::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.90, color=RED, lw=1, ls="--")
    ax.text(0.901, len(versions) - 0.6, "report-only ceiling (~0.90):\nexperts read the images", color=RED, fontsize=7.8, va="top")
    ax.set_xlim(0.78, 0.985)
    ax.set_xlabel("label AUC against the 58 expert-labelled studies (macro over 12 findings)")
    ax.set_title("Label ladder: every training-target version, scored against the experts", fontsize=10.2, color=INK, loc="left")
    ax.tick_params(axis="y", labelsize=8.4)
    for lab, col in (("single-prompt reader", ORANGE), ("reader ensemble", BLUE), ("image-teacher round", GREEN), ("public reference", GREY)):
        ax.bar([0], [0], color=col, label=lab)
    ax.legend(loc="upper right", fontsize=7.8, frameon=False, bbox_to_anchor=(1.0, 0.84))
    fig.tight_layout()
    save(fig, "fig4_labels.png")


# ------------------------------------------------------------------ fig 7: five findings, three instruments
def fig7_findings():
    findings = ["Synovitis", "PF OA", "Lateral OA", "Contusion", "Lateral Meniscus"]
    npos = {"Synovitis": 27, "PF OA": 21, "Lateral OA": 11, "Contusion": 19, "Lateral Meniscus": 23}
    gold = [.816, .910, .878, .935, .907]          # final own member (10 checkpoints) on gold-58, as in Table A1
    public = [.914, .938, .920, .938, .968]
    private = [.860, .926, .944, .914, .950]
    gerr = [1.96 * hanley_se(a, npos[f], 58 - npos[f]) for a, f in zip(gold, findings)]
    x = np.arange(len(findings)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    ax.bar(x - w, gold, w, color=GREY, yerr=gerr, capsize=3, error_kw=dict(lw=1, ecolor=INK), label="58 expert studies, own models (95% CI)")
    ax.bar(x, public, w, color=BLUE, yerr=[0.003] * 5, capsize=3, error_kw=dict(lw=1, ecolor=INK), label="public test, column-inversion probe (+-0.003)")
    ax.bar(x + w, private, w, color=ORANGE, yerr=[0.003] * 5, capsize=3, error_kw=dict(lw=1, ecolor=INK), label="private test, same probe")
    for xi, (g, p, q) in zip(x, zip(gold, public, private)):
        ax.text(xi, p + 0.006, f"{p:.3f}", ha="center", fontsize=7.2, color=INK)
        ax.text(xi + w, q + 0.006, f"{q:.3f}", ha="center", fontsize=7.2, color=INK)
    ax.set_xticks(x, findings)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("ROC AUC")
    ax.set_title("The five weak findings on three instruments: 58 studies cannot resolve what a probe can",
                 fontsize=10.2, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=7.8, frameon=False)
    fig.tight_layout()
    save(fig, "fig6_findings.png")


# ------------------------------------------------------------------ fig 8: public vs private
def fig8_private(df):
    d = df[df.public >= 0.9].copy()
    d["track"] = d.desc.map(tag_track)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for track, col, lab in (("own", BLUE, "own models only"), ("pub", ORANGE, "blended with public models")):
        s = d[d.track == track]
        ax.scatter(s.public, s.private, s=22, color=col, alpha=.55, lw=0, label=lab)
    x = np.linspace(0.9, 0.965, 10)
    ax.plot(x, x - 0.011, color=GREY, lw=1, ls="--")
    ax.text(0.9005, 0.9005 - 0.011 - 0.0025, "private = public - 0.011", color=GREY, fontsize=8)
    for tag, lab, dx, dy, ha in (("K12", "final blend (pick A): 0.962 / 0.951", -0.0008, 0.0014, "right"),
                                 ("PB3:", "own models only (pick B): 0.958 / 0.949", 0.0008, -0.0028, "left")):
        r = d[d.desc.str.startswith(tag)].iloc[0]
        ax.scatter([r.public], [r.private], s=140, facecolor="none", edgecolor=RED, lw=1.8, zorder=6)
        ax.annotate(lab, (r.public, r.private), xytext=(r.public + dx, r.private + dy), fontsize=8, color=RED, ha=ha)
    ax.set_xlabel("public leaderboard (30% of the hidden test)")
    ax.set_ylabel("private leaderboard (70%)")
    ax.set_xlim(0.9, 0.966); ax.set_ylim(0.884, 0.955)
    ax.set_title("All scored submissions: the private drop was uniform", fontsize=10.2, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, "fig7_private.png")


# ------------------------------------------------------------------ appendix: timeline
def figA_timeline(df):
    d = df[df.public >= 0.895].copy()   # excludes the column-inversion probes (0.878-0.891)
    d["track"] = d.desc.map(tag_track)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for track, col, lab in (("own", BLUE, "own models only"), ("pub", ORANGE, "blended with public models")):
        s = d[d.track == track]
        ax.scatter(s.date, s.public, s=14, color=col, alpha=.4, lw=0, label=lab)
        s = s.sort_values("date")
        ax.step(s.date, s.public.cummax(), where="post", color=col, lw=2.0, alpha=.9)
    ax.axhline(0.965, color=GREY, lw=1, ls="--"); ax.text(d.date.min(), 0.9655, "public gold line at close", color=GREY, fontsize=8)
    ax.set_ylim(0.88, 0.972); ax.set_ylabel("public macro ROC AUC")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title("Appendix: public score of every scored submission over the competition (lines = running best per track)",
                 fontsize=9.8, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, "figA_timeline.png")


if __name__ == "__main__":
    df = load_subs()
    a06, a14 = fig1_thesis()
    fig2_waterfall()
    fig4_labels()
    fig7_findings()
    fig8_private(df)
    figA_timeline(df)
    print("v06 per finding:", dict(zip(T, [round(a, 3) for a in a06])))
    print("v14 per finding:", dict(zip(T, [round(a, 3) for a in a14])))
