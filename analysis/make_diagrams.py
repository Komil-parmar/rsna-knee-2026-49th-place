"""Schematic figures for the write-up: fig3 system diagram (pipeline + blend + rules + label loop), fig5 model.
Run: py -3.14 notes/writeup/make_diagrams.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notes/writeup/figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#1f2937", "#2563eb", "#ea580c", "#16a34a", "#9ca3af", "#dc2626", "#7c3aed"
LIGHT = {BLUE: "#dbeafe", ORANGE: "#ffedd5", GREEN: "#dcfce7", GREY: "#f3f4f6", RED: "#fee2e2", PURPLE: "#ede9fe", INK: "#e5e7eb"}
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.facecolor": "white"})


def box(ax, x, y, w, h, title, body="", color=BLUE, fs=8.4, tfs=9.4, lw=1.4, radius=0.12, tpad=0.15, gap=0.30):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}", fc=LIGHT[color], ec=color, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - tpad, title, ha="center", va="top", fontsize=tfs, color=INK, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - tpad - gap, body, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.33)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.4, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig3_system():
    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 7.6); ax.axis("off")
    # ---- row 1: pixels
    y1 = 5.8; h1 = 1.6
    box(ax, 0.2, y1, 2.0, h1, "Raw DICOM", "570 GB, 4,407 studies\n24,371 series\n819k slices", GREY, fs=8.2)
    box(ax, 2.55, y1, 2.75, h1, "Preprocess next to the data", "decode, sort by position,\nleft-knee convention, 160 mm FOV,\n512 px uint8 in 8 shards", BLUE, fs=8.0, tfs=9.0)
    box(ax, 5.65, y1, 2.45, h1, "All-slice cache", "336 px, 130 mm crop,\nevery slice of every series,\nplane + contrast tags", BLUE, fs=8.2)
    box(ax, 8.45, y1, 1.85, h1, "5-fold training", "folds grouped by\nscanner fingerprint\n(83 groups)", GREEN, fs=8.2)
    box(ax, 10.65, y1, 1.75, h1, "Out-of-fold\npredictions", "", GREEN, tfs=9.0)
    for x0, x1 in ((2.2, 2.55), (5.3, 5.65), (8.1, 8.45), (10.3, 10.65)):
        arrow(ax, x0, y1 + h1 / 2, x1, y1 + h1 / 2)
    # ---- row 2: labels
    y2 = 3.6; h2 = 1.55
    box(ax, 0.2, y2, 2.0, h2, "Radiology reports", "4,407 reports,\n9 languages, read\nuntranslated; none at test", GREY, fs=8.0)
    box(ax, 2.55, y2, 2.75, h2, "LLM report readers", "descriptive extraction:\nmention class x severity -> p;\nfour readers averaged = v06", ORANGE, fs=8.0)
    box(ax, 5.65, y2, 2.45, h2, "Label round", "0.3 x text + 0.7 x teacher,\ncross-target decoupling;\nv10 -> v14, always anchored", ORANGE, fs=8.0)
    arrow(ax, 2.2, y2 + h2 / 2, 2.55, y2 + h2 / 2)
    arrow(ax, 5.3, y2 + h2 / 2, 5.65, y2 + h2 / 2)
    arrow(ax, 11.3, y1, 7.5, y2 + h2, color=GREEN, rad=0.42)             # OOF -> label round (arcs above the red box)
    ax.text(9.9, 5.62, "teacher", fontsize=8, color=GREEN, ha="center")
    arrow(ax, 6.5, y2 + h2, 9.2, y1, color=ORANGE, rad=-0.22)             # labels -> training
    ax.text(7.55, 5.55, "training targets", fontsize=8, color=ORANGE, ha="center")
    # the 58 expert studies: validation only
    box(ax, 8.45, y2, 3.95, h2, "58 expert-labelled studies", "never trained on; used only to score\nlabel versions and to screen members\n(paired noise ~0.0125 macro)", RED, fs=8.0)
    # ---- row 3: inference
    y3 = 0.2; h3 = 2.95
    box(ax, 0.2, y3, 12.2, h3, "Inference notebook, 9 h, offline, re-runs the preprocessing on the hidden DICOMs (decoder wheels bundled)", "", PURPLE, tfs=9.4)
    yb = y3 + 1.15; hb = 1.3
    box(ax, 0.45, yb, 3.0, hb, "Own models  w = 0.4", "10 checkpoints: 2 backbones x 5 folds,\nwhole-study reads (the own-only pick\nuses 20 and two slice grids)", BLUE, fs=7.9, tfs=8.8)
    box(ax, 3.7, yb, 2.8, hb, "Raptor  w = 0.2", "nartaa: CoAtNet-2 at 384 px\npretrained on OAI knee MRI,\n+ 0.35 x its 224 px mirror arm", ORANGE, fs=7.9, tfs=8.8)
    box(ax, 6.75, yb, 2.65, hb, "S6-all  w = 0.2", "nartaa: ConvNeXt trained on\nall 4,407 studies with its own\nreport labels", ORANGE, fs=7.9, tfs=8.8)
    box(ax, 9.65, yb, 2.5, hb, "R6  w = 0.2", "heliosli: compact ConvNeXt-S\n2.5D reader, 4 folds, flip TTA", ORANGE, fs=7.9, tfs=8.8)
    ax.text(6.3, y3 + 0.95, "per-finding percentile-rank blend  ->  exclusions from public read-outs (Lateral OA: own + R6 only; Fracture: without S6)",
            fontsize=8.0, color=INK, ha="center", va="top")
    ax.text(6.3, y3 + 0.6, "->  three definition rules (Synovitis composite, scanner-habit removal, OA decoupling)  ->  submission.csv",
            fontsize=8.0, color=INK, ha="center", va="top")
    arrow(ax, 5.48, y1, 5.48, y3 + h3, color=BLUE, ls=":", rad=0.0)
    ax.text(5.6, 3.3, "same preprocessing + own checkpoints", fontsize=7.6, color=BLUE, ha="left")
    save(fig, "fig2_system.png")


def fig5_model():
    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 6.0); ax.axis("off")
    yt = 3.85; ht = 1.85
    box(ax, 0.2, yt, 1.75, ht, "One study", "3-14 series: sagittal,\ncoronal, axial; fluid-\nsensitive or not;\n~190 slices", GREY, fs=8.0)
    box(ax, 2.25, yt, 2.45, ht, "Bag of 32 triplets", "random slices from any series;\nneighbours z-1, z, z+1 as the\n3 channels; 336 x 336 px\n[32, 3, 336, 336]", BLUE, fs=8.0)
    box(ax, 5.0, yt, 2.3, ht, "Shared backbone", "ConvNeXtV2-Tiny or ConvNeXt-\nTiny (ImageNet-22k), fully\nfine-tuned, pooled\n-> [32, 384]", BLUE, fs=8.0)
    box(ax, 7.6, yt, 2.1, ht, "+ embeddings", "plane (sag / cor / ax)\ncontrast (fluid, fat-sat)\nrelative position in series\n-> [32, 384]", BLUE, fs=8.0)
    box(ax, 10.0, yt, 2.0, ht, "Transformer mixer", "2 layers, 6 heads;\nevery slice attends to\nevery other slice\n-> [32, 384]", BLUE, fs=8.0)
    for x0, x1 in ((1.95, 2.25), (4.7, 5.0), (7.3, 7.6), (9.7, 10.0)):
        arrow(ax, x0, yt + ht / 2, x1, yt + ht / 2)
    yd = 1.55; hd = 1.75
    box(ax, 7.6, yd, 2.1, hd, "12 finding queries", "one learned query per\nfinding, cross-attention\nover the 32 tokens:\neach finding picks its slices", GREEN, fs=8.0)
    box(ax, 10.0, yd, 2.0, hd, "12 probabilities", "[12]; BCE on soft report\nlabels, confidence-weighted;\nEMA weights (0.999)", GREEN, fs=8.0)
    arrow(ax, 11.0, yt, 8.65, yd + hd, rad=-0.3)
    arrow(ax, 9.7, yd + hd / 2, 10.0, yd + hd / 2)
    box(ax, 0.2, yd, 3.1, hd, "Training recipe", "40-epoch cosine, 5% warm-up,\n2 head-only epochs, lr 1e-4 / 3e-4,\nweight decay 0.05; TrivialAugment on\n1/4 of bags + acquisition-style augs;\nthin 3D series subsampled", ORANGE, fs=8.0)
    box(ax, 3.55, yd, 3.1, hd, "Inference: whole study", "all slices, up to 96-160 per series,\nstaggered grids averaged; 5 folds x\n2 backbones = 10 models in the blend\n(20 in the own-only pick); fp16; no TTA", ORANGE, fs=8.0)
    ax.text(0.2, 0.95, "Where the gains came from: whole-study coverage and per-finding attention over fixed slots (+0.013 and +0.006 on the expert set), the 336 px / 130 mm input (+0.011),",
            fontsize=8.1, color=INK)
    ax.text(0.2, 0.62, "a slow-decay schedule with EMA, and the cross-attention decoder. Bigger encoders and foundation backbones tied or lost at the same recipe.",
            fontsize=8.1, color=INK)
    save(fig, "fig5_model.png")


if __name__ == "__main__":
    fig3_system()
    fig5_model()
