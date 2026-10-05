#!/usr/bin/env python3
"""p1 -- s37 2026-09-29. Redraw of s29 figC2_profile_contamination for the PI: same data
(analysis_s29/perbase_phase1_profiles.npz, phase1 H100, per-cohort mean of per-position NLL), same cohorts,
new design. Nothing recomputed. Per-base perplexity = exp(mean NLL at that position) = geometric mean.
Three variants: absolute profile, difference from the reference, and both stacked."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
H = Path(__file__).resolve().parent; Z = np.load(H.parent/"analysis_s29/perbase_phase1_profiles.npz")
OUT = H/"figures"/"profile"; OUT.mkdir(parents=True, exist_ok=True)
W = 9                                             # rolling-mean window (bp), stated in the caption
def roll(v): return np.convolve(np.pad(v, (W//2, W//2), mode="edge"), np.ones(W)/W, mode="valid")[:v.size]
C = [  # key, label, colour, line width, style
 ("C0_REF",         "Reference (C0)",          "#111111", 2.6, "-"),
 ("C1_illumina",    "C1 · sequencing error",   "#4e9a8c", 1.8, "-"),
 ("C4_briggs_low",  "C4 · aDNA damage, low",   "#f0a35e", 1.8, "-"),
 ("C4_briggs_high", "C4 · aDNA damage, high",  "#c75d0a", 2.0, "-"),
 ("C2_chimera_T2",  "C2 · chimera, close parents (T2)", "#8dbbe6", 1.8, "-"),
 ("C2_chimera_T6",  "C2 · chimera, distant parents (T6)", "#1f5fa8", 2.0, "-"),
 ("C3_insert",      "C3 · 16S insert",          "#9b3fb5", 2.2, "-")]
x = np.arange(1, 312)
prof = {k: np.exp(roll(Z[f"{k}__mean"])) for k, *_ in C}
mean51 = {k: float(np.exp(Z[f"{k}__mean"][50:].mean())) for k, *_ in C}
n = {k: int(Z[f"{k}__n"][0]) for k, *_ in C}
INK, SUB, GRID, BG = "#1f2328", "#5b616b", "#e7e7e3", "white"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c9cbc6", "axes.labelcolor": SUB,
                     "xtick.color": SUB, "ytick.color": SUB, "figure.facecolor": BG, "axes.facecolor": BG})
def style(ax):
    ax.axvspan(0, 50, color="#f1f0ec", lw=0, zorder=0)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlim(0, 312)
def absolute(ax, legend=True):
    style(ax)
    q25, q75 = np.exp(roll(Z["C0_REF__q25"])), np.exp(roll(Z["C0_REF__q75"]))
    ax.fill_between(x, q25, q75, color="#111111", alpha=.08, lw=0, zorder=1, label="Reference, middle 50 % of reads")
    for k, lab, col, lw, ls in C[::-1]:
        ax.plot(x, prof[k], color=col, lw=lw, ls=ls, zorder=3 if k != "C0_REF" else 4,
                label=f"{lab}   (mean {mean51[k]:.2f})")
    ax.set_ylim(1.0, 4.6); ax.set_ylabel("per-base perplexity", fontsize=12)
    ax.text(25, 4.45, "warm-up\n(excluded)", ha="center", va="top", fontsize=9, color=SUB, style="italic")
    if legend:
        h, l = ax.get_legend_handles_labels(); order = [len(C)-1-i for i in range(len(C))] + [0]
        order = [i for i in range(len(h)) if l[i].startswith("Reference (C0)")] + \
                [i for i in range(len(h)-1, -1, -1) if not l[i].startswith("Reference")]
        ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  frameon=False, fontsize=10, handlelength=2.2, borderaxespad=0)
def delta(ax, legend=False):
    style(ax); ax.axhline(0, color="#111111", lw=1.6, zorder=4)
    for k, lab, col, lw, ls in C[1:]:
        ax.plot(x, prof[k] - prof["C0_REF"], color=col, lw=lw, ls=ls, zorder=3, label=lab)
    ax.set_ylabel("difference from reference\n(per-base perplexity)", fontsize=12)
    ax.set_ylim(-0.7, 2.1)
    ax.text(52, 0.05, "reference", ha="left", va="bottom", fontsize=9, color="#111111")
    if legend: ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10, borderaxespad=0)
CAP = ("Evo 2 (phase 1, H100) scored the simulated cohorts; each line is one cohort's per-base perplexity along the 312-bp read\n"
       f"(geometric mean over ~{min(n.values()):,}–{max(n.values()):,} reads per cohort, {W}-bp rolling mean). Grey band = middle 50 % of reference reads.\n"
       "Shaded start = first 50 positions, where the model has too little context and every cohort is high; the mean in the legend\n"
       "uses positions 51–311. Lower panel = each cohort minus the reference (above 0 = more surprising than real COI).")
def save(fig, name):
    for e in ("png", "pdf"): fig.savefig(OUT/f"{name}.{e}", dpi=220, bbox_inches="tight")
    plt.close(fig)
# V1 stacked
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1], "hspace": .08})
absolute(a1); delta(a2); a2.set_xlabel("position in read (bp)", fontsize=12)
fig.suptitle("How each kind of contamination changes Evo 2's surprise along the read", x=.125, ha="left", fontsize=15, color=INK, y=.95)
fig.text(.125, .015, CAP, fontsize=9.3, color=SUB, va="top", linespacing=1.5)
save(fig, "P1_profile_contamination_stacked")
# V2 absolute only
fig, ax = plt.subplots(figsize=(13, 5.6)); absolute(ax); ax.set_xlabel("position in read (bp)", fontsize=12)
fig.suptitle("How each kind of contamination changes Evo 2's surprise along the read", x=.125, ha="left", fontsize=15, color=INK, y=.98)
fig.text(.125, -0.03, CAP.replace(" Lower panel = each cohort minus the reference (above 0 = more surprising than real COI).", ""), fontsize=9.3, color=SUB, va="top", linespacing=1.5)
save(fig, "P1_profile_contamination_absolute")
# V3 difference only
fig, ax = plt.subplots(figsize=(13, 5.2)); delta(ax, legend=True); ax.set_xlabel("position in read (bp)", fontsize=12)
fig.suptitle("Where each kind of contamination departs from real COI", x=.125, ha="left", fontsize=15, color=INK, y=.98)
fig.text(.125, -0.03, CAP.replace("Lower panel = each", "Each line = each"), fontsize=9.3, color=SUB, va="top", linespacing=1.5)
save(fig, "P1_profile_contamination_difference")
print({k: (n[k], round(mean51[k], 3)) for k in n})
