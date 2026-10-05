#!/usr/bin/env python3
"""p2 -- s37 2026-09-29. Per-position Evo 2 profile of the REAL off-target groups, same design as p1.
Data: offtarget_eval/features/nll.npy (s29 H200 forward run; per position -log p(observed base); checked in e3
against the scorer's own Tier-1 values), groups from reads_table.tsv (s35 labels, grey zone +-5 bits).
Per-base perplexity at a position = exp(mean NLL over the group's reads) = geometric mean, as in p1.
One figure per library, because each library is compared with ITS OWN eukaryotic COI reads.
Line style = how the pipeline treats the group: solid = passed every COI check (HIDDEN) or failed only the
stop check (STOP); dashed = caught by the length filter (evident). Colour = what the labeller says it is."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
H = Path(__file__).resolve().parent; E = H/"offtarget_eval"; OUT = H/"figures"/"profile"; OUT.mkdir(parents=True, exist_ok=True)
t = pd.read_csv(E/"reads_table.tsv", sep="\t", usecols=["read_id", "lib", "group_T5"], low_memory=False)
rows = pd.read_csv(E/"features/rows.tsv", sep="\t"); frow = dict(zip(rows.read_id, range(len(rows))))
t["frow"] = t.read_id.map(frow); t = t[t.frow.notna()]; t["frow"] = t.frow.astype(int)
NLL = np.load(E/"features/nll.npy", mmap_mode="r"); L = NLL.shape[1]; x = np.arange(1, L + 1)
W = 9
def roll(v): return np.convolve(np.pad(v, (W//2, W//2), mode="edge"), np.ones(W)/W, mode="valid")[:v.size]
def stats(idx):
    M = np.asarray(NLL[np.sort(idx)], dtype=np.float64)
    return M.mean(0), np.percentile(M, 25, axis=0), np.percentile(M, 75, axis=0), float(np.exp(M[:, 50:].mean()))
SPEC = {  # group -> label, colour, style, width
 "EUK":        ("Eukaryotic COI (reference)",               "#111111", "-",  2.6),
 "HID_prok":   ("HIDDEN · bacterial-like",                  "#d7301f", "-",  2.1),
 "HID_grey":   ("HIDDEN · grey (undecided)",                "#8c6bb1", "-",  1.9),
 "HID_notCOI": ("HIDDEN · not COI",                         "#d4a017", "-",  1.6),
 "STOP":       ("STOP · pseudogene-like",                   "#1b9e77", "-",  2.1),
 "EV_off401":  ("caught 401 class · bacterial COX1 homolog", "#f08a6c", "--", 2.0),
 "EV_off404":  ("caught 404 class · mostly not COI",        "#b8860b", "--", 2.0),
 "EV_off368":  ("caught 368 class · mostly euk-labelled",   "#3a7bd5", "--", 2.0)}
PLOTS = {"brazil": ("brazil", ["EUK", "HID_prok", "HID_grey", "HID_notCOI", "STOP"]),
         "abyssal": ("abyssal", ["EUK", "HID_prok", "HID_grey", "HID_notCOI", "EV_off401", "EV_off404", "EV_off368"]),
         "abyssal_hiddenonly": ("abyssal", ["EUK", "HID_prok", "HID_grey", "HID_notCOI"]),     # split of the busy abyssal plot
         "abyssal_caughtonly": ("abyssal", ["EUK", "EV_off401", "EV_off404", "EV_off368"]),
         "brazil_no_notCOI": ("brazil", ["EUK", "HID_prok", "HID_grey", "STOP"])}             # without the 67 noisy not-COI reads
LIBNAME = {"brazil": "Brazilian soil", "abyssal": "Abyssal sediment"}
INK, SUB, GRID = "#1f2328", "#5b616b", "#e7e7e3"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c9cbc6", "axes.labelcolor": SUB,
                     "xtick.color": SUB, "ytick.color": SUB, "figure.facecolor": "white", "axes.facecolor": "white"})
def style(ax):
    ax.axvspan(0, 50, color="#f1f0ec", lw=0, zorder=0); ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlim(0, L + 1)
summary = []
for pname, (lib, groups) in PLOTS.items():
    S = {}
    for g in groups:
        idx = t[(t.lib == lib) & (t.group_T5 == g)].frow.values
        if len(idx) == 0: continue
        S[g] = (*stats(idx), len(idx))
        if pname == lib: summary.append(dict(lib=lib, group=g, n=len(idx), mean_ppl_51_311=S[g][3]))
    ref = np.exp(roll(S["EUK"][0]))
    def absolute(ax):
        style(ax)
        ax.fill_between(x, np.exp(roll(S["EUK"][1])), np.exp(roll(S["EUK"][2])), color="#111111", alpha=.08, lw=0, zorder=1)
        for g in S:
            lab, col, ls, lw = SPEC[g]; n = S[g][4]
            ax.plot(x, np.exp(roll(S[g][0])), color=col, ls=ls, lw=lw if n >= 50 else 1.1, zorder=4 if g == "EUK" else 3,
                    alpha=1 if n >= 50 else .7, label=f"{lab}   n = {n:,} · mean {S[g][3]:.2f}")
        ax.set_ylabel("per-base perplexity", fontsize=12)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10, borderaxespad=0, handlelength=2.6)
    def delta(ax):
        style(ax); ax.axhline(0, color="#111111", lw=1.6, zorder=4)
        ax.text(L - 2, -0.06, "eukaryotic COI", ha="right", va="top", fontsize=9, color="#111111")
        for g in S:
            if g == "EUK": continue
            lab, col, ls, lw = SPEC[g]; n = S[g][4]
            ax.plot(x, np.exp(roll(S[g][0])) - ref, color=col, ls=ls, lw=lw if n >= 50 else 1.1, alpha=1 if n >= 50 else .7, zorder=3)
        ax.set_ylabel("difference from\neukaryotic COI", fontsize=12)
    CAP = (f"{LIBNAME[lib]} eDNA reads scored by Evo 2 (H200 run, 312-bp windows). Each line = one group's per-base perplexity\n"
           f"(geometric mean over the group's reads, {W}-bp rolling mean). Grey band = middle 50 % of the eukaryotic COI reads of the same library.\n"
           "Solid = reads the pipeline keeps (HIDDEN passed every COI check; STOP failed only the stop-codon check). Dashed = reads the\n"
           "length filter already removes. Colour = the s35 label (grey zone ±5 bits). Thin, faded line = fewer than 50 reads.\n"
           "Shaded start = first 50 positions (too little context); legend means use positions 51–311.")
    for variant in ("stacked", "absolute", "difference"):
        if variant == "stacked":
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.5, 9.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1], "hspace": .08})
            absolute(a1); delta(a2); axl = a2; cap = CAP + " Lower panel = each group minus eukaryotic COI."
        else:
            fig, axl = plt.subplots(figsize=(13.5, 5.6)); cap = CAP
            if variant == "absolute": absolute(axl)
            else:
                delta(axl); h = [plt.Line2D([], [], color=SPEC[g][1], ls=SPEC[g][2], lw=SPEC[g][3]) for g in S if g != "EUK"]
                axl.legend(h, [f"{SPEC[g][0]}   n = {S[g][4]:,}" for g in S if g != "EUK"], loc="upper left",
                           bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10, borderaxespad=0, handlelength=2.6)
        axl.set_xlabel("position in read (bp)", fontsize=12)
        fig.suptitle(f"{LIBNAME[lib]}: how each kind of off-target read changes Evo 2's surprise along the read",
                     x=.125, ha="left", fontsize=15, color=INK, y=.95 if variant == "stacked" else .99)
        fig.text(.125, .015 if variant == "stacked" else -0.03, cap, fontsize=9.3, color=SUB, va="top", linespacing=1.5)
        for e in ("png", "pdf"): fig.savefig(OUT/f"P2_profile_offtargets_{pname}_{variant}.{e}", dpi=220, bbox_inches="tight")
        plt.close(fig)
s = pd.DataFrame(summary); s.to_csv(OUT/"P2_group_means.tsv", sep="\t", index=False); print(s.round(3).to_string())
