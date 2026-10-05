#!/usr/bin/env python3
"""
k6_figures.py -- s37, 2026-09-29. Every s32 figure redrawn with the RANK standardisation (standalone), each one next
to its s32 original (comparison), and the RC-crossing figures. Titles only; explanations live in the captions doc.
Writes to nn_project/analysis_s37/figures/{s32_updated_rank, s32_old_vs_rank, rc_crossing}/  (PNG + PDF).
Run from analysis_s37/c2_rc after s32_rank/run_chain.sh has finished.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
HERE = Path(__file__).resolve().parent; NN = HERE.parents[1]
import os
OLD, NEW = NN/"analysis_s32", HERE/"s32_rank"; FIG = NN/"analysis_s37/figures"
if os.environ.get("SMOKE"): NEW, FIG = OLD, Path(os.environ["SMOKE"])        # test only: s32 data on both sides
for sub in ("s32_updated_rank", "s32_old_vs_rank", "rc_crossing"): (FIG/sub).mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
SURFACE, INK, INK2, MUTED, GRID, AXIS, MUTED2 = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#b9b8b2"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": .8,
    "grid.color": GRID, "grid.linewidth": .7, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
CHAN = ["logodds", "pmax_pobs", "entropy", "emb64", "probs4"]
CLAB = {"logodds": "log p(top1) − log p(obs)", "pmax_pobs": "p(top1) − p(obs)", "entropy": "entropy",
        "emb64": "embedding k=64", "probs4": "probs4 (A/C/G/T)"}
CCOL = {"logodds": BLUE, "pmax_pobs": ORANGE, "entropy": AQUA, "emb64": YELLOW, "probs4": MAGENTA}
ORDER = ["C1_illumina", "C2_chimera_T2", "C2_chimera_T3", "C2_chimera_T4", "C2_chimera_T5", "C2_chimera_T6", "C5_stacked_T2", "C3_insert"]
SHORT = {"C1_illumina": "C1", "C2_chimera_T2": "T2", "C2_chimera_T3": "T3", "C2_chimera_T4": "T4", "C2_chimera_T5": "T5",
         "C2_chimera_T6": "T6", "C5_stacked_T2": "C5", "C3_insert": "C3"}
TIERS = ORDER[1:6]
STD = {"old": "mean/sd standardisation (s32)", "new": "rank standardisation (s37)"}
def save(fig, sub, name):
    for ext in ("png", "pdf"): fig.savefig(FIG/sub/f"{name}.{ext}")
    plt.close(fig); print("  wrote", sub + "/" + name)
def auroc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float); a = np.concatenate([p, n]); r = pd.Series(a).rank().values
    return (r[:p.size].sum() - p.size * (p.size + 1) / 2) / (p.size * n.size)
def boot_auc(p, n, B=500, seed=0):
    r = np.random.default_rng(seed); p, n = np.asarray(p), np.asarray(n)
    v = [auroc(p[r.integers(0, len(p), len(p))], n[r.integers(0, len(n), len(n))]) for _ in range(B)]; return np.percentile(v, [2.5, 97.5])
def boot_med(x, B=1000, seed=0):
    r = np.random.default_rng(seed); x = np.asarray(x); v = [np.median(x[r.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])
D = {}
for tag, d in (("old", OLD), ("new", NEW)):
    D[tag] = dict(det=pd.read_csv(d/"j1_detection_per_seed.csv"), loc=pd.read_csv(d/"j1_localisation_per_seed.csv"),
                  per=pd.read_csv(d/"j5_per_read_errors.csv"), wm=pd.read_csv(d/"j2_worked_vs_missed.csv"),
                  bm=pd.read_csv(d/"j4_benchmark_scored.csv"))
# ======================================================================= S32 FIGURES
def f1(ax_row, dd, arms=("unmatched", "matched")):
    for ax, k in zip(ax_row, CHAN):
        g = dd["det"][dd["det"].channel == k]
        for arm, col, ls in [("unmatched", MUTED, "--"), ("matched", CCOL[k], "-")]:
            if arm not in arms: continue
            m = [g[(g.arm == arm) & (g.cohort == c)].auroc.mean() for c in ORDER]
            s = [g[(g.arm == arm) & (g.cohort == c)].auroc.std(ddof=1) for c in ORDER]
            ax.errorbar(range(8), m, yerr=s, color=col, ls=ls, marker="o", ms=4.5, lw=2, capsize=2, label=f"{arm} negatives", zorder=3)
        ax.axhline(.5, color=AXIS, lw=.8, zorder=1); ax.set_xticks(range(8)); ax.set_xticklabels([SHORT[c] for c in ORDER], fontsize=8)
        ax.set_title(CLAB[k], color=CCOL[k]); ax.set_ylim(.3, 1.0); ax.grid(axis="y", zorder=0)
    ax_row[0].set_ylabel("detection AUROC")
def fig1(tag):
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.2), sharey=True); f1(axes, D[tag]); axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Detection AUROC by cohort and channel — {STD[tag]}", x=.01, ha="left", y=1.04, fontsize=11, fontweight="bold"); return fig
def fig1_cmp():
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.2), sharey=True)
    for ax, k in zip(axes, CHAN):
        for tag, ls, col in (("old", "--", MUTED), ("new", "-", CCOL[k])):
            g = D[tag]["det"]; g = g[(g.channel == k) & (g.arm == "matched")]
            m = [g[g.cohort == c].auroc.mean() for c in ORDER]; s = [g[g.cohort == c].auroc.std(ddof=1) for c in ORDER]
            ax.errorbar(range(8), m, yerr=s, color=col, ls=ls, marker="o", ms=4.5, lw=2, capsize=2, label=STD[tag], zorder=3)
        ax.axhline(.5, color=AXIS, lw=.8); ax.set_xticks(range(8)); ax.set_xticklabels([SHORT[c] for c in ORDER], fontsize=8)
        ax.set_title(CLAB[k], color=CCOL[k]); ax.set_ylim(.3, 1.0); ax.grid(axis="y", zorder=0)
    axes[0].set_ylabel("detection AUROC (matched negatives)"); axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Detection AUROC — s32 standardisation vs rank standardisation", x=.01, ha="left", y=1.04, fontsize=11, fontweight="bold"); return fig
def f2(ax, dd, show_labels=True):
    loc = dd["loc"]; x = np.arange(5)
    ch = [loc[(loc.cohort == c) & (loc.channel == "logodds")].chance_med_abs_err.mean() for c in TIERS]
    ax.fill_between(x, ch, 75, color=GRID, alpha=.7, zorder=0); ax.plot(x, ch, color=AXIS, lw=1.4, ls=":", zorder=1)
    for k in CHAN:
        g = loc[loc.channel == k]
        if not len(g): continue
        m = np.array([g[g.cohort == c].med_abs_err.mean() for c in TIERS])
        lo = m - np.array([g[g.cohort == c].med_abs_err.min() for c in TIERS]); hi = np.array([g[g.cohort == c].med_abs_err.max() for c in TIERS]) - m
        ax.errorbar(x, m, yerr=[lo, hi], color=CCOL[k], lw=2, marker="o", ms=5.5, capsize=2, zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.2)
        if show_labels: ax.annotate(CLAB[k], xy=(x[-1], m[-1]), xytext=(x[-1] + .09, m[-1]), color=CCOL[k], fontsize=8.5, va="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["T2", "T3", "T4", "T5", "T6"]); ax.set_xlabel("C2 parental divergence  →  deeper")
    ax.set_ylabel("median |error|  (bp, lower is better)"); ax.set_xlim(-.25, 5.6); ax.set_ylim(0, 75); ax.grid(axis="y", zorder=0)
def fig2(tag):
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); f2(ax, D[tag]); ax.set_title(f"Localisation error by divergence tier — {STD[tag]}", loc="left"); return fig
def fig2_cmp():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, tag in zip(axes, ("old", "new")): f2(ax, D[tag]); ax.set_title(STD[tag], loc="left")
    fig.suptitle("Localisation error by divergence tier — s32 vs rank standardisation", x=.01, ha="left", y=1.03, fontsize=11, fontweight="bold"); return fig
def f3(ax, dd, c):
    grid = np.arange(0, 161)
    for k in CHAN:
        e = dd["per"][(dd["per"].cohort == c) & (dd["per"].channel == k)].err.values
        if len(e): ax.plot(grid, [(e <= t).mean() for t in grid], color=CCOL[k], lw=2, zorder=3, label=CLAB[k])
    ax.axvline(20, color=INK2, lw=1.1, ls="--", zorder=4, alpha=.65); ax.set_xlabel("|error|  (bp)"); ax.grid(zorder=0)
    ax.set_xlim(0, 160); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
def fig3(tag):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, c in zip(axes, ["C2_chimera_T6", "C2_chimera_T5"]): f3(ax, D[tag], c); ax.set_title({"C2_chimera_T6": "C2-deep T6", "C2_chimera_T5": "C2-deep T5"}[c], loc="left")
    axes[0].set_ylabel("fraction of reads localised within x"); axes[1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Cumulative localisation error — {STD[tag]}", x=.01, ha="left", y=1.03, fontsize=11, fontweight="bold"); return fig
def fig3_cmp():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, sharex=True)
    for i, tag in enumerate(("old", "new")):
        for j, c in enumerate(["C2_chimera_T6", "C2_chimera_T5"]):
            f3(axes[i, j], D[tag], c); axes[i, j].set_title(f"{ {'C2_chimera_T6':'T6','C2_chimera_T5':'T5'}[c]} · {STD[tag]}", loc="left")
        axes[i, 0].set_ylabel("fraction within x")
    axes[0, 1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Cumulative localisation error — s32 vs rank standardisation", x=.01, ha="left", y=1.01, fontsize=11, fontweight="bold"); return fig
def f4(axes, dd):
    for ax, (col, lab, fmt) in zip(axes, [("err", "median |error|  (bp)", None), ("hit", "reads localised within 20 bp", PercentFormatter(1.0))]):
        for c, cc, off in [("C2_chimera_T6", BLUE, -.2), ("C2_chimera_T5", ORANGE, .2)]:
            g = dd["wm"][dd["wm"].cohort == c].copy(); g["q"] = pd.qcut(g.score, 5, labels=[1, 2, 3, 4, 5])
            vals, lo, hi = [], [], []
            for q_, h in g.groupby("q", observed=True):
                x = h[col].astype(float).values; v = np.median(x) if col == "err" else x.mean()
                ci = boot_med(x) if col == "err" else np.percentile([np.random.default_rng(b).choice(x, len(x)).mean() for b in range(500)], [2.5, 97.5])
                vals.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
            ax.bar(np.arange(5) + off, vals, width=.365, color=cc, zorder=3, label={"C2_chimera_T6": "C2-deep T6", "C2_chimera_T5": "C2-deep T5"}[c],
                   yerr=[lo, hi], capsize=2, error_kw=dict(ecolor=INK2, lw=.8))
        ax.set_xticks(range(5)); ax.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"], fontsize=8)
        ax.set_xlabel("detection-score quintile"); ax.set_ylabel(lab); ax.grid(axis="y", zorder=0)
        if fmt: ax.yaxis.set_major_formatter(fmt)
    axes[0].legend(fontsize=8.5, loc="upper right")
def fig4(tag):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9)); f4(axes, D[tag])
    fig.suptitle(f"Accuracy by detection-score quintile (log-odds) — {STD[tag]}", x=.01, ha="left", y=1.04, fontsize=11, fontweight="bold"); return fig
def fig4_cmp():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    for i, tag in enumerate(("old", "new")): f4(axes[i], D[tag]); axes[i, 0].set_title(STD[tag], loc="left")
    fig.suptitle("Accuracy by detection-score quintile — s32 vs rank standardisation", x=.01, ha="left", y=1.01, fontsize=11, fontweight="bold"); return fig
METH = [("evo2_entropy", "Evo 2 · entropy"), ("evo2_score", "Evo 2 · log-odds"), ("uchime_ref", "uchime_ref\n(given the parent database)"), ("uchime_denovo", "uchime_denovo\n(uninformative here)")]
def f5(ax, dd, cols=(BLUE, AQUA)):
    bm = dd["bm"]; neg = bm[bm.is_chimera == 0]; x = np.arange(4)
    for c, cc, off, lab in [("C2_chimera_T6", cols[0], -.2, "C2-deep T6"), ("C2_chimera_T5", cols[1], .2, "C2-deep T5")]:
        g = bm[bm.cohort == c]; v = [auroc(g[m], neg[m]) for m, _ in METH]; ci = [boot_auc(g[m].values, neg[m].values) for m, _ in METH]
        b = ax.bar(x + off, v, width=.365, color=cc, zorder=3, label=lab, yerr=[[a - l for a, (l, h) in zip(v, ci)], [h - a for a, (l, h) in zip(v, ci)]],
                   capsize=2, error_kw=dict(ecolor=INK2, lw=.8))
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=6, color=INK2)
    ax.axhline(.5, color=AXIS, lw=.8); ax.set_xticks(x); ax.set_xticklabels([l for _, l in METH], fontsize=8)
    ax.set_ylabel("detection AUROC"); ax.set_ylim(0, 1.15); ax.grid(axis="y", zorder=0); ax.legend(fontsize=8.5, loc="upper left")
def fig5(tag):
    fig, ax = plt.subplots(figsize=(10, 4.4)); f5(ax, D[tag]); ax.set_title(f"Detection vs uchime on the same reads — {STD[tag]}", loc="left"); return fig
def fig5_cmp():
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
    for ax, tag in zip(axes, ("old", "new")): f5(ax, D[tag]); ax.set_title(STD[tag], loc="left")
    fig.suptitle("Detection vs uchime — s32 vs rank standardisation (uchime columns are identical by construction)", x=.01, ha="left", y=1.03, fontsize=11, fontweight="bold"); return fig
def f0(axs, dd):
    axA, axB, axC = axs; bm = dd["bm"]; neg = bm[bm.is_chimera == 0]; t6 = bm[bm.cohort == "C2_chimera_T6"]
    vals = [auroc(t6.uchime_ref, neg.uchime_ref), auroc(t6.evo2_entropy, neg.evo2_entropy)]
    cis = [boot_auc(t6.uchime_ref.values, neg.uchime_ref.values), boot_auc(t6.evo2_entropy.values, neg.evo2_entropy.values)]
    axA.barh([1, 0], vals, height=.52, color=[MUTED2, BLUE], zorder=3, xerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]],
             capsize=3, error_kw=dict(ecolor=INK2, lw=.9))
    for yy, v in zip([1, 0], vals): axA.text(v - .04, yy, f"{v:.3f}", va="center", ha="right", color="white", fontsize=13, fontweight="bold", zorder=4)
    axA.axvline(.5, color=AXIS, lw=1, ls=":"); axA.set_yticks([1, 0]); axA.set_yticklabels(["uchime_ref\n(given both parents)", "Evo 2 · entropy\n(no reference)"])
    axA.set_xlim(0, 1.06); axA.set_xlabel("detection AUROC"); axA.grid(axis="x", zorder=0); axA.tick_params(axis="y", length=0)
    axA.set_title("1 · Detecting a chimera", loc="left")
    g = dd["per"][(dd["per"].cohort == "C2_chimera_T6") & (dd["per"].channel == "logodds")]; err, est, bp = g.err.values, g.est.values, g.bp.values
    rng = np.random.default_rng(7); chance = np.concatenate([np.abs(est - bp[rng.permutation(len(bp))]) for _ in range(20)]); grid = np.arange(161)
    axB.plot(grid, [(chance <= t).mean() for t in grid], color=MUTED2, lw=2.2, ls="--", label="chance (truths permuted)")
    axB.plot(grid, [(err <= t).mean() for t in grid], color=BLUE, lw=3, label="Evo 2 · log-odds")
    f20, f4_ = (err <= 20).mean(), (err <= 4).mean(); axB.axvline(20, color=INK2, lw=1, ls=":")
    axB.scatter([20], [f20], s=60, color=BLUE, zorder=6, edgecolor=SURFACE, lw=2)
    axB.annotate(f"{f20:.0%} within 20 bp", xy=(20, f20), xytext=(34, f20 - .05), fontsize=9.5, fontweight="bold", color=BLUE)
    axB.annotate(f"{f4_:.0%} within 4 bp", xy=(4, f4_), xytext=(26, f4_ - .13), fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
    axB.set_xlim(0, 160); axB.set_ylim(0, 1); axB.grid(zorder=0); axB.yaxis.set_major_formatter(PercentFormatter(1.0)); axB.legend(loc="lower right", fontsize=8.5)
    axB.set_xlabel("distance from the true breakpoint  (bp)"); axB.set_ylabel("reads located within that distance"); axB.set_title("2 · Locating the junction", loc="left")
    q = dd["wm"][dd["wm"].cohort == "C2_chimera_T6"].copy(); q["quint"] = pd.qcut(q.score, 5, labels=[1, 2, 3, 4, 5])
    med = q.groupby("quint", observed=True).err.median(); ci = [boot_med(h.err.values) for _, h in q.groupby("quint", observed=True)]
    b = axC.bar(range(5), med.values, width=.66, color=BLUE, zorder=3, yerr=[[m - c[0] for m, c in zip(med.values, ci)], [c[1] - m for m, c in zip(med.values, ci)]],
                capsize=2, error_kw=dict(ecolor=INK2, lw=.8))
    axC.bar_label(b, fmt="%.0f bp", fontsize=9, padding=8, color=INK2, fontweight="bold")
    axC.set_xticks(range(5)); axC.set_xticklabels(["lowest", "", "middle", "", "highest"]); axC.set_xlabel("detection-score quintile  →")
    axC.set_ylabel("median error  (bp)"); axC.set_ylim(0, 60); axC.grid(axis="y", zorder=0); axC.set_title("3 · Knowing when to believe it", loc="left")
def fig0(tag):
    fig = plt.figure(figsize=(13.6, 4.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.3, 1.0], wspace=.35)
    f0([fig.add_subplot(gs[i]) for i in range(3)], D[tag])
    fig.suptitle(f"Evo 2 on simulated deep chimeras (C2-T6) — {STD[tag]}", x=.006, ha="left", y=1.04, fontsize=12, fontweight="bold"); return fig
def fig0_cmp():
    fig = plt.figure(figsize=(13.6, 9.4)); gs = fig.add_gridspec(2, 3, width_ratios=[1.02, 1.3, 1.0], wspace=.35, hspace=.45)
    for i, tag in enumerate(("old", "new")):
        axs = [fig.add_subplot(gs[i, j]) for j in range(3)]; f0(axs, D[tag]); axs[0].set_title(f"1 · Detecting · {STD[tag]}", loc="left")
    fig.suptitle("Summary figure — s32 (top) vs rank standardisation (bottom)", x=.006, ha="left", y=1.01, fontsize=12, fontweight="bold"); return fig
for name, fs, fc in [("fig0_summary", fig0, fig0_cmp), ("fig1_detection_ladder", fig1, fig1_cmp), ("fig2_localisation_by_tier", fig2, fig2_cmp),
                     ("fig3_error_distribution", fig3, fig3_cmp), ("fig4_score_vs_accuracy", fig4, fig4_cmp), ("fig5_benchmark", fig5, fig5_cmp)]:
    save(fs("new"), "s32_updated_rank", name + "_rank"); save(fc(), "s32_old_vs_rank", name + "_old_vs_rank")
