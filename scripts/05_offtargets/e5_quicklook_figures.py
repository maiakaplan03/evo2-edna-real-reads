#!/usr/bin/env python3
"""(point = median AUROC over 5 re-splits; the CI is of the re-split-averaged score, so a point can sit a hair outside it -> bars clipped at 0)
e5_quicklook_figures.py -- s37, 2026-09-29. Quick-look figures of the e4 results that exist
(hidden, stop, caught_missed complete; evident: off401/off404/off368 vs EUK complete, the three
sub-label comparisons not run). Classifier = ridge for every score (same fair learner for all),
split = cluster (97 % clusters never straddle train/test). Error bars = bootstrap 95 % CI."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
H = Path(__file__).resolve().parent; OUT = H.parent/"figures"/"offtarget"; OUT.mkdir(parents=True, exist_ok=True)
d = pd.concat([pd.read_csv(H/f"results_{f}.tsv", sep="\t") for f in ("hidden", "stop", "evident", "caught_missed")])
d = d.drop_duplicates(["comparison", "T", "lib", "band", "feature", "classifier", "split"])   # hidden has identical resumed rows (checked: max diff 0.0)
d = d[(d.split == "cluster") & (d.classifier == "ridge")]
EVO = ["ppl", "nll", "acgt", "embA", "embD"]; SEQ = ["gc", "kmer4", "onehot"]; DB = ["midori_id", "prof_score0"]
NAME = {"ppl": "perplexity", "nll": "per-base NLL", "acgt": "per-base A/C/G/T", "embA": "embedding A", "embD": "embedding D",
        "gc": "GC", "kmer4": "4-mer counts", "onehot": "raw read (one-hot)", "midori_id": "MIDORI identity", "prof_score0": "COI profile score"}
COL = {**{k: "#1f6fb4" for k in EVO}, **{k: "#e08a1e" for k in SEQ}, **{k: "#7a7a7a" for k in DB}}
import os
DROP = [x for x in os.environ.get("DROP", "").split(",") if x]; SUF = os.environ.get("SUF", "")   # variants: DROP=onehot SUF=_noOH
SEQ = [f for f in SEQ if f not in DROP]
FEATS = EVO + SEQ + DB
def panel(ax, g, title):
    g = g.set_index("feature")
    for i, f in enumerate(FEATS):
        if f not in g.index: continue
        r = g.loc[f]; ax.errorbar(r.auroc, i, xerr=[[max(r.auroc - r.ci_lo, 0)], [max(r.ci_hi - r.auroc, 0)]], fmt="o", color=COL[f], ms=6, capsize=3, lw=1.5)
    nl, nh = g.null_lo.min(), g.null_hi.max(); ax.axvspan(nl, nh, color="#dddddd", zorder=0)
    ax.axvline(0.5, color="k", lw=0.6, ls=":")
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels([NAME[f] for f in FEATS]); ax.set_ylim(len(FEATS) - 0.5, -0.5)   # fixed limits, not invert (invert toggles on shared axes)
    for t, c in zip(ax.get_yticklabels(), [COL[f] for f in FEATS]): t.set_color(c)
    ax.set_xlim(0.35, 1.02); ax.set_xlabel("AUROC (off-target vs eukaryotic COI)"); r0 = g.iloc[0]
    ax.set_title(f"{title}\nn = {int(r0.n_pos):,} vs {int(r0.n_neg):,}", fontsize=10)
    ax.axhline(len(EVO) - 0.5, color="#bbbbbb", lw=0.6); ax.axhline(len(EVO) + len(SEQ) - 0.5, color="#bbbbbb", lw=0.6)
A = d[d.framing == "A"]
def sel(c, T, lib): return A[(A.comparison == c) & (A["T"] == T) & (A.lib == lib)]
# FIG 1: the hidden reads (passed every COI check) -- the main question
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
panel(axs[0], sel("HID_prok_vs_EUK", 5, "brazil"), "HIDDEN bacterial-like vs eukaryotic COI — Brazil")
panel(axs[1], sel("HID_prok_vs_EUK", 5, "abyssal"), "HIDDEN bacterial-like vs eukaryotic COI — abyssal")
panel(axs[2], sel("STOP_vs_EUK", 5, "brazil"), "STOP (pseudogene-like) vs eukaryotic COI — Brazil")
fig.tight_layout(); [fig.savefig(OUT/f"OT1_frameworkA_hidden_stop{SUF}.{e}", dpi=200) for e in ("png", "pdf")]; plt.close(fig)
# FIG 1b: same at +-10 bits (sensitivity), side by side with +-5
fig, axs = plt.subplots(1, 4, figsize=(19, 5.2), sharey=True)
for ax, (T, lib) in zip(axs, [(5, "brazil"), (10, "brazil"), (5, "abyssal"), (10, "abyssal")]):
    panel(ax, sel("HID_prok_vs_EUK", T, lib), f"HIDDEN prok vs EUK — {lib}, grey zone ±{T} bits")
fig.tight_layout(); [fig.savefig(OUT/f"OT1b_hidden_T5_vs_T10{SUF}.{e}", dpi=200) for e in ("png", "pdf")]; plt.close(fig)
# FIG 2: sanity checks -- reads the length filter already caught, and caught vs missed bacterial
fig, axs = plt.subplots(1, 4, figsize=(19, 5.2), sharey=True)
panel(axs[0], sel("EV_off401_vs_EUK", 5, "abyssal"), "caught 401-bp class vs EUK — abyssal")
panel(axs[1], sel("EV_off404_vs_EUK", 5, "abyssal"), "caught 404-bp class vs EUK — abyssal")
panel(axs[2], sel("EV_off368_vs_EUK", 5, "abyssal"), "caught 368-bp class vs EUK — abyssal")
panel(axs[3], sel("EV_off401prok_vs_HID_prok", 5, "abyssal"), "caught bacterial (401) vs MISSED bacterial (HIDDEN)")
fig.tight_layout(); [fig.savefig(OUT/f"OT2_evident_and_caught_vs_missed{SUF}.{e}", dpi=200) for e in ("png", "pdf")]; plt.close(fig)
# FIG 3: framing B -- matched MIDORI identity bands
ORDER = ["<70", "70-80", "80-84", "84-88", "88-92", "92-95", "95-97", "97-98", "98-99", ">=99"]
SHOW = [f for f in ["ppl", "acgt", "embD", "kmer4", "onehot", "midori_id"] if f not in DROP]
STY = {"ppl": ("#9ecae1", "o"), "acgt": ("#1f6fb4", "s"), "embD": ("#08306b", "D"), "kmer4": ("#f2b36b", "^"), "onehot": ("#c0650a", "v"), "midori_id": ("#7a7a7a", "x")}
B = d[d.framing == "B"]
fig, axs = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1, 2.6]})
for ax, (c, lib, ttl) in zip(axs, [("HID_prok_vs_EUK", "brazil", "HIDDEN prok vs EUK — Brazil"), ("HID_prok_vs_EUK", "abyssal", "HIDDEN prok vs EUK — abyssal"), ("STOP_vs_EUK", "brazil", "STOP vs EUK — Brazil")]):
    g = B[(B.comparison == c) & (B["T"] == 5) & (B.lib == lib)]; bands = [b for b in ORDER if b in set(g.band)]
    for j, f in enumerate(SHOW):
        h = g[g.feature == f].set_index("band").reindex(bands); x = np.arange(len(bands)) + (j - (len(SHOW) - 1) / 2) * 0.07
        ax.errorbar(x, h.auroc, yerr=[(h.auroc - h.ci_lo).clip(lower=0), (h.ci_hi - h.auroc).clip(lower=0)], color=STY[f][0], marker=STY[f][1], ms=5, capsize=2, lw=1.3, label=NAME[f])
    npos = g.drop_duplicates("band").set_index("band").reindex(bands).n_pos.astype(int)
    ax.set_xticks(range(len(bands))); ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bands, npos)], fontsize=8)
    ax.axhline(0.5, color="k", lw=0.6, ls=":"); ax.set_ylim(0.3, 1.03); ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("identity to closest MIDORI COI (%)"); ax.set_ylabel("AUROC")
axs[2].legend(loc="lower right", fontsize=8, ncol=2)
fig.tight_layout(); [fig.savefig(OUT/f"OT3_frameworkB_identity_bands{SUF}.{e}", dpi=200) for e in ("png", "pdf")]; plt.close(fig)
print("done", sorted(p.name for p in OUT.iterdir()))
