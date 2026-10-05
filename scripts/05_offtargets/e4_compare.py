#!/usr/bin/env python3
"""
e4_compare.py -- s37, 2026-09-29.  RUN ON THE MAC, one family per call (each writes its own results file,
and is resumable: rerunning skips rows already written):
    caffeinate python3 e4_compare.py hidden        HIDDEN off-targets vs eukaryotic COI (labels +-5 and +-10 bits)
    caffeinate python3 e4_compare.py stop          pseudogene-like STOP reads vs eukaryotic COI (Brazil)
    caffeinate python3 e4_compare.py evident       off-targets the length filter caught (off401/404/368) vs eukaryotic COI
    caffeinate python3 e4_compare.py caught_missed caught bacterial (off401) vs missed bacterial (HIDDEN prok), abyssal

THE QUESTION IN EVERY ROW: given one read from group POS and one from group NEG (same library), how often does
this score rank the POS read higher?  = AUROC. 0.5 = cannot tell them apart; 1 = always; below 0.5 = reversed.
FRAMINGS  A = all reads of the two groups.  B = only reads in the same band of identity to full MIDORI
          (both sides >= MIN_N reads), so "further from the database" cannot explain the separation.
SCORES    Evo 2: ppl, embA, embD, nll, acgt, codon3.   Competitors: gc, kmer4, midori_id, prof_score0, onehot (raw read).
          (the length/stop/profile FILTERS are reported as a detection table in the figures step: they catch
          100 % of evident/STOP/PROFNEG and 0 % of HIDDEN by construction)
HOW A SCORE IS LEARNED (trained scores only; nothing is ever tested on a read it was trained on)
  split     'random': each read to half 0 or 1.  'cluster': each 97 % cluster to half 0 or 1 (near-copies of a
            read never sit in the other half). Both are reported; the difference = leakage.
  classifier 'ridge': features standardised on the training half, ridge regression to +1/-1, penalty chosen on the
            training half by generalised cross-validation (a fair, strong linear classifier for EVERY score,
            competitors included, so no competitor is handicapped by a weak method).
            'centroid': the novelty-ladder method (s29 w6/w9): score = cos(read, POS centroid) - cos(read, NEG
            centroid), centroids from the training half. Vectors only.
  point     median AUROC over R=5 random re-splits (each read scored by the model trained on the other half).
  CI        2.5-97.5 % of 200 bootstrap resamples (of reads, or of clusters for the cluster split) of each read's
            score averaged over the 5 re-splits.
  null      the same procedure after shuffling POS/NEG labels, 20 times (2.5 / 50 / 97.5 %).
  raw       untrained scalars only: AUROC with a direction fixed in advance (ppl +, gc +, midori_id -, prof_score0 -).
CAPS      at most CAP reads per group (random, seeded by the comparison name) -- runtime; n before/after is recorded.
"""
import sys, time, hashlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN rows are expected for groups too small to split; they are written as NaN
from pathlib import Path
T0 = time.time()
HERE = Path(__file__).resolve().parent; FE = HERE/"features"
FAMILY = sys.argv[1]; assert FAMILY in ("hidden", "stop", "evident", "caught_missed")
CAP, MIN_N, R, NBOOT, NNULL = 3000, 30, 5, 200, 20
if len(sys.argv) > 2 and sys.argv[2] == "--smoke": CAP, R, NBOOT, NNULL, FE = 300, 2, 20, 3, HERE/"features_smoke"
OUTF = HERE/(f"results_{FAMILY}.tsv" if FE.name == "features" else f"smoke_results_{FAMILY}.tsv")
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
def seed(name): return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")
# ---------------- data
t = pd.read_csv(HERE/"reads_table.tsv", sep="\t", low_memory=False).drop(columns="seq")
rows = pd.read_csv(FE/"rows.tsv", sep="\t"); frow = dict(zip(rows.read_id, range(len(rows))))
t["frow"] = t.read_id.map(frow); t = t[t.frow.notna()].copy(); t["frow"] = t.frow.astype(int)
# clusters (e2 a)
cl = {}
for line in open(HERE/"clusters_97.uc"):
    f = line.rstrip("\n").split("\t")
    if f[0] in ("S", "H"): cl[f[8]] = int(f[1])
t["cluster"] = t.read_id.map(cl); assert t.cluster.notna().all(), "a read has no cluster -- rerun e2"
# MIDORI identity for the reads never searched in s26 (e2 b), same rule as s26: best id among hits with qcov >= 90
h = pd.read_csv(HERE/"hits_midori_missing.tsv", sep="\t", header=None,
                names=["query", "target", "id", "alnlen", "mism", "opens", "ql", "tl", "qcov", "tcov"])
h = h[(h.target != "*") & (h.qcov >= 90)]
t["midori_id"] = t.midori_id.fillna(t.read_id.map(h.groupby("query").id.max()))
t["midori_id_filled"] = t.midori_id.fillna(59.0)          # no hit >= 60 % (qcov >= 90) -> below every band edge
BANDS = [(0, 70, "<70"), (70, 80, "70-80"), (80, 84, "80-84"), (84, 88, "84-88"), (88, 92, "88-92"),
         (92, 95, "92-95"), (95, 97, "95-97"), (97, 98, "97-98"), (98, 99, "98-99"), (99, 101, ">=99")]
t["band"] = pd.cut(t.midori_id_filled, [b[0] for b in BANDS] + [101], right=False, labels=[b[2] for b in BANDS]).astype(str)
VEC = {k: np.load(FE/f"{k}.npy", mmap_mode="r") for k in ("embA", "embD", "nll", "acgt", "codon3", "kmer4", "onehot") if (FE/f"{k}.npy").exists()}
SCAL = {"ppl": ("ppl_s29", +1), "gc": ("gc", +1), "midori_id": ("midori_id_filled", -1), "prof_score0": ("prof_score0_full", -1)}   # e1b: stored where it existed, same formula recomputed elsewhere
def feat(name, idx):
    if name in SCAL: return t[SCAL[name][0]].values[idx].astype(np.float64)[:, None]
    X = np.asarray(VEC[name][t.frow.values[idx]], dtype=np.float64).reshape(len(idx), -1)
    if name == "kmer4": X = X / 309.0
    return X
# ---------------- statistics
def auroc(p, n):
    p = p[np.isfinite(p)]; n = n[np.isfinite(n)]
    if p.size < 5 or n.size < 5: return np.nan
    a = np.concatenate([p, n]); r = pd.Series(a).rank().values
    return (r[:p.size].sum() - p.size * (p.size + 1) / 2) / (p.size * n.size)
FALLBACKS = [0]
def svd_safe(Z):
    """numpy's fast SVD (LAPACK gesdd) occasionally reports 'did not converge' on finite, nearly
    rank-deficient matrices. Fallback: the same decomposition from the eigenvectors of the smaller
    Gram matrix (Z'Z or ZZ'), dropping directions with no variance. Same U,S,V up to sign; counted."""
    try:
        return np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        FALLBACKS[0] += 1; n, d = Z.shape
        if n >= d:
            ev, V = np.linalg.eigh(Z.T @ Z); o = np.argsort(ev)[::-1]; ev, V = ev[o], V[:, o]
            k = ev > ev[0] * 1e-12; S = np.sqrt(ev[k]); V = V[:, k]; U = (Z @ V) / S
            return U, S, V.T
        ev, U = np.linalg.eigh(Z @ Z.T); o = np.argsort(ev)[::-1]; ev, U = ev[o], U[:, o]
        k = ev > ev[0] * 1e-12; S = np.sqrt(ev[k]); U = U[:, k]; V = (Z.T @ U) / S
        return U, S, V.T
def fit_ridge(X, y):
    m = X.mean(0); s = X.std(0); s[s < 1e-9] = 1.0; Z = (X - m) / s; yc = y - y.mean()
    U, S, Vt = svd_safe(Z); Uy = U.T @ yc; n = len(y); best = None
    for lam in (S[0] ** 2) * np.logspace(-6, 1, 15):
        f = S ** 2 / (S ** 2 + lam); res = yc - U @ (f * Uy); df = f.sum()
        g = (res @ res / n) / max(1 - df / n, 1e-6) ** 2
        if best is None or g < best[0]: best = (g, lam)
    w = Vt.T @ ((S / (S ** 2 + best[1])) * Uy)
    return lambda Xt: ((Xt - m) / s) @ w
def fit_centroid(X, y, standardise):
    if standardise:
        m = X.mean(0); s = X.std(0); s[s < 1e-9] = 1.0; T = lambda A: A_l2(((A - m) / s))
    else: T = A_l2
    Z = T(X); cp = Z[y > 0].mean(0); cn = Z[y < 0].mean(0); cp /= np.linalg.norm(cp); cn /= np.linalg.norm(cn)
    return lambda Xt: T(Xt) @ cp - T(Xt) @ cn
def A_l2(A): return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
def crossfit(X, y, folds, clf, name):
    sc = np.full(len(y), np.nan)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        if (y[tr] > 0).sum() < 5 or (y[tr] < 0).sum() < 5 or te.sum() == 0: return sc
        if clf == "ridge": model = fit_ridge(X[tr], y[tr])
        else: model = fit_centroid(X[tr], y[tr], standardise=name not in ("embA", "embD"))
        sc[te] = model(X[te])
    return sc
def make_folds(units, rng):
    u, inv = np.unique(units, return_inverse=True); return rng.integers(0, 2, len(u))[inv]
def boot_ci(score, y, units, rng):
    u, inv = np.unique(units, return_inverse=True); members = [np.where(inv == k)[0] for k in range(len(u))]
    out = []
    for _ in range(NBOOT):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(u), len(u))])
        out.append(auroc(score[pick][y[pick] > 0], score[pick][y[pick] < 0]))
    return np.nanpercentile(out, [2.5, 97.5])
# ---------------- comparisons of this family
def grp(T): return t[f"group_T{T}"]
C = []   # (name, T, lib, framing, band, pos_mask, neg_mask, pos_label, neg_label)
def add(name, T, lib, pos, neg, pl, nl):
    L = t.lib == lib; P, N = pos & L, neg & L
    C.append((name, T, lib, "A", "all", P, N, pl, nl))
    for _, _, b in BANDS:
        Pb, Nb = P & (t.band == b), N & (t.band == b)
        if Pb.sum() >= MIN_N and Nb.sum() >= MIN_N: C.append((name, T, lib, "B", b, Pb, Nb, pl, nl))
EUK = lambda T: grp(T) == "EUK"
if FAMILY == "hidden":
    for T in (5, 10):
        for lib in ("brazil", "abyssal"):
            add("HID_prok_vs_EUK", T, lib, grp(T) == "HID_prok", EUK(T), "HID_prok", "EUK")
if FAMILY == "stop":
    add("STOP_vs_EUK", 5, "brazil", grp(5) == "STOP", EUK(5), "STOP", "EUK")
    add("STOP_euklabel_vs_EUK", 5, "brazil", (grp(5) == "STOP") & (t.label == "euk"), EUK(5), "STOP(euk label)", "EUK")
if FAMILY == "evident":
    for g in ("EV_off401", "EV_off404", "EV_off368"):
        add(f"{g}_vs_EUK", 5, "abyssal", grp(5) == g, EUK(5), g, "EUK")
    add("EV_off404_notCOI_vs_EUK", 5, "abyssal", (grp(5) == "EV_off404") & (t.label == "notCOI"), EUK(5), "EV_off404(notCOI)", "EUK")
    add("EV_off404_prok_vs_EUK", 5, "abyssal", (grp(5) == "EV_off404") & (t.label == "prok"), EUK(5), "EV_off404(prok)", "EUK")
    add("EV_off368_euklabel_vs_EUK", 5, "abyssal", (grp(5) == "EV_off368") & (t.label == "euk"), EUK(5), "EV_off368(euk label)", "EUK")
if FAMILY == "caught_missed":
    add("EV_off401prok_vs_HID_prok", 5, "abyssal", (grp(5) == "EV_off401") & (t.label == "prok"), grp(5) == "HID_prok",
        "EV_off401(prok): caught", "HID_prok: missed")
FEATS = [("ppl", "ridge"), ("gc", "ridge"), ("midori_id", "ridge"), ("prof_score0", "ridge"), ("codon3", "ridge"),
         ("embA", "ridge"), ("embA", "centroid"), ("embD", "ridge"), ("embD", "centroid"), ("nll", "ridge"),
         ("nll", "centroid"), ("acgt", "ridge"), ("acgt", "centroid"), ("kmer4", "ridge"), ("kmer4", "centroid")]
# raw-sequence competitor (e3b, added 2026-09-29): is Evo 2's acgt/nll more than a copy of the read?
if "onehot" in VEC: FEATS += [("onehot", "ridge"), ("onehot", "centroid")]
done = set()
if OUTF.exists():
    old = pd.read_csv(OUTF, sep="\t"); done = set(zip(old["comparison"], old["T"], old["lib"], old["band"], old["feature"], old["classifier"], old["split"]))   # old["T"], NOT old.T (= transpose)
log(f"family {FAMILY}: {len(C)} comparisons x {len(FEATS)} scores x 2 splits; {len(done)} rows already done")
for (name, T, lib, fr, band, P, N, pl, nl) in C:
    key = f"{name}|T{T}|{lib}|{band}"; rng0 = np.random.default_rng(seed(key))
    ip, ineg = np.where(P.values)[0], np.where(N.values)[0]; npos0, nneg0 = len(ip), len(ineg)
    if npos0 > CAP: ip = rng0.choice(ip, CAP, replace=False)
    if nneg0 > CAP: ineg = rng0.choice(ineg, CAP, replace=False)
    idx = np.concatenate([ip, ineg]); y = np.r_[np.ones(len(ip)), -np.ones(len(ineg))]
    units = {"random": np.arange(len(idx)), "cluster": t.cluster.values[idx]}
    log(f"{key}: pos {npos0} (used {len(ip)}), neg {nneg0} (used {len(ineg)}), "
        f"pos clusters {len(np.unique(units['cluster'][y > 0]))}, neg clusters {len(np.unique(units['cluster'][y < 0]))}")
    out = []
    for fname, clf in FEATS:
        X = None
        for split in ("random", "cluster"):
            if (name, T, lib, band, fname, clf, split) in done: continue
            if X is None: X = feat(fname, idx)
            if not np.isfinite(X).all():     # a score that does not exist for some reads is reported, never imputed
                out.append(dict(family=FAMILY, comparison=name, T=T, lib=lib, framing=fr, band=band, pos=pl, neg=nl,
                                n_pos=npos0, n_neg=nneg0, feature=fname, classifier=clf, split=split,
                                note=f"{int((~np.isfinite(X).all(1)).sum())} reads without this score"))
                continue
            rng = np.random.default_rng(seed(key + fname + clf + split))
            reps, S = [], []
            for r in range(R):
                s_ = crossfit(X, y, make_folds(units[split], rng), clf, fname); S.append(s_)
                reps.append(auroc(s_[y > 0], s_[y < 0]))
            ms = np.nanmean(np.vstack(S), 0); lo, hi = boot_ci(ms, y, units[split], rng)
            nulls = []
            for _ in range(NNULL):
                yp = rng.permutation(y); s_ = crossfit(X, yp, make_folds(units[split], rng), clf, fname)
                nulls.append(auroc(s_[yp > 0], s_[yp < 0]))
            row = dict(family=FAMILY, comparison=name, T=T, lib=lib, framing=fr, band=band, pos=pl, neg=nl,
                       n_pos=npos0, n_neg=nneg0, n_pos_used=len(ip), n_neg_used=len(ineg), feature=fname,
                       evo2=fname in ("ppl", "embA", "embD", "nll", "acgt", "codon3"), classifier=clf, split=split,
                       auroc=np.nanmedian(reps), auroc_reps_min=np.nanmin(reps), auroc_reps_max=np.nanmax(reps),
                       auroc_avgscore=auroc(ms[y > 0], ms[y < 0]), ci_lo=lo, ci_hi=hi,
                       null_lo=np.nanpercentile(nulls, 2.5), null_med=np.nanmedian(nulls), null_hi=np.nanpercentile(nulls, 97.5))
            if fname in SCAL:
                v = X[:, 0] * SCAL[fname][1]; row["raw_auroc_fixed_direction"] = auroc(v[y > 0], v[y < 0])
            out.append(row)
    if out:
        df = pd.DataFrame(out); df.to_csv(OUTF, sep="\t", index=False, mode="a", header=not OUTF.exists())
        best = df[df.split == "random"].sort_values("auroc", ascending=False).head(3)
        if FALLBACKS[0]: log(f"   SVD fallbacks so far: {FALLBACKS[0]}")
        log("   written; top (random split):", "; ".join(f"{a}/{b} {c:.3f}" for a, b, c in zip(best.feature, best.classifier, best.auroc)))
log(f"== e4 {FAMILY} DONE -> {OUTF.name}   (SVD fallbacks used: {FALLBACKS[0]})")
