"""Recomputes the paper's headline numbers from the per-fold files in data/ (CPU only, a few seconds).
Usage:  python reproduce_key_numbers.py        (needs numpy, pandas, scipy)"""
import os, numpy as np, pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rd = lambda f: pd.read_csv(os.path.join(D, f))

def nb_test(d, ratio=0.25):
    """Nadeau-Bengio corrected resampled t-test on per-fold differences d (fractions)."""
    n = len(d); se = np.sqrt((1/n + ratio) * d.var(ddof=1)); t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), n - 1); h = stats.t.ppf(.975, n - 1) * se
    return d.mean()*100, (d.mean()-h)*100, (d.mean()+h)*100, p
def holm(ps):
    ps = np.array(ps); o = np.argsort(ps); out = np.empty(len(ps)); run = 0
    for i, ix in enumerate(o): run = max(run, min(1, ps[ix]*(len(ps)-i))); out[ix] = run
    return out

F = rd("folds_accuracy_7.csv"); F["DBHT"] = rd("dbht_25folds_acc.csv")["acc"].values
print("== Table 1: accuracy mean +- SD over 25 folds")
for m in ["NB", "XGB", "RF", "LR", "SVM", "DBZS", "DBFT", "DBHT"]:
    print(f"  {m:5s} {F[m].mean()*100:6.2f} +- {F[m].std(ddof=1)*100:4.2f}")

print("\n== Key paired tests (Delta pp, 95% CI, raw p)")
for a, b in [("DBFT","SVM"), ("DBFT","LR"), ("SVM","LR"), ("DBHT","DBFT"), ("DBHT","SVM")]:
    print("  %-5s - %-5s %+.2f [%+.2f, %+.2f]  p=%.4f" % (a, b, *nb_test((F[a]-F[b]).values)))
print("  DB-HT vs the 7 other models (Holm over 7):")
others = ["DBFT","SVM","LR","NB","RF","XGB","DBZS"]; r = [nb_test((F.DBHT-F[o]).values) for o in others]; h = holm([x[3] for x in r])
for o, x, ph in zip(others, r, h): print("    DB-HT - %-5s %+.2f [%+.2f, %+.2f]  pHolm=%.4f" % (o, *x[:3], ph))

print("\n== All 21 pairs among the 7 pre-planned models: number with Holm p < 0.05")
P = rd("pairwise_7models.csv"); print("  ", int((P.p_holm < 0.05).sum()), "of", len(P))

print("\n== Exact truncation control for the SVM (25 folds)")
S = rd("svm_exact_truncation_folds.csv"); print("  means:", {c: round(S[c].mean()*100, 2) for c in ["full","first510","last510","ht128_382"]})
T = {"SVM full - first510": (S.full-S.first510).values, "SVM last510 - first510": (S.last510-S.first510).values,
     "SVM full - last510": (S.full-S.last510).values, "SVM full - head+tail": (S.full-S.ht128_382).values,
     "DB-FT - SVM first510": (F.DBFT-S.first510).values, "DB-HT - SVM head+tail": (F.DBHT-S.ht128_382).values}
r = [nb_test(v) for v in T.values()]; h = holm([x[3] for x in r])
for k, x, ph in zip(T, r, h): print("  %-26s %+.2f [%+.2f, %+.2f]  pHolm=%.4f" % (k, *x[:3], ph))

print("\n== Learning curve (DB-FT first 10 folds; ratio = 400 / n_train)")
L = rd("learning_curve_dbft.csv"); C = rd("learning_curve_classical.csv")
for n in [100, 200, 400, 800, 1600]:
    a = L[(L.model=="DBFT") & (L.n==n)].set_index(["rep","fold"]).acc
    b = C[(C.model=="SVM") & (C.n==n)].set_index(["rep","fold"]).acc.loc[a.index]
    d = (a-b).values; x = nb_test(d, ratio=400/n)
    print("  n=%4d DB-FT %.2f SVM %.2f  Delta %+.2f [%+.2f, %+.2f]  wins %d/10" % (n, a.mean()*100, b.mean()*100, *x[:3], (d>0).sum()))
