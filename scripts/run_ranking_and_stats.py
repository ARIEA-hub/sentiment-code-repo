"""
Companion script for the ECIR 2027 revision.

Two jobs:
  1) stats: corrected resampled t-tests (Nadeau & Bengio) + Holm correction + 95% CIs
     from fold-level accuracy CSVs (one column per model, 25 rows).
  2) ranking: per-fold ranking metrics (nDCG@k, AP, ROC-AUC) from saved test-fold scores.

Usage
-----
  python run_ranking_and_stats.py stats   folds_accuracy.csv [--paired LR,RF,XGB]
  python run_ranking_and_stats.py ranking scores_dir/

folds_accuracy.csv : columns LR,RF,XGB,DBZS,DBFT ; 25 rows ordered by (repeat, fold).
scores_dir/        : one file per model, <MODEL>.npz with arrays
                       y      (n_folds, n_test)  true labels
                       score  (n_folds, n_test)  P(positive) or decision score
                     Save these inside your CV loop (predict_proba / softmax of logits).
"""
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, ndcg_score

N_TEST_OVER_TRAIN = 0.25  # 400 test / 1600 train for 5-fold CV on 2,000 reviews


def corrected_t(delta, paired=True, a=None, b=None, ratio=N_TEST_OVER_TRAIN):
    """Nadeau-Bengio corrected resampled t-test. Returns mean diff, t, p, 95% CI."""
    if paired:
        n = len(delta)
        m, var = delta.mean(), delta.var(ddof=1)
    else:  # conservative: variances add (assumes non-negative correlation)
        n = len(a)
        m, var = a.mean() - b.mean(), a.var(ddof=1) + b.var(ddof=1)
    se = np.sqrt((1.0 / n + ratio) * var)
    t = m / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, t, p, (m - h, m + h)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def run_stats(path, paired_group):
    df = pd.read_csv(path)
    models = list(df.columns)
    rows = []
    for a, b in itertools.combinations(models, 2):
        both_paired = a in paired_group and b in paired_group
        x, y = df[a].to_numpy(), df[b].to_numpy()
        m, t, p, ci = corrected_t(x - y, paired=both_paired, a=x, b=y)
        sd = (x - y).std(ddof=1) if both_paired else np.sqrt(x.var(ddof=1) + y.var(ddof=1))
        rows.append(dict(A=a, B=b, kind="paired" if both_paired else "cross-run",
                         diff_pts=m, ci_lo=ci[0], ci_hi=ci[1], t=t, p_raw=p, cohens_d=m / sd))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv("pairwise_corrected_tests.csv", index=False)


def dcg_at_k(rel_sorted, k):
    rel = rel_sorted[:k]
    return float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))


def run_ranking(scores_dir):
    import glob, os
    results = []
    for f in sorted(glob.glob(os.path.join(scores_dir, "*.npz"))):
        name = os.path.splitext(os.path.basename(f))[0]
        z = np.load(f)
        Y, S = z["y"], z["score"]
        for query, flip in (("positive", False), ("negative", True)):
            per_fold = []
            for y, s in zip(Y, S):
                rel = (1 - y) if flip else y
                sc = (-s) if flip else s
                per_fold.append(dict(
                    ndcg10=ndcg_score([rel], [sc], k=10),
                    ndcg50=ndcg_score([rel], [sc], k=50),
                    ap=average_precision_score(rel, sc),
                    auc=roc_auc_score(rel, sc)))
            d = pd.DataFrame(per_fold)
            results.append(dict(model=name, query=query,
                                **{f"{c}_mean": d[c].mean() for c in d},
                                **{f"{c}_std": d[c].std(ddof=1) for c in d}))
    out = pd.DataFrame(results)
    print(out.round(4).to_string(index=False))
    out.to_csv("ranking_metrics.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("stats", "ranking"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "stats":
        paired = ["LR", "RF", "XGB"]
        if "--paired" in sys.argv:
            paired = sys.argv[sys.argv.index("--paired") + 1].split(",")
        run_stats(sys.argv[2], paired)
    else:
        run_ranking(sys.argv[2])
