# Anonymous artifact: supervision-matched comparison of DistilBERT and classical sentiment models

Per-fold results, per-review scores and scripts for the submitted paper
(Polarity v2.0 / RT-2k, 5x5 stratified CV, n=25 test folds of 400 reviews).

## Quick check (CPU, seconds)
    pip install -r requirements.txt
    python3 reproduce_key_numbers.py
Recomputes from `data/` the headline numbers: Table 1 means, paired Nadeau-Bengio
tests with Holm adjustment (11 of 21 pairs significant), the exact-truncation
tests and the learning-curve differences. (One rounding note: the n=1600
learning-curve difference is 0.825 pp; the paper prints +0.83, this script +0.82.)

## Map from paper to files
| Paper item | Files |
|---|---|
| Table 1 (accuracy, F1, AP, AUC) | `data/folds_accuracy_7.csv`, `data/ranking_metrics.csv`, `scores/*.npz` |
| Table 2 (21 pairwise tests) | `data/pairwise_7models.csv` |
| DB-HT (head+tail) results | `data/dbht_25folds_acc.csv`, `data/dbft_headtail_*.csv`, `scores/DBHT_25folds.npz` |
| Table 3 / truncation control | `data/svm_exact_truncation_folds.csv`, `data/mech_tests_exact.csv` |
| Learning curve (Table 4, Fig. 1b) | `data/learning_curve_dbft.csv`, `data/learning_curve_classical.csv`, `data/lc_stats.csv` |
| Tuned baselines / NBSVM reproduction | `data/tuned_baselines_folds.csv`, `data/wm_repro_folds.csv` |

## Scripts (`scripts/`)
CPU scripts (numpy, pandas, scipy, scikit-learn, xgboost, nltk) produce the classical
models, statistics and ranking metrics. The `colab_*` scripts fine-tune DistilBERT and
need a GPU (we used a Colab T4; each run is resumable per fold). Scripts keep the
path layout of the original working directory, so adjust the data path at the top.

Data: Polarity v2.0 (Pang & Lee), obtainable from the Cornell movie-review data page; not redistributed here.
