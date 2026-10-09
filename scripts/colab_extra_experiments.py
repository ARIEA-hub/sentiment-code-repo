# ============================================================================
# Colab (T4 GPU) -- EXTRA ECIR EXPERIMENTS.  Self-contained: paste into ONE cell
# of a fresh Colab notebook (Runtime -> GPU), run, wait, and a zip downloads.
# Runs three blocks (set the RUN_* flags to skip any).  ~2.5-3 h total on a T4.
#   D) HEAD+TAIL DB-FT on Polarity (first 128 + last 382 word-pieces; Sun et al. 2019)
#      -- tests whether DB-FT loses on LONG reviews because of 512-token truncation. ~50 min
#   B) MATCHED IMDb: same 5,000-review stratified subsample + same folds for
#      LR/SVM/RF/XGB/DB-ZS/DB-FT (1 rep x 5 folds), with ranking scores. ~60 min
#   A) DB-FT learning curve on Polarity (train sizes 100..1600, 2 reps x 5 folds). ~90 min
#   C) Efficiency: parameters, model size, GPU/CPU inference latency + throughput. ~2 min
# Each block downloads its OWN zip when it finishes (so a Colab disconnect loses
# at most the block in progress). Run order: D, B, A, C. Set RUN_* to resume.
# Fold seeds follow the main study: StratifiedKFold(random_state=r*1000+42).
# DB-FT sees RAW text (own tokenizer); classical models see the preprocessed text.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","scikit-learn","xgboost","scipy",
                "transformers","torch","datasets","nltk"],check=False)
import os, re, time, random, shutil, json
import numpy as np, pandas as pd, torch
import nltk
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, pipeline)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

RUN_A, RUN_B, RUN_C, RUN_D = True, True, True, True
CKPT = "distilbert-base-uncased-finetuned-sst-2-english"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "/content/extra_results"; os.makedirs(OUT, exist_ok=True)
def ship(tag):
    shutil.make_archive(f"/content/extra_{tag}", "zip", OUT)
    try:
        from google.colab import files; files.download(f"/content/extra_{tag}.zip")
    except Exception as e: print("download manually:", f"/content/extra_{tag}.zip", e)
STOP = {"a","an","the","and","or","but","in","on","at","to","for","of","with","by","from","is","was",
        "are","were","be","been","have","has","had","do","does","did","will","would","could","should",
        "may","might","this","that","these","those","it","its"}
def prep(t):
    t = re.sub(r"[^a-z\s]", " ", t.lower())
    return " ".join(w for w in t.split() if w not in STOP and len(w) > 1)
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def tfidf(): return TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2)

class DS(Dataset):
    def __init__(s, texts, labels, tok):
        s.enc = tok(list(texts), truncation=True, padding=True, max_length=512, return_tensors="pt"); s.l = labels
    def __len__(s): return s.enc["input_ids"].shape[0]
    def __getitem__(s, i):
        it = {k: v[i] for k, v in s.enc.items()}
        if s.l is not None: it["labels"] = torch.tensor(s.l[i])
        return it

def ft_scores(tr_t, tr_y, te_t, lr=2e-5, epochs=3, bs=16):
    tok = AutoTokenizer.from_pretrained(CKPT)
    m = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=2, ignore_mismatched_sizes=True).to(DEV)
    trd = DataLoader(DS(tr_t, list(tr_y), tok), batch_size=bs, shuffle=True)
    ted = DataLoader(DS(te_t, None, tok), batch_size=bs)
    opt = AdamW(m.parameters(), lr=lr); tot = len(trd)*epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1*tot), tot)
    m.train()
    for _ in range(epochs):
        for b in trd:
            b = {k: v.to(DEV) for k, v in b.items()}
            m(**b).loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
    m.eval(); sc = []
    with torch.no_grad():
        for b in ted:
            b = {k: v.to(DEV) for k, v in b.items()}
            sc.extend(torch.softmax(m(**b).logits, 1)[:, 1].cpu().numpy().tolist())
    del m; torch.cuda.empty_cache(); return np.array(sc)

_zs = None
def zs_scores(texts):
    global _zs
    if _zs is None:
        _zs = pipeline("text-classification", model=CKPT, device=0 if torch.cuda.is_available() else -1,
                       batch_size=64, truncation=True, max_length=512, top_k=None)
    return np.array([{x["label"]: x["score"] for x in r}["POSITIVE"] for r in _zs(list(texts))])

def rec(**k): return k

# ----------------------------- data -----------------------------------------
nltk.download("movie_reviews", quiet=True)
from nltk.corpus import movie_reviews
pt, pl = [], []
for lab, cat in [(0, "neg"), (1, "pos")]:
    for f in movie_reviews.fileids(cat): pt.append(movie_reviews.raw(f)); pl.append(lab)
pt, pl = np.array(pt), np.array(pl); pc = np.array([prep(t) for t in pt])


# ----------------------------- D) head+tail DB-FT ---------------------------
class DSHT(Dataset):
    """first 128 + last 382 word-pieces (+CLS/SEP); shorter reviews untouched"""
    def __init__(s, texts, labels, tok, head=128, tail=382):
        ids = tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
        seqs = [[tok.cls_token_id] + (x if len(x) <= head+tail else x[:head] + x[-tail:]) + [tok.sep_token_id] for x in ids]
        L = max(len(x) for x in seqs)
        s.ids = torch.tensor([x + [tok.pad_token_id]*(L-len(x)) for x in seqs])
        s.att = torch.tensor([[1]*len(x) + [0]*(L-len(x)) for x in seqs]); s.l = labels
    def __len__(s): return s.ids.shape[0]
    def __getitem__(s, i):
        it = {"input_ids": s.ids[i], "attention_mask": s.att[i]}
        if s.l is not None: it["labels"] = torch.tensor(s.l[i])
        return it

def ft_scores_ht(tr_t, tr_y, te_t, lr=2e-5, epochs=3, bs=16):
    tok = AutoTokenizer.from_pretrained(CKPT)
    m = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=2, ignore_mismatched_sizes=True).to(DEV)
    trd = DataLoader(DSHT(tr_t, list(tr_y), tok), batch_size=bs, shuffle=True)
    ted = DataLoader(DSHT(te_t, None, tok), batch_size=bs)
    opt = AdamW(m.parameters(), lr=lr); tot = len(trd)*epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1*tot), tot)
    m.train()
    for _ in range(epochs):
        for b in trd:
            b = {k: v.to(DEV) for k, v in b.items()}
            m(**b).loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
    m.eval(); sc = []
    with torch.no_grad():
        for b in ted:
            b = {k: v.to(DEV) for k, v in b.items()}
            sc.extend(torch.softmax(m(**b).logits, 1)[:, 1].cpu().numpy().tolist())
    del m; torch.cuda.empty_cache(); return np.array(sc)

if RUN_D:
    rows = []; Y = []; Sx = []; TE = []
    for r in range(2):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r*1000+42)
        for k, (tr, te) in enumerate(skf.split(pc, pl)):
            set_seed(r*100+k)
            t0 = time.time(); sc = ft_scores_ht(pt[tr], pl[tr], pt[te]); tt = time.time()-t0
            Y.append(pl[te]); Sx.append(sc); TE.append(te)
            rows.append(rec(rep=r, fold=k, model="DBFT_headtail", acc=accuracy_score(pl[te], (sc > .5).astype(int)), time=tt))
            print("D", r, k, rows[-1]["acc"], flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/dbft_headtail_folds.csv", index=False)
            np.savez(f"{OUT}/dbft_headtail_scores.npz", y=np.stack(Y), score=np.stack(Sx), te_idx=np.stack(TE))
    ship("D")

# ----------------------------- B) matched IMDb ------------------------------
if RUN_B:
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    it = np.array([x["text"] for x in ds["train"]] + [x["text"] for x in ds["test"]])
    il = np.array([x["label"] for x in ds["train"]] + [x["label"] for x in ds["test"]])
    rng = np.random.RandomState(2026)
    idx = np.concatenate([rng.choice(np.where(il == c)[0], 2500, replace=False) for c in (0, 1)])
    it, il = it[idx], il[idx]; ic = np.array([prep(t) for t in it])
    rows = []; Y = []; S = {m: [] for m in ["LR", "SVM", "RF", "XGB", "DBZS", "DBFT"]}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(skf.split(ic, il)):
        set_seed(k); y = il[te]; Y.append(y)
        v = tfidf(); Xtr = v.fit_transform(ic[tr]); Xte = v.transform(ic[te])
        for name, clf in [("LR", LogisticRegression(class_weight="balanced", max_iter=1000)),
                          ("SVM", LinearSVC(class_weight="balanced")),
                          ("RF", RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=k)),
                          ("XGB", xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=.05, eval_metric="logloss", verbosity=0, random_state=k))]:
            clf.fit(Xtr, il[tr])
            s = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)
            thr = 0 if name == "SVM" else .5
            S[name].append(s); rows.append(rec(fold=k, model=name, acc=accuracy_score(y, (s > thr).astype(int)), f1=f1_score(y, (s > thr).astype(int))))
        s = zs_scores(it[te]); S["DBZS"].append(s)
        rows.append(rec(fold=k, model="DBZS", acc=accuracy_score(y, (s > .5).astype(int)), f1=f1_score(y, (s > .5).astype(int))))
        t0 = time.time(); s = ft_scores(it[tr], il[tr], it[te]); tt = time.time()-t0; S["DBFT"].append(s)
        rows.append(rec(fold=k, model="DBFT", acc=accuracy_score(y, (s > .5).astype(int)), f1=f1_score(y, (s > .5).astype(int)), time=tt))
        print("B fold", k, {r["model"]: round(r["acc"], 4) for r in rows if r["fold"] == k}, flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/imdb_matched_folds.csv", index=False)
    os.makedirs(f"{OUT}/imdb_scores", exist_ok=True)
    for m, v in S.items(): np.savez(f"{OUT}/imdb_scores/{m}.npz", y=np.stack(Y), score=np.stack(v))
    ship("B")

# ----------------------------- A) DB-FT learning curve ----------------------
if RUN_A:
    SIZES = [100, 200, 400, 800, 1600]; rows = []
    for r in range(2):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r*1000+42)
        for k, (tr, te) in enumerate(skf.split(pc, pl)):
            rng = np.random.RandomState(r*100+k)
            for n in SIZES:
                sub = tr if n >= len(tr) else np.concatenate(
                    [rng.choice(tr[pl[tr] == c], n//2, replace=False) for c in (0, 1)])
                set_seed(r*100+k)
                t0 = time.time(); sc = ft_scores(pt[sub], pl[sub], pt[te]); tt = time.time()-t0
                rows.append(rec(rep=r, fold=k, n=n, model="DBFT", acc=accuracy_score(pl[te], (sc > .5).astype(int)), time=tt))
                print("A", r, k, n, rows[-1]["acc"], flush=True)
                pd.DataFrame(rows).to_csv(f"{OUT}/learning_curve_dbft.csv", index=False)
            # DB-ZS needs no training, but report it once per fold for the curve's flat reference
            sc = zs_scores(pt[te])
            rows.append(rec(rep=r, fold=k, n=0, model="DBZS", acc=accuracy_score(pl[te], (sc > .5).astype(int)), time=0))
            pd.DataFrame(rows).to_csv(f"{OUT}/learning_curve_dbft.csv", index=False)
    ship("A")

# ----------------------------- C) efficiency --------------------------------
if RUN_C:
    res = {}; sample = list(pt[:256])
    tok = AutoTokenizer.from_pretrained(CKPT)
    mdl = AutoModelForSequenceClassification.from_pretrained(CKPT)
    res["params_M"] = sum(p.numel() for p in mdl.parameters())/1e6
    res["fp32_size_MB"] = res["params_M"]*4
    def bench(device, n=256, bs=32):
        m = mdl.to(device).eval(); t0 = time.time()
        with torch.no_grad():
            for i in range(0, n, bs):
                b = tok(sample[i:i+bs], truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
                m(**b)
        if device.type == "cuda": torch.cuda.synchronize()
        dt = time.time()-t0; return dict(throughput_rev_per_s=n/dt, ms_per_review=1000*dt/n)
    if torch.cuda.is_available():
        bench(torch.device("cuda"), 64); res["dbft_gpu_T4"] = bench(torch.device("cuda"))
        res["gpu_peak_mem_MB"] = torch.cuda.max_memory_allocated()/2**20
    res["dbft_cpu"] = bench(torch.device("cpu"), n=64, bs=16)
    v = tfidf(); X = v.fit_transform(pc[:1600]); lr = LogisticRegression(max_iter=1000).fit(X, pl[:1600])
    Xs = v.transform(pc[:256]); t0 = time.time()
    for _ in range(20): lr.predict(v.transform(pc[:256]))
    dt = (time.time()-t0)/20; res["lr_cpu"] = dict(throughput_rev_per_s=256/dt, ms_per_review=1000*dt/256)
    res["lr_size_MB"] = (lr.coef_.nbytes + 8*len(v.vocabulary_))/2**20
    json.dump(res, open(f"{OUT}/efficiency.json", "w"), indent=1); print(res)
    ship("C")

print('ALL DONE')
