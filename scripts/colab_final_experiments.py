# ============================================================================
# Colab (T4 GPU) -- FINAL EXPERIMENTS for the ECIR paper.  Paste into ONE cell, run.
# Runtime -> Change runtime type -> T4 GPU.  Keep the tab open.  ~2 h total.
#  S) exact tokenizer-based truncation control for the SVM (CPU part, ~5 min)
#  H) head+tail DB-HT on Polarity, repetitions 3-5 (15 more folds -> 25 total), ~65 min
#  I) matched IMDb: 4,000-review stratified subsample, 5 folds, LR/SVM/NBSVM-bi/DB-ZS/DB-FT/DB-HT, ~50 min
# Each block downloads its OWN zip when it finishes: extra_S.zip, extra_H.zip, extra_I.zip.
# Results are saved after EVERY fold. If Colab disconnects: reconnect, run this same
# cell again; blocks skip folds already present in /content/extra_results (if the VM
# was recycled, upload your partial CSVs into /content/extra_results/ first).
# A crash in one block is printed and the next block still runs.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","scikit-learn","scipy","transformers","torch","datasets","nltk"],check=False)
import os, re, time, random, shutil, traceback
import numpy as np, pandas as pd, torch, nltk
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, pipeline)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

RUN_S, RUN_H, RUN_I = True, True, True
CKPT = "distilbert-base-uncased-finetuned-sst-2-english"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DEV.type == "cuda", "Switch the runtime to a T4 GPU first."
OUT = "/content/extra_results"; os.makedirs(OUT, exist_ok=True)
TOK = AutoTokenizer.from_pretrained(CKPT)
def ship(tag):
    shutil.make_archive(f"/content/extra_{tag}", "zip", OUT)
    try:
        from google.colab import files; files.download(f"/content/extra_{tag}.zip")
    except Exception as e: print("download manually:", f"/content/extra_{tag}.zip", e)
STOP = {"a","an","the","and","or","but","in","on","at","to","for","of","with","by","from","is","was","are","were","be","been","have","has","had","do","does","did","will","would","could","should","may","might","this","that","these","those","it","its"}
def prep(t):
    t = re.sub(r"[^a-z\s]", " ", t.lower()); return " ".join(w for w in t.split() if w not in STOP and len(w) > 1)
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def tfidf(): return TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2)

# ---- encoders (word-piece id lists, with [CLS]/[SEP]) ----
def ids_of(texts): return TOK(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
def enc_first(texts, n=510): return [[TOK.cls_token_id] + x[:n] + [TOK.sep_token_id] for x in ids_of(texts)]
def enc_ht(texts, head=128, tail=382):
    return [[TOK.cls_token_id] + (x if len(x) <= head+tail else x[:head] + x[-tail:]) + [TOK.sep_token_id] for x in ids_of(texts)]
class DSV(Dataset):
    def __init__(s, seqs, labels=None): s.q = seqs; s.l = labels
    def __len__(s): return len(s.q)
    def __getitem__(s, i): return (s.q[i], None if s.l is None else int(s.l[i]))
def collate(b):
    L = max(len(x[0]) for x in b)
    ids = torch.tensor([x[0] + [TOK.pad_token_id]*(L-len(x[0])) for x in b])
    att = torch.tensor([[1]*len(x[0]) + [0]*(L-len(x[0])) for x in b]); out = {"input_ids": ids, "attention_mask": att}
    if b[0][1] is not None: out["labels"] = torch.tensor([x[1] for x in b])
    return out
def fit_scores(enc_tr, ytr, enc_te, lr=2e-5, epochs=3, bs=16):
    m = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=2, ignore_mismatched_sizes=True).to(DEV)
    trd = DataLoader(DSV(enc_tr, ytr), batch_size=bs, shuffle=True, collate_fn=collate)
    ted = DataLoader(DSV(enc_te), batch_size=32, collate_fn=collate)
    opt = AdamW(m.parameters(), lr=lr); tot = len(trd)*epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1*tot), tot); m.train()
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
    if _zs is None: _zs = pipeline("text-classification", model=CKPT, device=0, batch_size=64, truncation=True, max_length=512, top_k=None)
    return np.array([{x["label"]: x["score"] for x in r}["POSITIVE"] for r in _zs(list(texts))])
def nbsvm_bi(tr, ytr, te, C=1.0, beta=0.25, alpha=1.0):
    v = CountVectorizer(ngram_range=(1,2), binary=True, token_pattern=r"[A-Za-z0-9']+|[^\sA-Za-z0-9']"); A = v.fit_transform(tr); B = v.transform(te)
    p = alpha + np.asarray(A[ytr==1].sum(0)).ravel(); q = alpha + np.asarray(A[ytr==0].sum(0)).ravel()
    r = np.log((p/np.abs(p).sum())/(q/np.abs(q).sum())); A2 = A.multiply(r).tocsr(); B2 = B.multiply(r).tocsr()
    clf = LinearSVC(C=C).fit(A2, ytr); w = clf.coef_.ravel(); w2 = (1-beta)*np.abs(w).mean() + beta*w
    return (B2 @ w2 + clf.intercept_[0])
def load_rows(name):
    f = f"{OUT}/{name}"; return pd.read_csv(f).to_dict("records") if os.path.exists(f) else []

nltk.download("movie_reviews", quiet=True)
from nltk.corpus import movie_reviews
pt, pl = [], []
for lab, cat in [(0, "neg"), (1, "pos")]:
    for f in movie_reviews.fileids(cat): pt.append(movie_reviews.raw(f)); pl.append(lab)
pt, pl = np.array(pt), np.array(pl)

# ----------------------------- S) exact truncation control for the SVM ------
def block_S():
    rows = load_rows("svm_exact_truncation_folds.csv"); done = {(r["rep"], r["fold"]) for r in rows}
    ids = ids_of(pt)
    def dec(sel): return np.array([prep(TOK.decode(sel(x))) for x in ids], dtype=object)
    views = {"full": np.array([prep(t) for t in pt], dtype=object),
             "first510": dec(lambda x: x[:510]), "last510": dec(lambda x: x[-510:]),
             "ht128_382": dec(lambda x: x if len(x) <= 510 else x[:128] + x[-382:])}
    for r in range(5):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r*1000+42).split(pt, pl)):
            if (r, k) in done: continue
            row = dict(rep=r, fold=k)
            for name, V in views.items():
                v = tfidf(); A = v.fit_transform(V[tr]); B = v.transform(V[te])
                row[name] = accuracy_score(pl[te], LinearSVC(C=1, class_weight="balanced").fit(A, pl[tr]).predict(B))
            rows.append(row); pd.DataFrame(rows).to_csv(f"{OUT}/svm_exact_truncation_folds.csv", index=False)
    d = pd.DataFrame(rows); print("S done:", d.drop(columns=["rep","fold"]).mean().round(4).to_dict()); ship("S")

# ----------------------------- H) head+tail, repetitions 3-5 ----------------
def block_H():
    rows = load_rows("dbft_headtail_reps3to5.csv"); done = {(r["rep"], r["fold"]) for r in rows}
    sc_store = {}
    for r in range(2, 5):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r*1000+42).split(pt, pl)):
            if (r, k) in done: continue
            set_seed(r*100+k); t0 = time.time()
            sc = fit_scores(enc_ht(pt[tr]), pl[tr], enc_ht(pt[te])); tt = time.time()-t0
            rows.append(dict(rep=r, fold=k, model="DBFT_headtail", acc=accuracy_score(pl[te], (sc > .5).astype(int)), time=tt))
            np.save(f"{OUT}/ht_scores_r{r}_f{k}.npy", np.stack([pl[te], sc, te]))
            print("H", r, k, rows[-1]["acc"], flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/dbft_headtail_reps3to5.csv", index=False)
    print("H done"); ship("H")

# ----------------------------- I) matched IMDb ------------------------------
def block_I():
    from datasets import load_dataset
    try: ds = load_dataset("stanfordnlp/imdb")
    except Exception: ds = load_dataset("imdb")
    it = np.array([x["text"] for x in ds["train"]] + [x["text"] for x in ds["test"]])
    il = np.array([x["label"] for x in ds["train"]] + [x["label"] for x in ds["test"]])
    rng = np.random.RandomState(2026)
    idx = np.concatenate([rng.choice(np.where(il == c)[0], 2000, replace=False) for c in (0, 1)])
    it, il = it[idx], il[idx]; ic = np.array([prep(re.sub(r"<br\s*/?>", " ", t)) for t in it], dtype=object)
    it = np.array([re.sub(r"<br\s*/?>", " ", t) for t in it], dtype=object)
    L = np.array([len(x) for x in ids_of(it)]); print("IMDb subsample: median word-pieces %d, share >510: %.3f" % (np.median(L), (L > 510).mean()))
    rows = load_rows("imdb_matched_folds.csv"); done = {r["fold"] for r in rows}
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(ic, il)):
        if k in done: continue
        y = il[te]; row = dict(fold=k); S = {}
        v = tfidf(); A = v.fit_transform(ic[tr]); B = v.transform(ic[te])
        S["LR"] = LogisticRegression(class_weight="balanced", max_iter=1000).fit(A, il[tr]).predict_proba(B)[:, 1]
        S["SVM"] = LinearSVC(C=1, class_weight="balanced").fit(A, il[tr]).decision_function(B)
        S["NBSVM_bi"] = nbsvm_bi(it[tr].astype(str).tolist(), il[tr], it[te].astype(str).tolist())
        S["DBZS"] = zs_scores(it[te])
        set_seed(k); t0 = time.time(); S["DBFT"] = fit_scores(enc_first(it[tr]), il[tr], enc_first(it[te])); row["t_dbft"] = time.time()-t0
        set_seed(k); t0 = time.time(); S["DBHT"] = fit_scores(enc_ht(it[tr]), il[tr], enc_ht(it[te])); row["t_dbht"] = time.time()-t0
        for n, s in S.items(): row[n] = accuracy_score(y, (s > (0 if n in ("SVM", "NBSVM_bi") else .5)).astype(int))
        np.savez(f"{OUT}/imdb_scores_f{k}.npz", y=y, te=te, **S)
        rows.append(row); print("I fold", k, {n: round(row[n], 4) for n in S}, flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/imdb_matched_folds.csv", index=False)
    print("I done"); ship("I")

for flag, fn, nm in [(RUN_S, block_S, "S"), (RUN_H, block_H, "H"), (RUN_I, block_I, "I")]:
    if flag:
        try: fn()
        except Exception: print("BLOCK", nm, "FAILED:"); traceback.print_exc()
print("ALL DONE")
