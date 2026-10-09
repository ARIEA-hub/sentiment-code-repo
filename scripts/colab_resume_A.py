# ============================================================================
# RESUME block A (DB-FT learning curve) on a Colab T4 -- does NOT restart.
# 1) Runtime -> Change runtime type -> T4 GPU.  2) Paste this WHOLE file into ONE
# cell and run.  It already contains the 55 finished rows from your earlier run,
# computes ONLY the missing ones (rep 1 / fold 4: n=200,400,800,1600 + DB-ZS,
# ~10 min), and downloads learning_curve_dbft_FULL.csv.  Send me that file.
# The subsampling RNG is advanced exactly as in the original run, so the
# training subsets are identical to what an uninterrupted run would have used.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","torch","nltk","scikit-learn"],check=False)
import os, io, time, random
import numpy as np, pandas as pd, torch, nltk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, pipeline)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

CKPT = "distilbert-base-uncased-finetuned-sst-2-english"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DEV.type == "cuda", "Switch the runtime to a T4 GPU first (Runtime -> Change runtime type)."
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

DONE_CSV = """rep,fold,n,model,acc,time
0,0,100,DBFT,0.835,25.598170280456543
0,0,200,DBFT,0.8325,42.02545213699341
0,0,400,DBFT,0.8425,73.46765089035034
0,0,800,DBFT,0.8775,136.81125259399414
0,0,1600,DBFT,0.8925,263.04756259918213
0,0,0,DBZS,0.825,0.0
0,1,100,DBFT,0.865,25.684762001037598
0,1,200,DBFT,0.85,41.537123680114746
0,1,400,DBFT,0.875,72.66053080558777
0,1,800,DBFT,0.8775,135.80332922935486
0,1,1600,DBFT,0.8825,262.06862020492554
0,1,0,DBZS,0.8325,0.0
0,2,100,DBFT,0.86,25.433592557907104
0,2,200,DBFT,0.875,41.832979679107666
0,2,400,DBFT,0.865,72.758615732193
0,2,800,DBFT,0.89,135.65599489212036
0,2,1600,DBFT,0.9025,262.21891808509827
0,2,0,DBZS,0.8575,0.0
0,3,100,DBFT,0.8725,25.443750381469727
0,3,200,DBFT,0.87,41.390997648239136
0,3,400,DBFT,0.8725,73.80466914176941
0,3,800,DBFT,0.8775,136.60520815849304
0,3,1600,DBFT,0.895,261.9831032752991
0,3,0,DBZS,0.8525,0.0
0,4,100,DBFT,0.8575,25.40555429458618
0,4,200,DBFT,0.855,41.45614242553711
0,4,400,DBFT,0.8625,72.76079773902893
0,4,800,DBFT,0.865,135.63793063163757
0,4,1600,DBFT,0.8825,263.0265097618103
0,4,0,DBZS,0.82,0.0
1,0,100,DBFT,0.845,26.161036491394043
1,0,200,DBFT,0.8475,41.586225509643555
1,0,400,DBFT,0.87,72.8172562122345
1,0,800,DBFT,0.875,136.02057814598083
1,0,1600,DBFT,0.89,262.28333735466003
1,0,0,DBZS,0.835,0.0
1,1,100,DBFT,0.8525,25.538649082183838
1,1,200,DBFT,0.88,41.435397148132324
1,1,400,DBFT,0.875,72.79079389572144
1,1,800,DBFT,0.8725,136.0383496284485
1,1,1600,DBFT,0.8925,262.4410033226013
1,1,0,DBZS,0.8475,0.0
1,2,100,DBFT,0.875,25.491771936416626
1,2,200,DBFT,0.8575,41.42412185668945
1,2,400,DBFT,0.8825,72.89150834083557
1,2,800,DBFT,0.9,136.46164655685425
1,2,1600,DBFT,0.895,263.4448642730713
1,2,0,DBZS,0.85,0.0
1,3,100,DBFT,0.8475,26.404719352722168
1,3,200,DBFT,0.8425,41.6159884929657
1,3,400,DBFT,0.8725,72.81573271751404
1,3,800,DBFT,0.87,135.9558629989624
1,3,1600,DBFT,0.8975,262.1499710083008
1,3,0,DBZS,0.8125,0.0
1,4,100,DBFT,0.86,25.555046319961548"""
old = pd.read_csv(io.StringIO(DONE_CSV))
done = set(zip(old.rep, old.fold, old.n, old.model))
rows = old.to_dict("records")
print("already finished rows:", len(rows))

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
        _zs = pipeline("text-classification", model=CKPT, device=0, batch_size=64,
                       truncation=True, max_length=512, top_k=None)
    return np.array([{x["label"]: x["score"] for x in r}["POSITIVE"] for r in _zs(list(texts))])

nltk.download("movie_reviews", quiet=True)
from nltk.corpus import movie_reviews
pt, pl = [], []
for lab, cat in [(0, "neg"), (1, "pos")]:
    for f in movie_reviews.fileids(cat): pt.append(movie_reviews.raw(f)); pl.append(lab)
pt, pl = np.array(pt), np.array(pl)

OUTF = "/content/learning_curve_dbft_FULL.csv"
SIZES = [100, 200, 400, 800, 1600]
for r in range(2):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r*1000+42)
    for k, (tr, te) in enumerate(skf.split(pt, pl)):
        rng = np.random.RandomState(r*100+k)
        for n in SIZES:
            # always draw, so the RNG state matches the original run even for skipped sizes
            sub = tr if n >= len(tr) else np.concatenate(
                [rng.choice(tr[pl[tr] == c], n//2, replace=False) for c in (0, 1)])
            if (r, k, n, "DBFT") in done: continue
            set_seed(r*100+k)
            t0 = time.time(); sc = ft_scores(pt[sub], pl[sub], pt[te]); tt = time.time()-t0
            rows.append(dict(rep=r, fold=k, n=n, model="DBFT", acc=accuracy_score(pl[te], (sc > .5).astype(int)), time=tt))
            done.add((r, k, n, "DBFT"))
            print("A", r, k, n, rows[-1]["acc"], flush=True)
            pd.DataFrame(rows).to_csv(OUTF, index=False)
        if (r, k, 0, "DBZS") not in done:
            sc = zs_scores(pt[te])
            rows.append(dict(rep=r, fold=k, n=0, model="DBZS", acc=accuracy_score(pl[te], (sc > .5).astype(int)), time=0))
            done.add((r, k, 0, "DBZS")); print("ZS", r, k, rows[-1]["acc"], flush=True)
            pd.DataFrame(rows).to_csv(OUTF, index=False)

df = pd.DataFrame(rows); df.to_csv(OUTF, index=False)
print(df[df.model == "DBFT"].groupby("n").acc.agg(["mean", "count"]))
try:
    from google.colab import files; files.download(OUTF)
except Exception as e: print("download manually:", OUTF, e)
print("DONE")
