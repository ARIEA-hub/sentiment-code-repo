import os, numpy as np, pandas as pd
os.environ["NLTK_DATA"]="nltk_data"
from nltk.corpus import movie_reviews as m
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
texts=[];labels=[]
for lab,cat in [(0,"neg"),(1,"pos")]:
    for f in m.fileids(cat): texts.append(m.raw(f)); labels.append(lab)
labels=np.array(labels); words=np.array([len(t.split()) for t in texts]); N=len(labels)
h=np.load("extraD/dbft_headtail_scores.npz"); H=h["score"]; HY=h["y"]; HT=h["te_idx"]
te=[]
for r in range(2):
    for tr,t in StratifiedKFold(5,shuffle=True,random_state=r*1000+42).split(np.zeros(N),labels): te.append(t)
assert all((a==b).all() for a,b in zip(te,HT)) and all((labels[t]==y).all() for t,y in zip(te,HY)), "fold mismatch"
print("folds identical to baseline folds: OK")
hacc=((H>.5)==HY).mean(1); print("HT acc per fold",np.round(hacc*100,2)); print("HT mean %.2f sd %.2f"%(hacc.mean()*100,hacc.std(ddof=1)*100))
csv=pd.read_csv("extraD/dbft_headtail_folds.csv"); print("csv-vs-npz acc max diff",np.abs(csv.acc.values-hacc).max())
F=pd.read_csv("folds_accuracy_7.csv")
n=10
def nb(d):
    t=d.mean()/np.sqrt((1/n+1/4)*d.var(ddof=1)); p=2*stats.t.sf(abs(t),n-1)
    se=np.sqrt((1/n+1/4)*d.var(ddof=1)); c=stats.t.ppf(.975,n-1)
    return d.mean()*100,(d.mean()-c*se)*100,(d.mean()+c*se)*100,p
res=[]
for b in ["DBFT","SVM","LR","DBZS"]:
    base=F[b].values[:10]; d=hacc-base
    mu,lo,hi,p=nb(d); res.append((b,base.mean()*100,mu,lo,hi,p,(d>0).sum()))
ps=np.array([r[5] for r in res[:3]]); order=np.argsort(ps); holm=np.empty(3)
run=0
for i,idx in enumerate(order): run=max(run,min(1,ps[idx]*(3-i))); holm[idx]=run
for i,r in enumerate(res):
    print("HT-%s: base %.2f  Δ=%+.2f CI[%+.2f,%+.2f] p=%.4f %s folds won %d/10"%(r[0],*r[1:6],("Holm3=%.4f"%holm[i]) if i<3 else "",r[6]))
# ranking metrics from scores (per-fold AUC/AP), same 10 folds
def metr(S,Y):
    return np.array([roc_auc_score(y,s) for s,y in zip(S,Y)]),np.array([average_precision_score(y,s) for s,y in zip(S,Y)])
hA,hP=metr(H,HY); print("HT AUC %.3f AP %.3f"%(hA.mean(),hP.mean()))
for f in ["DBFT","SVM","LR"]:
    z=np.load(f"rk/{f}.npz"); A,P=metr(z["score"][:10],z["y"][:10]); d=hA-A
    mu,lo,hi,p=nb(d); print(f"  {f} AUC %.3f AP %.3f | HT-{f} AUC Δ=%+.3f CI[%+.3f,%+.3f] p=%.3f"%(A.mean(),P.mean(),mu/100,lo/100,hi/100,p))
# bootstrap over reviews + length
def corr_mat(S,Y,thr):
    c=np.full((2,N),np.nan)
    for i,(t,s,y) in enumerate(zip(te,S,Y)): c[i//5,t]=((s>thr).astype(int)==y)
    return c
ch=corr_mat(H,HY,.5); cs={f:corr_mat(np.load(f"rk/{f}.npz")["score"][:10],np.load(f"rk/{f}.npz")["y"][:10],.5 if f!="SVM" else 0) for f in ["DBFT","SVM","LR","DBZS"]}
rng=np.random.RandomState(0)
def boot(a,b,mask,B=10000):
    d=(a-b).mean(0)[mask]; idx=rng.randint(0,len(d),(B,len(d))); bs=d[idx].mean(1)*100
    return d.mean()*100,*np.percentile(bs,[2.5,97.5])
print("Bootstrap over reviews (2 reps avg):")
for f in ["DBFT","SVM","LR"]:
    for lab,mask in [("all",np.ones(N,bool)),("<=512",words<=512),(">512",words>512),(">800",words>800)]:
        print("  HT-%s %-6s n=%d  d=%+.2f CI[%+.2f,%+.2f]"%((f,lab,mask.sum())+boot(ch,cs[f],mask)))
print("Accuracy by length bucket (2-rep avg):")
for lab,mask in [("<=512",words<=512),("513-800",(words>512)&(words<=800)),(">800",words>800),(">512",words>512)]:
    print("  %-8s n=%d  HT %.1f  DBFT %.1f  SVM %.1f  LR %.1f  ZS %.1f"%(lab,mask.sum(),ch.mean(0)[mask].mean()*100,*[cs[f].mean(0)[mask].mean()*100 for f in ["DBFT","SVM","LR","DBZS"]]))
# time
print("train time per fold mean %.1f s"%csv.time.mean())
