import os, numpy as np, pandas as pd
os.environ["NLTK_DATA"]="nltk_data"
from nltk.corpus import movie_reviews as m
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
T=[];y=[]
for lab,cat in [(0,"neg"),(1,"pos")]:
    for f in m.fileids(cat): T.append(m.raw(f)); y.append(lab)
y=np.array(y); N=len(y); words=np.array([len(t.split()) for t in T])
te_idx=[]
for r in range(5):
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=r*1000+42).split(np.zeros(N),y): te_idx.append(te)
# merge HT scores (25 folds)
h0=np.load("extraD/dbft_headtail_scores.npz"); S=[None]*25
for i in range(10): assert (h0["te_idx"][i]==te_idx[i]).all(); S[i]=h0["score"][i]
for r in range(2,5):
    for k in range(5):
        a=np.load(f"exH/ht_scores_r{r}_f{k}.npy"); i=r*5+k
        assert (a[2].astype(int)==te_idx[i]).all() and (a[0].astype(int)==y[te_idx[i]]).all(); S[i]=a[1]
S=np.array(S); Y=np.array([y[t] for t in te_idx])
np.savez("ht_scores_25folds.npz",y=Y,score=S,te_idx=np.array(te_idx))
hacc=((S>.5)==Y).mean(1); print("DB-HT 25 folds: %.2f ± %.2f"%(hacc.mean()*100,hacc.std(ddof=1)*100))
pd.DataFrame({"acc":hacc}).to_csv("dbht_25folds_acc.csv",index=False)
F=pd.read_csv("folds_accuracy_7.csv"); W=pd.read_csv("wm_repro_folds.csv"); SX=pd.read_csv("exS/svm_exact_truncation_folds.csv")
def nb(d):
    n=len(d); se=np.sqrt((1/n+.25)*d.var(ddof=1)); t=d.mean()/se; p=2*stats.t.sf(abs(t),n-1); c=stats.t.ppf(.975,n-1); return d.mean()*100,(d.mean()-c*se)*100,(d.mean()+c*se)*100,p
def holm(ps):
    ps=np.array(ps); o=np.argsort(ps); h=np.empty(len(ps)); run=0
    for i,ix in enumerate(o): run=max(run,min(1,ps[ix]*(len(ps)-i))); h[ix]=run
    return h
print("\n== DB-HT vs each model (25 folds), Holm over 7")
names=["DBFT","SVM","LR","NB","RF","XGB","DBZS"]; res=[nb(hacc-F[n].values) for n in names]; hp=holm([r[3] for r in res])
for n,r,ph in zip(names,res,hp): print("DB-HT - %-5s %+.2f [%+.2f,%+.2f] p=%.4f pHolm=%.4f  (folds won %d/25)"%(n,*r,ph,(hacc>F[n].values).sum()))
r=nb(hacc-W["NBSVM-bi"].values); print("DB-HT - NBSVM-bi %+.2f [%+.2f,%+.2f] p=%.4f"%r)
# ranking
def rk(Sc):
    A=np.array([roc_auc_score(yy,s) for s,yy in zip(Sc,Y)]); P=np.array([average_precision_score(yy,s) for s,yy in zip(Sc,Y)])
    n50=np.mean([ndcg_score([rel],[sc],k=50) for s,yy in zip(Sc,Y) for rel,sc in [(yy,s),(1-yy,-s)]]); return A,P,n50
hA,hP,hN=rk(S); print("\nDB-HT AUC %.3f AP %.3f nDCG@50 %.4f"%(hA.mean(),hP.mean(),hN))
for f in ["DBFT","SVM"]:
    z=np.load(f"rk/{f}.npz"); A,P,n=rk(z["score"]); d=hA-A; print(" vs %s: AUC %.3f; ΔAUC %+.3f [%+.3f,%+.3f] p=%.4f"%(f,A.mean(),*[v/100 if i<3 else v for i,v in enumerate(nb(d))]))
# length analysis (5-rep avg)
def corr(Sc,thr):
    c=np.zeros((5,N))
    for i,(t,s) in enumerate(zip(te_idx,Sc)): c[i//5,t]=((s>thr).astype(int)==y[t])
    return c
cH=corr(S,.5); cD=corr(np.load("rk/DBFT.npz")["score"],.5); cS=corr(np.load("rk/SVM.npz")["score"],0); cL=corr(np.load("rk/LR.npz")["score"],.5)
rng=np.random.RandomState(0)
def boot(a,b,mask,B=10000):
    d=(a-b).mean(0)[mask]; idx=rng.randint(0,len(d),(B,len(d))); bs=d[idx].mean(1)*100; return d.mean()*100,*np.percentile(bs,[2.5,97.5])
print("\nLength buckets (5-rep avg): HT / DBFT / SVM / LR")
for lab,mask in [("<=512",words<=512),("513-800",(words>512)&(words<=800)),(">800",words>800),("all",words>0)]:
    print("%-8s n=%d  HT %.1f DBFT %.1f SVM %.1f LR %.1f | HT-SVM %+.2f [%+.2f,%+.2f] | HT-DBFT %+.2f [%+.2f,%+.2f]"%(lab,mask.sum(),*[x.mean(0)[mask].mean()*100 for x in (cH,cD,cS,cL)],*boot(cH,cS,mask),*boot(cH,cD,mask)))
x=np.log2(words)
for lab,d in [("HT-SVM",(cH-cS).mean(0)),("DBFT-SVM",(cD-cS).mean(0))]:
    sl=np.polyfit(x,d,1)[0]; bs=[np.polyfit(x[i],d[i],1)[0] for i in (rng.randint(0,N,N) for _ in range(3000))]
    print("slope %s per doubling: %+.2f pp [%+.2f,%+.2f]"%(lab,sl*100,*np.percentile(bs,[2.5,97.5])*100))
# exact SVM truncation
print("\n== exact SVM truncation means:",{c:round(SX[c].mean()*100,2) for c in ["full","first510","last510","ht128_382"]})
tests={"SVM full - SVM first510":(SX.full-SX.first510).values,"SVM last510 - SVM first510":(SX.last510-SX.first510).values,
 "SVM full - SVM last510":(SX.full-SX.last510).values,"SVM full - SVM ht128_382":(SX.full-SX.ht128_382).values,
 "DB-FT - SVM first510":(F.DBFT-SX.first510).values,"DB-HT - SVM ht128_382":hacc-SX.ht128_382.values}
res=[nb(d) for d in tests.values()]; hp=holm([r[3] for r in res]); out=[]
for (k,_),r,ph in zip(tests.items(),res,hp): print("%-30s %+.2f [%+.2f,%+.2f] p=%.4f pHolm=%.4f"%(k,*r,ph)); out.append((k,*r,ph))
pd.DataFrame(out,columns=["test","d","lo","hi","p","pholm"]).to_csv("mech_tests_exact.csv",index=False)
# continuous: unchanged
