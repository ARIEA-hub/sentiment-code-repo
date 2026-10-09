import os, numpy as np, pandas as pd
os.environ["NLTK_DATA"]="nltk_data"
from nltk.corpus import movie_reviews as m
from sklearn.model_selection import StratifiedKFold
texts=[];labels=[]
for lab,cat in [(0,"neg"),(1,"pos")]:
    for f in m.fileids(cat): texts.append(m.raw(f)); labels.append(lab)
labels=np.array(labels); words=np.array([len(t.split()) for t in texts])
N=len(labels); te_idx=[]
for r in range(5):
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=r*1000+42).split(np.zeros(N),labels): te_idx.append(te)
M={"DB-FT":("DBFT",.5),"SVM":("SVM",0),"LR":("LR",.5),"DB-ZS":("DBZS",.5)}
corr={}
for name,(f,thr) in M.items():
    z=np.load(f"rk/{f}.npz"); Y,S=z["y"],z["score"]
    c=np.zeros((5,N)); 
    for i,(te,y,s) in enumerate(zip(te_idx,Y,S)):
        assert (labels[te]==y).all()
        c[i//5,te]=((s>thr).astype(int)==y)
    corr[name]=c   # reps x reviews
rng=np.random.RandomState(0)
def boot(a,b,B=10000):
    d=(corr[a]-corr[b]).mean(0)   # per-review mean over 5 reps
    idx=rng.randint(0,N,(B,N)); bs=d[idx].mean(1)*100
    return d.mean()*100,np.percentile(bs,[2.5,97.5]),(bs<=0).mean()*2
for a,b in [("DB-FT","SVM"),("DB-FT","LR"),("SVM","LR")]:
    print(a,"-",b,"Δ=%.2f CI [%.2f,%.2f] boot-p=%.3f"%((lambda r:(r[0],r[1][0],r[1][1],min(1,r[2])))(boot(a,b))))
# error overlap
a=corr["DB-FT"].mean(0)>=.5; b=corr["SVM"].mean(0)>=.5   # majority-over-reps correctness
both=(a&b).mean();onlyF=(a&~b).mean();onlyS=(~a&b).mean();neither=(~a&~b).mean()
print("majority-correct: both %.3f onlyDBFT %.3f onlySVM %.3f neither %.3f"%(both,onlyF,onlyS,neither))
# length analysis
for lab,mask in [("<=512 words",words<=512),(">512 words",words>512)]:
    print(lab,mask.sum(),{k:round(v[:,mask].mean()*100,2) for k,v in corr.items()})
for lab,mask in [("<=300",words<=300),("300-512",(words>300)&(words<=512)),("512-800",(words>512)&(words<=800)),(">800",words>800)]:
    print(lab,mask.sum(),{k:round(v[:,mask].mean()*100,2) for k,v in corr.items()})
# DB-FT-SVM diff by length bucket with bootstrap CI
for lab,mask in [("<=512",words<=512),(">512",words>512)]:
    d=(corr["DB-FT"]-corr["SVM"])[:,mask].mean(0); idx=rng.randint(0,mask.sum(),(10000,mask.sum())); bs=d[idx].mean(1)*100
    print(lab,"DBFT-SVM Δ=%.2f CI [%.2f,%.2f]"%(d.mean()*100,*np.percentile(bs,[2.5,97.5])))
