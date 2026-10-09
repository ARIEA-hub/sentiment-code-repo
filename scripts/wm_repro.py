import os, re, numpy as np, pandas as pd
os.environ["NLTK_DATA"]="nltk_data"
from nltk.corpus import movie_reviews as m
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
T=[];y=[]
for lab,cat in [(0,"neg"),(1,"pos")]:
    for f in m.fileids(cat): T.append(m.raw(f).lower()); y.append(lab)   # Wang&Manning: simple tokenization, no stopword removal
y=np.array(y);T=np.array(T,dtype=object);N=len(y)
def feats(ng,tr,te):
    v=CountVectorizer(ngram_range=(1,ng),binary=True,token_pattern=r"[A-Za-z0-9']+|[^\sA-Za-z0-9']"); A=v.fit_transform(tr); B=v.transform(te); return A,B
def mnb(ng):
    def f(tr,ytr,te):
        A,B=feats(ng,tr,te); return MultinomialNB(alpha=1).fit(A,ytr).predict(B)
    return f
def svm(ng,C=0.1):
    def f(tr,ytr,te):
        A,B=feats(ng,tr,te); return LinearSVC(C=C).fit(A,ytr).predict(B)
    return f
def nbsvm(ng,C=1.0,beta=0.25,alpha=1.0):
    def f(tr,ytr,te):
        A,B=feats(ng,tr,te)
        p=alpha+np.asarray(A[ytr==1].sum(0)).ravel(); q=alpha+np.asarray(A[ytr==0].sum(0)).ravel()
        r=np.log((p/np.abs(p).sum())/(q/np.abs(q).sum()))
        A2=A.multiply(r).tocsr(); B2=B.multiply(r).tocsr()
        clf=LinearSVC(C=C).fit(A2,ytr); w=clf.coef_.ravel(); wb=np.abs(w).mean(); w2=(1-beta)*wb+beta*w
        return ((B2@w2+clf.intercept_[0])>0).astype(int)
    return f
M={"MNB-uni":mnb(1),"MNB-bi":mnb(2),"SVM-uni":svm(1),"SVM-bi":svm(2),"NBSVM-uni":nbsvm(1),"NBSVM-bi":nbsvm(2)}
paper={"MNB-uni":83.45,"MNB-bi":85.85,"SVM-uni":86.25,"SVM-bi":87.40,"NBSVM-uni":87.80,"NBSVM-bi":89.45}
out={}
for name,fn in M.items():
    a10=[];a25=[]
    # 10-fold CV (as in the original), one shuffled run
    for tr,te in StratifiedKFold(10,shuffle=True,random_state=42).split(T,y): a10.append((fn(T[tr],y[tr],T[te])==y[te]).mean())
    for r in range(5):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r*1000+42).split(T,y): a25.append((fn(T[tr],y[tr],T[te])==y[te]).mean())
    out[name]=np.array(a25)
    print(f"{name:10s} paper {paper[name]:.2f} | 10-fold {np.mean(a10)*100:.2f} | 5x5 {np.mean(a25)*100:.2f} ± {np.std(a25,ddof=1)*100:.2f}",flush=True)
pd.DataFrame(out).to_csv("wm_repro_folds.csv",index=False)
