import os, re, numpy as np, pandas as pd, time
os.environ["NLTK_DATA"]="nltk_data"
from nltk.corpus import movie_reviews as m
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from scipy import sparse
STOP={"a","an","the","and","or","but","in","on","at","to","for","of","with","by","from","is","was","are","were","be","been","have","has","had","do","does","did","will","would","could","should","may","might","this","that","these","those","it","its"}
def prep(t):
    t=re.sub(r"[^a-z\s]"," ",t.lower()); return " ".join(w for w in t.split() if w not in STOP and len(w)>1)
T=[];y=[]
for lab,cat in [(0,"neg"),(1,"pos")]:
    for f in m.fileids(cat): T.append(m.raw(f)); y.append(lab)
y=np.array(y); P=np.array([prep(t) for t in T],dtype=object); N=len(y)
Cs=[0.03,0.1,0.3,1,3,10]
def run(name,fn):
    accs=[];t0=time.time()
    for r in range(5):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r*1000+42).split(np.zeros(N),y):
            accs.append(fn(P[tr],y[tr],P[te],y[te]))
    a=np.array(accs); print(f"{name}: {a.mean()*100:.2f} ± {a.std(ddof=1)*100:.2f}  ({time.time()-t0:.0f}s)",flush=True); return a
def svm_tuned(ng):
    def fn(Xt,yt,Xe,ye):
        v=TfidfVectorizer(max_features=20000 if ng<3 else 40000,ngram_range=(1,ng),sublinear_tf=True,min_df=2)
        A=v.fit_transform(Xt); B=v.transform(Xe)
        g=GridSearchCV(LinearSVC(class_weight="balanced"),{"C":Cs},cv=3,scoring="accuracy").fit(A,yt)
        return (g.predict(B)==ye).mean()
    return fn
def nbsvm(Xt,yt,Xe,ye):
    v=CountVectorizer(ngram_range=(1,2),binary=True,min_df=2); A=v.fit_transform(Xt); B=v.transform(Xe)
    p=np.asarray(A[yt==1].sum(0)).ravel()+1; q=np.asarray(A[yt==0].sum(0)).ravel()+1
    rr=np.log((p/p.sum())/(q/q.sum()))
    A2=A.multiply(rr).tocsr(); B2=B.multiply(rr).tocsr()
    g=GridSearchCV(LogisticRegression(max_iter=2000,class_weight="balanced"),{"C":[0.1,0.3,1,3,10,30]},cv=3).fit(A2,yt)
    return (g.predict(B2)==ye).mean()
res={}
res["SVM_tuned_12"]=run("SVM tuned C, 1-2gram",svm_tuned(2))
res["SVM_tuned_13"]=run("SVM tuned C, 1-3gram",svm_tuned(3))
res["NBSVM"]=run("NB-LR (NBSVM-style) 1-2gram binary, C tuned",nbsvm)
pd.DataFrame(res).to_csv("tuned_baselines_folds.csv",index=False)
