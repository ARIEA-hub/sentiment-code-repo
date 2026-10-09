import re, time, numpy as np, pandas as pd, os, glob
os.environ["NLTK_DATA"]="nltk_data"
from nltk.corpus import movie_reviews
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

STOP={"a","an","the","and","or","but","in","on","at","to","for","of","with","by","from","is","was","are","were","be","been","have","has","had","do","does","did","will","would","could","should","may","might","this","that","these","those","it","its"}
def prep(t):
    t=re.sub(r"[^a-z\s]"," ",t.lower())
    return " ".join(w for w in t.split() if w not in STOP and len(w)>1)
texts=[];labels=[]
for lab,cat in [(0,"neg"),(1,"pos")]:
    for f in movie_reviews.fileids(cat):
        texts.append(movie_reviews.raw(f)); labels.append(lab)
texts=np.array(texts); labels=np.array(labels); clean=np.array([prep(t) for t in texts])
print(len(texts), labels.sum())

def tf(): return TfidfVectorizer(max_features=20000,ngram_range=(1,2),sublinear_tf=True,min_df=2)
MODELS={
 "LR":lambda:LogisticRegression(class_weight="balanced",max_iter=1000,C=1.0),
 "SVM":lambda:LinearSVC(C=1.0,class_weight="balanced"),
 "NB":lambda:MultinomialNB(alpha=1.0),
}
def score(clf,X):
    return clf.predict_proba(X)[:,1] if hasattr(clf,"predict_proba") else clf.decision_function(X)

rows=[]; S={k:[] for k in list(MODELS)+["LR_raw"]}; Y=[]
LC=[]  # learning curve rows
SIZES=[100,200,400,800,1200,1600]
for r in range(5):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=r*1000+42)
    for k,(tr,te) in enumerate(skf.split(clean,labels)):
        y_te=labels[te]; Y.append(y_te)
        v=tf(); Xtr=v.fit_transform(clean[tr]); Xte=v.transform(clean[te])
        for name,mk in MODELS.items():
            clf=mk(); t0=time.time(); clf.fit(Xtr,labels[tr]); tt=time.time()-t0
            s=score(clf,Xte); pred=(s>(0.5 if name!="SVM" else 0)).astype(int)
            S[name].append(s)
            rows.append(dict(rep=r,fold=k,model=name,acc=accuracy_score(y_te,pred),f1=f1_score(y_te,pred),time=tt))
        # LR on raw text (same pipeline the transformers see)
        v2=tf(); Xr=v2.fit_transform(texts[tr]); Xrt=v2.transform(texts[te])
        clf=LogisticRegression(class_weight="balanced",max_iter=1000).fit(Xr,labels[tr]); s=clf.predict_proba(Xrt)[:,1]
        S["LR_raw"].append(s); rows.append(dict(rep=r,fold=k,model="LR_raw",acc=accuracy_score(y_te,(s>.5).astype(int)),f1=f1_score(y_te,(s>.5).astype(int)),time=0))
        # learning curve (stratified subsample of the training fold, same test fold)
        rng=np.random.RandomState(r*100+k)
        for n in SIZES:
            if n<len(tr):
                sub=np.concatenate([rng.choice(tr[labels[tr]==c],n//2,replace=False) for c in (0,1)])
            else: sub=tr
            v3=tf(); Xs=v3.fit_transform(clean[sub]); Xst=v3.transform(clean[te])
            for name in ("LR","SVM"):
                clf=MODELS[name]().fit(Xs,labels[sub]); s=score(clf,Xst)
                pred=(s>(0.5 if name!="SVM" else 0)).astype(int)
                LC.append(dict(rep=r,fold=k,model=name,n=n,acc=accuracy_score(y_te,pred)))
df=pd.DataFrame(rows); df.to_csv("cpu_baselines_folds.csv",index=False)
pd.DataFrame(LC).to_csv("learning_curve_classical.csv",index=False)
print((df.groupby("model")[["acc","f1"]].agg(["mean","std"])*100).round(2))
print(pd.DataFrame(LC).groupby(["model","n"]).acc.mean().unstack().round(4)*100)
os.makedirs("new_scores",exist_ok=True)
for m,v in S.items(): np.savez(f"new_scores/{m}.npz",y=np.stack(Y),score=np.stack(v))
# check against paper's LR fold-0 accuracy 0.8775
print("LR fold0 acc:",df[(df.model=='LR')].acc.iloc[0])
