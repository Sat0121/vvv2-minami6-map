import pandas as pd,numpy as np,sys,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'ml'); import backtest as bt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
F=pd.read_pickle('ml/features.pkl'); bt.F=F
SMALL=["games_s7_rk","games_s3_rk","bona_s9_rk","bona_s1_rk","diff_s9_rk","streak_minus","nb_max_1","kuse","games_m30_rk"]
mk_small=lambda: make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=0.05,max_iter=2000))
days=sorted(F.date.unique()); test=[d for d in days if d>=pd.Timestamp("2026-01-15")]
res=[];fit={};last=None
for t in test:
    if last is None or (t-last).days>=14:
        tr=F[F.date<t]
        for n,mk,fs in (("LR",bt.models()["LR"],bt.LR_FEATS),("GBM",bt.models()["GBM"],bt.ALL),("LR小",mk_small,SMALL)):
            m=mk(); m.fit(tr[fs],tr.y); fit[n]=(m,fs)
        last=t
    te=F[F.date==t]; avg=te.y_diff.mean(); rate=te.y.mean()
    p={n:pd.Series(m.predict_proba(te[fs])[:,1]).rank().values for n,(m,fs) in fit.items()}
    g7=pd.Series(-te.games_s7.fillna(1e9).values).rank().values
    S={"LR小":p["LR小"],"LR+GBM":p["LR"]+p["GBM"],"LR+GBM+回転7":p["LR"]+p["GBM"]+g7,"LR小+回転7":p["LR小"]+g7}
    for n,s in S.items():
        top=te.iloc[np.argsort(-s,kind="stable")[:5]]
        res.append(dict(date=t,model=n,hits=top.y.sum(),any=int(top.y.sum()>0),lift=top.y_diff.mean()-avg,rand=5*rate))
R=pd.DataFrame(res); R0=pd.read_csv('ml/backtest_days.csv',parse_dates=['date'])
R=pd.concat([R0[R0.model!="LR+GBM"],R]); R.to_csv('ml/backtest_all.csv',index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",20)
print(bt.summary(R).round(3).to_string())
