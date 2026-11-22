"""全データで学習し、最新日の翌日向けおすすめ5台を出す。
予測対象日の特徴量は、最新日までのデータに「空の翌日」を足して作る。"""
import pandas as pd,numpy as np,sys,json,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'ml')
from features import load, build
import backtest as bt
df=load(); last=df.date.max(); nxt=last+pd.Timedelta(days=1)
cur=df[df.date==last].copy(); ghost=cur.assign(date=nxt, diff=0, games=np.nan, bb=np.nan, rb=np.nan, bona=0)
F=build(pd.concat([df,ghost],ignore_index=True))
tr=F[F.date<=last]; te=F[F.date==nxt].reset_index(drop=True)
lr=bt.models()["LR"](); lr.fit(tr[bt.LR_FEATS],tr.y)
gb=bt.models()["GBM"](); gb.fit(tr[bt.ALL],tr.y)
p_lr=lr.predict_proba(te[bt.LR_FEATS])[:,1]; p_gb=gb.predict_proba(te[bt.ALL])[:,1]
g7=-te.games_s7.fillna(1e9)
score=pd.Series(p_lr).rank()+pd.Series(p_gb).rank()+g7.rank()
te["score"]=score/ (3*len(te)); te["p"]=(p_lr+p_gb)/2
te=te.sort_values("score",ascending=False)
out=[dict(no=int(r.machine), score=round(float(r.score),3), p=round(float(r.p),3), g7=None if pd.isna(r.games_s7) else int(r.games_s7/7), d9=None if pd.isna(r.diff_s9) else int(r.diff_s9), sm=int(r.streak_minus)) for r in te.itertuples()]
json.dump(dict(target=str(nxt.date()), based_on=str(last.date()), ranking=out), open('ml/prediction.json','w'), ensure_ascii=False)
for i,o in enumerate(out[:10],1): print(i,o)
print('bottom',[o['no'] for o in out[-5:]])
