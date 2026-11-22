"""過去日の AI Recommend を作る。
各日 t について、t より前のデータだけで学習したモデル(predict.py と同じ構成)で上位5台を出し、
ml/pred_history.json に保存する。既に計算済みの日はスキップする。
使い方: python3 ml/history.py [開始日 YYYY-MM-DD]
"""
import sys, json, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from features import load, build
import backtest as bt

START = sys.argv[1] if len(sys.argv) > 1 else "2026-10-10"
PATH = "ml/pred_history.json"
hist = json.load(open(PATH)) if os.path.exists(PATH) else {}
F = build(load())
days = [d for d in sorted(F.date.unique()) if d >= pd.Timestamp(START)]
for t in days:
    key = pd.Timestamp(t).strftime("%Y-%m-%d")
    if key in hist: continue
    tr, te = F[F.date < t], F[F.date == t].reset_index(drop=True)
    lr = bt.models()["LR"](); lr.fit(tr[bt.LR_FEATS], tr.y)
    gb = bt.models()["GBM"](); gb.fit(tr[bt.ALL], tr.y)
    p_lr = lr.predict_proba(te[bt.LR_FEATS])[:, 1]; p_gb = gb.predict_proba(te[bt.ALL])[:, 1]
    score = pd.Series(p_lr).rank() + pd.Series(p_gb).rank() + (-te.games_s7.fillna(1e9)).rank()
    top = te.assign(score=score.values, p=(p_lr + p_gb) / 2).sort_values("score", ascending=False).head(5)
    hist[key] = [[int(r.machine), round(float(r.p), 3), int(r.y_diff)] for r in top.itertuples()]
    print(key, [h[0] for h in hist[key]], "hits", sum(h[2] > 5000 for h in hist[key]), flush=True)
    json.dump(hist, open(PATH, "w"))
h = sum(sum(x[2] > 5000 for x in v) for v in hist.values()); n = len(hist)
print(f"days {n}  +5000 in top5: {h/n:.2f}/day")
