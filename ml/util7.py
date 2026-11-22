"""−5点の評価に合わせた効用(3000超え1点+万枚7点)を直接学習するモデルを追加で試す"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from model4 import prep, feat_cols, P
from model3 import combine
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
F = prep(pd.read_pickle("ml/features4.pkl")); ALL, _ = feat_cols(F)
F["util7"] = F["y3"] + 7 * F["y10"]
cache = pd.read_pickle("ml/compare4_cache.pkl"); out = []; m = None; last = None
for t, te, s in cache:
    if last is None or (t - last).days >= 14:
        tr = F[F.date < t]; m = HGR(**P).fit(tr[ALL], tr["util7"]); last = t
    X = F[F.date == t].reset_index(drop=True)
    s = dict(s); s["UTIL7"] = m.predict(X[ALL]); out.append((t, te, s))
pd.to_pickle(out, "ml/compare4_cache.pkl")
