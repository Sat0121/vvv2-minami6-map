"""万枚も狙うモデル(v4)
評価(1日あたり): スコア = (+3000超えの台数) + 2×(+万枚の台数) − (対象台にいた+万枚台のうち、推奨5台から漏れた台数)
→ +万枚台は拾えば+3点(3000超えの1点を含む)、逃せば−1点
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from model3 import prep as prep3, feat_cols as feat_cols3, combine, EXCLUDED, PREV_MAX, LAYOUT_START

BIG = 10000

def prep(F):
    F = prep3(F)
    F["y10"] = (F["y_diff"] > BIG).astype(int)
    F["util"] = F["y3"] + 2 * F["y10"] + F["y10"]  # 拾ったときの価値(3000超え1点+万枚2点+逃さなかった1点)
    return F

def feat_cols(F):
    ALL, LR = feat_cols3(F)
    ALL = [c for c in ALL if c not in ("y10", "util")]
    return ALL, LR

P = dict(max_iter=150, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=0)
def fit_parts(tr, ALL):
    m = {}
    for name, y in (("GBM3", "y3"), ("GBM5", "y5"), ("GBM10", "y10")):
        c = HGC(**P).fit(tr[ALL], tr[y]); m[name] = (lambda X, c=c: c.predict_proba(X[ALL])[:, 1])
    q = HGR(loss="quantile", quantile=0.9, **P).fit(tr[ALL], tr["y_diff"]); m["Q90"] = lambda X, q=q: q.predict(X[ALL])
    u = HGR(**P).fit(tr[ALL], tr["util"]); m["UTIL"] = lambda X, u=u: u.predict(X[ALL])
    return m

def part_scores(models, te):
    s = {k: f(te) for k, f in models.items()}
    s["G7"] = -te["games_s7"].fillna(1e9).values
    s["SM"] = te["streak_minus"].fillna(0).values
    return s

def day_score(te, top):
    el = te["eligible"].values; y = te["y_diff"].values
    h3 = (y[top] > 3000).sum(); h10 = (y[top] > BIG).sum(); tot10 = ((y > BIG) & el).sum()
    return h3, h10, tot10, h3 + 2 * h10 - (tot10 - h10)
