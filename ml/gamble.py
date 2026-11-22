"""ギャン中 Recommend: 万枚を最大化するモデル
評価(1日あたり、推奨5台):
  - その日に候補台の万枚(>+10000)があれば: 推奨5台に入った万枚台1台につき1点
  - 万枚がない弱い日は: その日の差枚1位・2位(候補台の中)が推奨5台に入っていれば、それぞれ0.9点
候補台: 1311〜1316・2218(見世物小屋)を除く全台。前日の差枚では除外しない(据え置き狙いも許す)
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from model4 import prep as prep4, feat_cols as feat_cols4, P
from model3 import EXCLUDED, LAYOUT_START

BIG, WEAK_W = 10000, 0.9

def prep(F):
    F = prep4(F)
    ex = F["machine"].isin(EXCLUDED) & (F["date"] >= LAYOUT_START)
    F["elig_g"] = ~ex
    # 候補台の中でのその日の差枚順位
    F["drank"] = F["y_diff"].where(F["elig_g"]).groupby(F["date"]).rank(ascending=False, method="first")
    big_day = F[F["elig_g"]].groupby("date")["y10"].max()
    F["big_day"] = F["date"].map(big_day).fillna(0).astype(int)
    # 正解ラベル: 万枚、または万枚のない日の1位・2位
    F["yg"] = ((F["y10"] == 1) | ((F["big_day"] == 0) & (F["drank"] <= 2))).astype(int)
    F["ytop2"] = (F["drank"] <= 2).astype(int)
    F["gval"] = np.where(F["y10"] == 1, 1.0, np.where((F["big_day"] == 0) & (F["drank"] <= 2), WEAK_W, 0.0))
    return F

def feat_cols(F):
    ALL, LR = feat_cols4(F)
    return [c for c in ALL if c not in ("elig_g", "drank", "big_day", "yg", "ytop2", "gval")], LR

def fit_parts(tr, ALL):
    m = {}
    for name, y in (("GG", "yg"), ("G10", "y10"), ("TOP2", "ytop2")):
        c = HGC(**P).fit(tr[ALL], tr[y]); m[name] = (lambda X, c=c: c.predict_proba(X[ALL])[:, 1])
    for name, q in (("Q90", 0.9), ("Q95", 0.95)):
        r = HGR(loss="quantile", quantile=q, **P).fit(tr[ALL], tr["y_diff"]); m[name] = (lambda X, r=r: r.predict(X[ALL]))
    v = HGR(**P).fit(tr[ALL], tr["gval"]); m["GVAL"] = lambda X, v=v: v.predict(X[ALL])
    return m

def part_scores(models, te):
    s = {k: f(te) for k, f in models.items()}
    s["G7"] = -te["games_s7"].fillna(1e9).values
    s["SM"] = te["streak_minus"].fillna(0).values
    s["GAMES30"] = te["games_m30"].fillna(0).values  # よく回される台
    return s

def combine_g(te, s, w):
    el = te["elig_g"].values; tot = np.zeros(len(te))
    for k, wk in w.items():
        if wk: tot += wk * pd.Series(np.where(el, s[k], np.nan)).rank(pct=True).fillna(0).values
    tot[~el] = -1e9
    return tot

def day_score(te, top):
    el = te["elig_g"].values; y = te["y_diff"].values; dr = te["drank"].values
    big = ((y > BIG) & el).sum()
    if big > 0:
        return (y[top] > BIG).sum(), big, 1
    return WEAK_W * (dr[top] <= 2).sum(), 0, 0
