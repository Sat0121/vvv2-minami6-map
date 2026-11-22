"""v4: 事前に決めた候補を、前半(1/15〜6/30)で比べて1つ選び、後半(7/1〜11/21)と現配置で確かめる"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "ml")
from model4 import prep, feat_cols, fit_parts, part_scores, day_score
from model3 import combine
F = prep(pd.read_pickle("ml/features4.pkl")); ALL, _ = feat_cols(F)
days = [d for d in sorted(F.date.unique()) if d >= pd.Timestamp("2026-01-15")]
cache = []; models = None; last = None
for t in days:
    if last is None or (t - last).days >= 14:
        models = fit_parts(F[F.date < t], ALL); last = t
    te = F[F.date == t].reset_index(drop=True)
    cache.append((t, te[["machine", "y_diff", "eligible"]].copy(), part_scores(models, te)))
pd.to_pickle(cache, "ml/compare4_cache.pkl")
C = {
    "ランダム": None,
    "現行 GBM3+GBM5+G7+SM": dict(GBM3=1, GBM5=1, G7=1, SM=1),
    "現行+GBM10": dict(GBM3=1, GBM5=1, GBM10=1, G7=1, SM=1),
    "現行+GBM10×2": dict(GBM3=1, GBM5=1, GBM10=2, G7=1, SM=1),
    "現行+Q90": dict(GBM3=1, GBM5=1, Q90=1, G7=1, SM=1),
    "現行+GBM10+Q90": dict(GBM3=1, GBM5=1, GBM10=1, Q90=1, G7=1, SM=1),
    "GBM3+GBM10+G7+SM": dict(GBM3=1, GBM10=1, G7=1, SM=1),
    "GBM10+G7+SM": dict(GBM10=1, G7=1, SM=1),
    "UTIL": dict(UTIL=1),
    "UTIL+G7+SM": dict(UTIL=1, G7=1, SM=1),
    "UTIL+GBM3+GBM5+G7+SM": dict(UTIL=1, GBM3=1, GBM5=1, G7=1, SM=1),
}
rng = np.random.default_rng(0)
rows = []
for name, w in C.items():
    rec = []
    for t, te, s in cache:
        el = np.where(te["eligible"].values)[0]
        if w is None:  # ランダムは200回平均
            r = np.array([day_score(te, rng.choice(el, 5, replace=False)) for _ in range(200)]).mean(0)
        else:
            top = np.argsort(-combine(te, s, w), kind="stable")[:5]; r = day_score(te, top)
        rec.append((t, *r))
    d = pd.DataFrame(rec, columns=["t", "h3", "h10", "tot10", "S"])
    part = lambda m: d[m]
    A = d[d.t < pd.Timestamp("2026-07-01")]; B = d[d.t >= pd.Timestamp("2026-07-01")]; Cc = d[d.t >= pd.Timestamp("2026-10-10")]
    rows.append({"モデル": name, "前半スコア": A.S.mean(), "後半スコア": B.S.mean(), "現配置スコア": Cc.S.mean(),
                 "全期間スコア": d.S.mean(), "±": d.S.std() / np.sqrt(len(d)),
                 "3000超え/日": d.h3.mean(), "万枚/日": d.h10.mean(), "万枚捕捉率": d.h10.sum() / d.tot10.sum()})
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
R.to_csv("ml/compare4_result.csv", index=False)
