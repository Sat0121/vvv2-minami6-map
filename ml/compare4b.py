"""万枚を逃したら−5点の評価で、v4の候補を比べ直す(部品の予測は compare4 のキャッシュを使う)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "ml")
from model3 import combine
PEN = float(sys.argv[1]) if len(sys.argv) > 1 else 5
cache = pd.read_pickle("ml/compare4_cache.pkl")
def score(te, top):
    y = te["y_diff"].values; el = te["eligible"].values
    h3 = (y[top] > 3000).sum(); h10 = (y[top] > 10000).sum(); tot = ((y > 10000) & el).sum()
    return h3, h10, tot, h3 + 2 * h10 - PEN * (tot - h10)
C = {
    "ランダム": None,
    "v3 GBM3+GBM5+G7+SM": dict(GBM3=1, GBM5=1, G7=1, SM=1),
    "v4 +GBM10+Q90(現行)": dict(GBM3=1, GBM5=1, GBM10=1, Q90=1, G7=1, SM=1),
    "v4 GBM10×2": dict(GBM3=1, GBM5=1, GBM10=2, Q90=1, G7=1, SM=1),
    "v4 GBM10×2+Q90×2": dict(GBM3=1, GBM5=1, GBM10=2, Q90=2, G7=1, SM=1),
    "GBM10+Q90+G7+SM": dict(GBM10=1, Q90=1, G7=1, SM=1),
    "GBM10+Q90": dict(GBM10=1, Q90=1),
    "GBM10+G7+SM": dict(GBM10=1, G7=1, SM=1),
    "GBM5+GBM10+Q90+G7+SM": dict(GBM5=1, GBM10=1, Q90=1, G7=1, SM=1),
    "現行+GBM10+Q90": dict(GBM3=1, GBM5=1, GBM10=1, Q90=1, G7=1, SM=1),
    "UTIL+G7+SM": dict(UTIL=1, G7=1, SM=1),
    "Q90+G7+SM": dict(Q90=1, G7=1, SM=1),
    "UTIL7(−5点用の効用)": dict(UTIL7=1),
    "UTIL7+G7+SM": dict(UTIL7=1, G7=1, SM=1),
    "現行+UTIL7": dict(GBM3=1, GBM5=1, GBM10=1, Q90=1, G7=1, SM=1, UTIL7=1),
}
C.pop("現行+GBM10+Q90")
rng = np.random.default_rng(0); rows = []
for name, w in C.items():
    rec = []
    for t, te, s in cache:
        el = np.where(te["eligible"].values)[0]
        r = np.array([score(te, rng.choice(el, 5, replace=False)) for _ in range(200)]).mean(0) if w is None else score(te, np.argsort(-combine(te, s, w), kind="stable")[:5])
        rec.append((t, *r))
    d = pd.DataFrame(rec, columns=["t", "h3", "h10", "tot", "S"])
    A = d[d.t < pd.Timestamp("2026-07-01")]; B = d[d.t >= pd.Timestamp("2026-07-01")]; Cc = d[d.t >= pd.Timestamp("2026-10-10")]
    rows.append({"モデル": name, "前半": A.S.mean(), "後半": B.S.mean(), "現配置": Cc.S.mean(), "全期間": d.S.mean(), "±": d.S.std() / np.sqrt(len(d)),
                 "+3000/日": d.h3.mean(), "万枚/日": d.h10.mean(), "捕捉率": d.h10.sum() / d.tot.sum()})
R = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False)); R.to_csv(f"ml/compare4_pen{int(PEN)}.csv", index=False)
