"""事前に決めた少数の候補だけを、304日のウォークフォワードで比べる(重みの総当たりはしない)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "ml")
from model3 import prep, feat_cols, fit_parts, part_scores, combine
FEAT = sys.argv[1] if len(sys.argv) > 1 else "ml/features2.pkl"
F = prep(pd.read_pickle(FEAT)); ALL, LR = feat_cols(F)
days = [d for d in sorted(F.date.unique()) if d >= pd.Timestamp("2026-01-15")]
cache = []; models = None; last = None
for t in days:
    if last is None or (t - last).days >= 14:
        models = fit_parts(F[F.date < t], ALL, LR); last = t
    te = F[F.date == t].reset_index(drop=True)
    cache.append((t, te[["machine", "y_diff", "y3", "eligible"]].copy(), part_scores(models, te)))
pd.to_pickle(cache, FEAT.replace(".pkl", "_cache3.pkl"))
C = {
    "ランダム(候補内)": None,
    "回転数ワースト7日": dict(G7=1),
    "旧モデル LR5+GBM5+G7": dict(LR5=1, GBM5=1, G7=1),
    "LR3+GBM3": dict(LR3=1, GBM3=1),
    "LR3+GBM3+G7": dict(LR3=1, GBM3=1, G7=1),
    "LR3+GBM3+G7+連続マイナス": dict(LR3=1, GBM3=1, G7=1, SM=1),
    "GBM3+GBM5+G7+連続マイナス": dict(GBM3=1, GBM5=1, G7=1, SM=1),
    "全部品(等重み)": dict(LR3=1, GBM3=1, LR5=1, GBM5=1, REG=1, G7=1, SM=1),
}
rows = []
for name, w in C.items():
    rec = []
    for t, te, s in cache:
        el = te["eligible"].values
        if w is None:
            v = 5 * te["y3"].values[el].mean(); l = 0
        else:
            top = np.argsort(-combine(te, s, w), kind="stable")[:5]
            v = te["y3"].values[top].sum(); l = te["y_diff"].values[top].mean() - te["y_diff"].values[el].mean()
        rec.append((t, v, l))
    d = pd.DataFrame(rec, columns=["t", "v", "l"])
    f = pd.qcut(d.index, 3, labels=["1月〜", "5月〜", "8月〜"])
    r = dict(モデル=name, 全期間=d.v.mean(), 誤差=d.v.std() / np.sqrt(len(d)), 平均比=d.l.mean())
    r.update(d.groupby(f).v.mean().to_dict()); r["現配置"] = d[d.t >= pd.Timestamp("2026-10-10")].v.mean()
    rows.append(r)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))
