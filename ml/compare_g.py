"""ギャン中モデルの候補比較: 前半(1/15〜6/30)で選び、後半(7/1〜11/21)と現配置で確認"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "ml")
from gamble import prep, feat_cols, fit_parts, part_scores, combine_g, day_score
F = prep(pd.read_pickle("ml/features4.pkl")); ALL, _ = feat_cols(F)
days = [d for d in sorted(F.date.unique()) if d >= pd.Timestamp("2026-01-15")]
cache = []; models = None; last = None
for t in days:
    if last is None or (t - last).days >= 14:
        models = fit_parts(F[F.date < t], ALL); last = t
    te = F[F.date == t].reset_index(drop=True)
    cache.append((t, te[["machine", "y_diff", "elig_g", "drank", "eligible"]].copy(), part_scores(models, te)))
pd.to_pickle(cache, "ml/compare_g_cache.pkl")
C = {
    "ランダム": None,
    "AI Recommend v4(参考)": "v4",
    "GG(専用ラベル)": dict(GG=1),
    "GVAL(点数を直接予測)": dict(GVAL=1),
    "G10(万枚確率)": dict(G10=1),
    "Q95(上振れ)": dict(Q95=1),
    "GG+G10": dict(GG=1, G10=1),
    "GG+Q95": dict(GG=1, Q95=1),
    "GG+G10+Q95": dict(GG=1, G10=1, Q95=1),
    "GG+G10+Q90+TOP2": dict(GG=1, G10=1, Q90=1, TOP2=1),
    "GG+G10+Q95+GVAL": dict(GG=1, G10=1, Q95=1, GVAL=1),
    "GG+G10+Q95+G7+SM": dict(GG=1, G10=1, Q95=1, G7=1, SM=1),
    "GG+G10+Q95+よく回る台": dict(GG=1, G10=1, Q95=1, GAMES30=1),
}
v4 = {t: s for t, te, s in pd.read_pickle("ml/compare4_cache.pkl")}
from model3 import combine
rng = np.random.default_rng(0); rows = []
for name, w in C.items():
    rec = []
    for t, te, s in cache:
        el = np.where(te["elig_g"].values)[0]
        if w is None:
            r = np.array([day_score(te, rng.choice(el, 5, replace=False)) for _ in range(300)]).mean(0)
        elif w == "v4":
            top = np.argsort(-combine(te, v4[t], dict(GBM3=1, GBM5=1, GBM10=1, Q90=1, G7=1, SM=1)), kind="stable")[:5]; r = day_score(te, top)
        else:
            r = day_score(te, np.argsort(-combine_g(te, s, w), kind="stable")[:5])
        rec.append((t, *r))
    d = pd.DataFrame(rec, columns=["t", "S", "big", "bigday"])
    A = d[d.t < pd.Timestamp("2026-07-01")]; B = d[d.t >= pd.Timestamp("2026-07-01")]; Cc = d[d.t >= pd.Timestamp("2026-10-10")]
    strong = d[d.bigday == 1]; weak = d[d.bigday == 0]
    rows.append({"モデル": name, "前半": A.S.mean(), "後半": B.S.mean(), "現配置": Cc.S.mean(), "全期間": d.S.mean(), "±": d.S.std() / np.sqrt(len(d)),
                 "万枚の日:捕捉台数/日": strong.S.mean(), "万枚捕捉率": strong.S.sum() / strong.big.sum(), "弱い日:点/日": weak.S.mean()})
R = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False)); R.to_csv("ml/compare_g_result.csv", index=False)
print("日数", len(cache), "万枚のある日", int(d.bigday.sum()))
