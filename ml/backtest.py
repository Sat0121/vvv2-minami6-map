"""ウォークフォワード検証: その日より前のデータだけで学習し、その日の上位5台を選ぶ"""
import sys, numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "ml"); from features import load, build
warnings.filterwarnings("ignore")

F = build(load())
F.to_pickle("ml/features.pkl")
EXCL = {"date", "machine", "y", "y_diff", "y_games"}
ALL = [c for c in F.columns if c not in EXCL]
LR_FEATS = [c for c in ALL if c.endswith("_rk")] + ["streak_minus", "streak_plus", "kuse", "hit5k_r30", "nb_max_1", "tail_match", "pay_1"]

def models():
    return {
        "LR": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "GBM": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=0),
    }
FEATS = {"LR": LR_FEATS, "GBM": ALL}

def run(start="2026-01-15", step=14, k=5):
    days = sorted(F["date"].unique()); test_days = [d for d in days if d >= pd.Timestamp(start)]
    res = []; fitted = {}; last_fit = None
    for t in test_days:
        if last_fit is None or (t - last_fit).days >= step:
            tr = F[F["date"] < t]
            for name, mk in models().items():
                m = mk(); m.fit(tr[FEATS[name]], tr["y"]); fitted[name] = m
            last_fit = t
        te = F[F["date"] == t].copy()
        avg, rate = te["y_diff"].mean(), te["y"].mean()
        scores = {name: fitted[name].predict_proba(te[FEATS[name]])[:, 1] for name in fitted}
        scores["回転数ワースト(5日)"] = -te["games_s5"].fillna(1e9).values
        scores["回転数ワースト(7日平均の少なさ)"] = -te["games_s7"].fillna(1e9).values
        scores["差枚ワースト(8日)"] = -te["diff_s8"].fillna(1e9).values if "diff_s8" in te else -te["diff_s9"].fillna(1e9).values
        scores["LR+GBM"] = pd.Series(scores["LR"]).rank().values + pd.Series(scores["GBM"]).rank().values
        for name, s in scores.items():
            idx = np.argsort(-np.asarray(s), kind="stable")[:k]; top = te.iloc[idx]
            res.append(dict(date=t, model=name, hits=top["y"].sum(), any=int(top["y"].sum() > 0), lift=top["y_diff"].mean() - avg, rand=k * rate))
    R = pd.DataFrame(res)
    R.to_csv("ml/backtest_days.csv", index=False)
    return R

def summary(R):
    R = R.copy(); R["half"] = np.where(R["date"] < R["date"].quantile(0.5), "前半", "後半")
    g = R.groupby("model").agg(日数=("hits", "size"), 的中_5台中=("hits", "mean"), ランダム=("rand", "mean"), 一台以上の日=("any", "mean"), 平均比_枚=("lift", "mean"))
    sd = R.groupby("model")["hits"].std() / np.sqrt(R.groupby("model")["hits"].size())
    g["±"] = sd
    h = R.pivot_table(index="model", columns="half", values="hits", aggfunc="mean")
    h2 = R[R["date"] >= "2026-10-10"].groupby("model")["hits"].mean().rename("現配置")
    return g.join(h).join(h2).sort_values("的中_5台中", ascending=False)

if __name__ == "__main__":
    R = run()
    pd.set_option("display.width", 200)
    print(summary(R).round(3))
