"""+3000枚狙いモデル: 候補ルールと部品モデル、ウォークフォワードのスコア計算
ルール:
  - 前日の差枚が+3000枚を超えた台は狙わない
  - 1311〜1316 と 2218(見世物小屋)は狙わない(現配置 2026/10/10〜)
目的: 推奨5台のうち、当日+3000枚を超えた台数を最大化
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

HIT = 3000
PREV_MAX = 3000
EXCLUDED = {1311, 1312, 1313, 1315, 1316, 2218}
LAYOUT_START = pd.Timestamp("2026-10-10")

def prep(F):
    F = F.copy()
    F["y3"] = (F["y_diff"] > HIT).astype(int)
    F["y5"] = (F["y_diff"] > 5000).astype(int)
    F["y_clip"] = F["y_diff"].clip(-6000, 15000)
    ex = F["machine"].isin(EXCLUDED) & (F["date"] >= LAYOUT_START)
    F["eligible"] = ~((F["diff_s1"] > PREV_MAX) | ex)
    return F

EXCL = {"date", "machine", "y", "y_diff", "y_games", "y3", "y5", "y_clip", "eligible"}
def feat_cols(F):
    allc = [c for c in F.columns if c not in EXCL]
    lr = [c for c in allc if c.endswith("_rk")] + ["streak_minus", "streak_plus", "kuse", "hit5k_r30", "nb_max_1", "tail_match", "pay_1"]
    return allc, lr

def lr_model(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
def gbc(): return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=0)
def gbr(): return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=0)

def fit_parts(tr, ALL, LR):
    """部品モデルを学習して、予測関数の辞書を返す"""
    m = {}
    a = lr_model().fit(tr[LR], tr["y3"]); m["LR3"] = lambda X, a=a: a.predict_proba(X[LR])[:, 1]
    b = gbc().fit(tr[ALL], tr["y3"]); m["GBM3"] = lambda X, b=b: b.predict_proba(X[ALL])[:, 1]
    c = lr_model().fit(tr[LR], tr["y5"]); m["LR5"] = lambda X, c=c: c.predict_proba(X[LR])[:, 1]
    d = gbc().fit(tr[ALL], tr["y5"]); m["GBM5"] = lambda X, d=d: d.predict_proba(X[ALL])[:, 1]
    e = gbr().fit(tr[ALL], tr["y_clip"]); m["REG"] = lambda X, e=e: e.predict(X[ALL])
    return m

PARTS = ["LR3", "GBM3", "LR5", "GBM5", "REG", "G7", "SM"]
def part_scores(models, te):
    s = {k: f(te) for k, f in models.items()}
    s["G7"] = -te["games_s7"].fillna(1e9).values           # 直近7日の回転数が少ない
    s["SM"] = te["streak_minus"].fillna(0).values            # 連続マイナス日数
    return s

def combine(te, s, w):
    """候補だけで順位(0〜1)を付け、重み付きで足す"""
    el = te["eligible"].values
    tot = np.zeros(len(te))
    for k, wk in w.items():
        if wk == 0: continue
        r = pd.Series(np.where(el, s[k], np.nan)).rank(pct=True).fillna(0).values
        tot += wk * r
    tot[~el] = -1e9
    return tot
