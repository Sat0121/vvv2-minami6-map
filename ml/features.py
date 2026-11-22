"""ヴァルヴレイヴ2 南6条 翌日+5000枚超え予測用の特徴量作成
入力: vvv2_1y.csv (date,machine,diff,games,bb,rb)
各行 = (予測対象日 t, 台)。特徴量は t より前のデータだけから作る。
"""
import numpy as np, pandas as pd

TARGET = 5000

def load(path="vvv2_1y.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df["machine"] = df["machine"].astype(int)
    df["bona"] = df["bb"].fillna(0) + df["rb"].fillna(0)
    df["games"] = df["games"].astype(float)
    df.loc[df["games"] <= 0, "games"] = np.nan  # 0Gは稼働なし
    return df.sort_values(["date", "machine"]).reset_index(drop=True)

def build(df):
    days = sorted(df["date"].unique())
    piv = {k: df.pivot(index="date", columns="machine", values=k).reindex(days) for k in ("diff", "games", "bona", "rb")}
    D, G, B = piv["diff"], piv["games"], piv["bona"]
    present = D.notna()
    feats = {}
    sh = lambda X, k=1: X.shift(k)
    roll = lambda X, n: X.shift(1).rolling(n, min_periods=n).sum()
    for n in (1, 2, 3, 5, 7, 9):
        feats[f"diff_s{n}"] = roll(D, n)
        feats[f"games_s{n}"] = roll(G.fillna(0), n).where(roll(present.astype(float), n) == n)
        feats[f"bona_s{n}"] = roll(B, n)
    R = piv["rb"]
    for n in (1, 3, 7):
        g = roll(G.fillna(0), n); b = roll(B, n)
        feats[f"bprob_s{n}"] = (b / g.replace(0, np.nan))          # ボーナス確率(回数/回転)
        feats[f"rbshare_s{n}"] = roll(R, n) / b.replace(0, np.nan)  # ボーナスのうちRBの割合
    feats["bprob_m30"] = (B.shift(1).rolling(30, min_periods=5).sum() / G.shift(1).rolling(30, min_periods=5).sum())
    big = (D > 10000).astype(float).where(present)
    feats["n10k_r30"] = big.shift(1).rolling(30, min_periods=5).sum()
    feats["n10k_r90"] = big.shift(1).rolling(90, min_periods=10).sum()
    feats["max_d30"] = D.shift(1).rolling(30, min_periods=5).max()
    feats["n3k_r30"] = (D > 3000).astype(float).where(present).shift(1).rolling(30, min_periods=5).mean()
    feats["games_m30"] = G.shift(1).rolling(30, min_periods=5).mean()
    feats["diff_m30"] = D.shift(1).rolling(30, min_periods=5).mean()
    feats["hit5k_r30"] = (D > TARGET).astype(float).where(present).shift(1).rolling(30, min_periods=5).mean()
    feats["hist_n"] = present.astype(float).shift(1).cumsum()
    # 連続マイナス/プラス日数
    def streak(cond):
        out = pd.DataFrame(0.0, index=D.index, columns=D.columns); cur = np.zeros(D.shape[1])
        c = np.asarray(cond)
        for i in range(len(D)):
            out.iloc[i] = cur
            cur = np.where(c[i], cur + 1, 0)
        return out
    feats["streak_minus"] = streak((D < 0).values & present.values)
    feats["streak_plus"] = streak((D > 0).values & present.values)
    # 機械割(前日)
    feats["pay_1"] = (1 + D / (3 * G)).shift(1)
    # 前日の隣(番号順)の差枚: 台番号順に左右
    prevD = D.shift(1)
    left = pd.DataFrame(np.nan, index=D.index, columns=D.columns); right = left.copy()
    for t in D.index:
        cols = [c for c in D.columns if present.loc[t, c] or (t in prevD.index and not np.isnan(prevD.loc[t, c]))]
        cols = sorted(cols)
        for i, c in enumerate(cols):
            if i > 0: left.loc[t, c] = prevD.loc[t, cols[i-1]]
            if i < len(cols) - 1: right.loc[t, c] = prevD.loc[t, cols[i+1]]
    feats["nb_left_1"], feats["nb_right_1"] = left, right
    feats["nb_max_1"] = np.fmax(left, right)
    # 長期の台クセ(その台の過去全日の平均差枚 − その日の平均)
    rel = D.sub(D.mean(axis=1), axis=0)
    feats["kuse"] = rel.shift(1).expanding(min_periods=10).mean()

    rows = []
    for name, X in feats.items():
        s = X.stack(future_stack=True).rename(name); rows.append(s)
    F = pd.concat(rows, axis=1)
    F.index.names = ["date", "machine"]
    F = F.join(D.stack().rename("y_diff"), how="inner")  # 当日に稼働データがある台だけ
    F = F.join(G.stack(future_stack=True).rename("y_games"))
    F = F.reset_index()
    # その日の中での順位(パーセンタイル): ワースト/ベスト系の代わり
    rank_cols = [c for c in F.columns if c.startswith(("diff_s", "games_s", "bona_s", "bprob_", "rbshare_"))] + ["games_m30", "diff_m30", "pay_1"]
    for c in rank_cols:
        F[c + "_rk"] = F.groupby("date")[c].rank(pct=True)
        F[c + "_rkmin"] = F.groupby("date")[c].rank(method="min")  # 1=ワースト1
    # 日付と台番号
    F["dom"] = F["date"].dt.day; F["dow"] = F["date"].dt.dayofweek
    F["tail"] = F["machine"] % 10
    F["tail_match"] = (F["tail"] == F["dom"] % 10).astype(int)
    F["is8"] = (F["dom"] % 10 == 8).astype(int)
    F["prev_hall_avg"] = F["date"].map(D.mean(axis=1).shift(1))
    F["layout"] = (F["date"] >= "2026-10-10").astype(int)
    F = F[F["y_diff"].notna()].copy()
    F["y"] = (F["y_diff"] > TARGET).astype(int)
    return F
