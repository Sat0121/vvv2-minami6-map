"""ギャン中 Recommend(GG+万枚確率+上振れ95%点+回転数ワースト7日+連続マイナス、候補内の順位を等重み)
  python3 ml/predict_g.py           翌日の推奨 → ml/prediction_g.json
  python3 ml/predict_g.py history   10/10以降の各日の推奨(その日より前のデータだけで学習)→ ml/pred_history_g.json
"""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from features import load, build
from model3 import LAYOUT_START
from gamble import prep, feat_cols, fit_parts, part_scores, combine_g, BIG

W = dict(GG=1, G10=1, Q95=1, G7=1, SM=1)
def rank_day(tr, te, ALL):
    s = part_scores(fit_parts(tr, ALL), te)
    return te.assign(score=combine_g(te, s, W), p10=s["G10"], q95=s["Q95"]).sort_values("score", ascending=False)

if __name__ == "__main__":
    df = load()
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        F = prep(build(df)); ALL, _ = feat_cols(F); hist = {}
        for t in [d for d in sorted(F.date.unique()) if d >= LAYOUT_START]:
            te = F[F.date == t].reset_index(drop=True)
            top = rank_day(F[F.date < t], te, ALL).head(5)
            k = pd.Timestamp(t).strftime("%Y-%m-%d")
            hist[k] = [[int(x.machine), round(float(x.p10), 3), int(x.y_diff), int(x.drank)] for x in top.itertuples()]
            print(k, [h[0] for h in hist[k]], flush=True)
        json.dump(hist, open("ml/pred_history_g.json", "w"))
    else:
        last = df.date.max(); nxt = last + pd.Timedelta(days=1)
        ghost = df[df.date == last].assign(date=nxt, diff=0, games=np.nan, bb=np.nan, rb=np.nan, bona=0)
        F = prep(build(pd.concat([df, ghost], ignore_index=True))); ALL, _ = feat_cols(F)
        te = F[F.date == nxt].reset_index(drop=True)
        r = rank_day(F[F.date <= last], te, ALL); r = r[r.elig_g]
        out = dict(target=str(nxt.date()), based_on=str(last.date()),
                   ranking=[dict(no=int(x.machine), p10=round(float(x.p10), 3), q95=int(x.q95), prev=None if pd.isna(x.diff_s1) else int(x.diff_s1),
                                 g7=None if pd.isna(x.games_s7) else int(x.games_s7 / 7), sm=int(x.streak_minus)) for x in r.head(10).itertuples()])
        json.dump(out, open("ml/prediction_g.json", "w"), ensure_ascii=False)
        for i, o in enumerate(out["ranking"][:5], 1): print(i, o)
