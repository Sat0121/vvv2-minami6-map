"""v4モデル(GBM3+GBM5+GBM10+Q90+回転数ワースト7日+連続マイナス、候補内の順位を等重みで合計)
  python3 ml/predict3.py           翌日の推奨 → ml/prediction.json
  python3 ml/predict3.py history   10/10以降の各日の推奨(その日より前のデータだけで学習)→ ml/pred_history.json
"""
import sys, json, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from features import load, build
from model3 import combine, EXCLUDED, PREV_MAX, LAYOUT_START
from model4 import prep, feat_cols, fit_parts, part_scores

W = dict(GBM3=1, GBM5=1, GBM10=1, Q90=1, G7=1, SM=1)

def fit(tr, ALL):
    return fit_parts(tr, ALL)

def rank_day(models, te, ALL):
    s = part_scores(models, te)
    sc = combine(te, s, W)
    return te.assign(score=sc, p3=s["GBM3"], p5=s["GBM5"], p10=s["GBM10"]).sort_values("score", ascending=False)

def reason(r):
    return dict(no=int(r.machine), p=round(float(r.p3), 3), p5=round(float(r.p5), 3), p10=round(float(r.p10), 3),
                g7=None if pd.isna(r.games_s7) else int(r.games_s7 / 7), d9=None if pd.isna(r.diff_s9) else int(r.diff_s9),
                sm=int(r.streak_minus), prev=None if pd.isna(r.diff_s1) else int(r.diff_s1))

if __name__ == "__main__":
    df = load()
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        F = prep(build(df)); ALL, _ = feat_cols(F)
        hist = {}
        for t in [d for d in sorted(F.date.unique()) if d >= LAYOUT_START]:
            te = F[F.date == t].reset_index(drop=True)
            r = rank_day(fit(F[F.date < t], ALL), te, ALL)
            top = r[r.eligible].head(5)
            hist[pd.Timestamp(t).strftime("%Y-%m-%d")] = [[int(x.machine), round(float(x.p3), 3), int(x.y_diff), round(float(x.p10), 3)] for x in top.itertuples()]
            print(pd.Timestamp(t).date(), [h[0] for h in hist[pd.Timestamp(t).strftime("%Y-%m-%d")]], flush=True)
        json.dump(hist, open("ml/pred_history.json", "w"))
        n = len(hist); h = sum(sum(x[2] > 3000 for x in v) for v in hist.values())
        h10 = sum(sum(x[2] > 10000 for x in v) for v in hist.values())
        print(f"{n}日 上位5台中の+3000超え 平均{h/n:.2f}台 / 万枚 {h10}台")
    else:
        last = df.date.max(); nxt = last + pd.Timedelta(days=1)
        cur = df[df.date == last]
        ghost = cur.assign(date=nxt, diff=0, games=np.nan, bb=np.nan, rb=np.nan, bona=0)
        F = prep(build(pd.concat([df, ghost], ignore_index=True))); ALL, _ = feat_cols(F)
        tr = F[F.date <= last]; te = F[F.date == nxt].reset_index(drop=True)
        r = rank_day(fit(tr, ALL), te, ALL)
        ok = r[r.eligible]; ex = r[~r.eligible]
        out = dict(target=str(nxt.date()), based_on=str(last.date()), hit=3000, model="v4",
                   ranking=[reason(x) for x in ok.itertuples()],
                   excluded=[dict(no=int(x.machine), prev=None if pd.isna(x.diff_s1) else int(x.diff_s1), why="前日+3000超え" if (x.diff_s1 or 0) > PREV_MAX else "対象外の台") for x in ex.itertuples()])
        out["probs"] = {str(int(x.machine)): dict(p=round(float(x.p3), 3), p10=round(float(x.p10), 3), prev=None if pd.isna(x.diff_s1) else int(x.diff_s1)) for x in r.itertuples()}
        json.dump(out, open("ml/prediction.json", "w"), ensure_ascii=False)
        for i, o in enumerate(out["ranking"][:8], 1): print(i, o)
        print("除外:", [(e["no"], e["why"]) for e in out["excluded"]])
