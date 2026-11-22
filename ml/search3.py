"""ウォークフォワードで部品スコアを作り、重みの組み合わせを前半で選んで後半で確かめる"""
import sys, itertools, json, numpy as np, pandas as pd
sys.path.insert(0, "ml")
from model3 import prep, feat_cols, fit_parts, part_scores, combine, PARTS

F = prep(pd.read_pickle("ml/features.pkl"))
ALL, LR = feat_cols(F)
days = [d for d in sorted(F.date.unique()) if d >= pd.Timestamp("2026-01-15")]
cache = []; models = None; last = None
for t in days:
    if last is None or (t - last).days >= 14:
        models = fit_parts(F[F.date < t], ALL, LR); last = t
    te = F[F.date == t].reset_index(drop=True)
    cache.append((t, te[["machine", "y_diff", "y3", "eligible"]].copy(), part_scores(models, te)))
    print(pd.Timestamp(t).date(), end=" ", flush=True)
print()
pd.to_pickle(cache, "ml/search3_cache.pkl")

def evaluate(w, sub):
    hits, rnd, lift = [], [], []
    for t, te, s in sub:
        sc = combine(te, s, w); top = np.argsort(-sc, kind="stable")[:5]
        el = te["eligible"].values
        hits.append(te["y3"].values[top].sum())
        rnd.append(5 * te["y3"].values[el].mean())
        lift.append(te["y_diff"].values[top].mean() - te["y_diff"].values[el].mean())
    return np.mean(hits), np.mean(rnd), np.mean(lift), np.std(hits) / np.sqrt(len(hits))

SPLIT = pd.Timestamp("2026-07-01")
sel = [c for c in cache if c[0] < SPLIT]; val = [c for c in cache if c[0] >= SPLIT]; cur = [c for c in cache if c[0] >= pd.Timestamp("2026-10-10")]
grid = [0, 0.5, 1]
rows = []
for ws in itertools.product(grid, repeat=len(PARTS)):
    if sum(ws) == 0: continue
    w = dict(zip(PARTS, ws))
    h, r, l, se = evaluate(w, sel)
    rows.append((h, ws))
rows.sort(reverse=True)
named = {
    "旧モデル(LR5+GBM5+G7)": dict(LR5=1, GBM5=1, G7=1),
    "回転数ワースト7日のみ": dict(G7=1),
    "LR3": dict(LR3=1), "GBM3": dict(GBM3=1), "LR3+GBM3": dict(LR3=1, GBM3=1),
}
out = []
for i, (h, ws) in enumerate(rows[:15]):
    named[f"探索{i+1}"] = dict(zip(PARTS, ws))
for name, w in named.items():
    a = evaluate(w, sel); b = evaluate(w, val); c = evaluate(w, cur)
    out.append(dict(name=name, w={k: v for k, v in w.items() if v}, sel=round(a[0], 3), val=round(b[0], 3), val_rand=round(b[1], 3), val_se=round(b[3], 3), val_lift=round(b[2]), cur=round(c[0], 3), cur_rand=round(c[1], 3)))
R = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(R.to_string())
R.to_csv("ml/search3_result.csv", index=False)
