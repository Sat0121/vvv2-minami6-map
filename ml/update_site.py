"""index.html のデータ(SRC: 現配置の全日)と おすすめ5台(PRED) を書き換える
使い方: python3 ml/update_site.py index.html
"""
import sys, re, json, pandas as pd
LAYOUT_START = "2026-10-10"  # 現在の台配置になった日
html_path = sys.argv[1]
df = pd.read_csv("vvv2_1y.csv", parse_dates=["date"])
cur = df[df.date >= LAYOUT_START]
dates = sorted(cur.date.unique())
m = {}
for no, g in cur.groupby("machine"):
    g = g.set_index("date").reindex(dates)
    col = lambda k: [None if pd.isna(v) else int(v) for v in g[k]]
    m[str(no)] = [col("diff"), col("games"), col("bb"), col("rb")]
src = {"dates": [pd.Timestamp(d).strftime("%m/%d") for d in dates], "m": m}
pred = json.load(open("ml/prediction.json"))
h = open(html_path, encoding="utf8").read()
h = re.sub(r"const SRC = \{.*?\};\n", lambda _: "const SRC = " + json.dumps(src, separators=(",", ":")) + ";\n", h, count=1, flags=re.S)
h = re.sub(r"const PRED = \{.*?\};\n", lambda _: "const PRED = " + json.dumps(pred, ensure_ascii=False, separators=(",", ":")) + ";\n", h, count=1, flags=re.S)
open(html_path, "w", encoding="utf8").write(h)
print("dates", len(dates), src["dates"][0], "-", src["dates"][-1], "machines", len(m), "pred for", pred["target"])
