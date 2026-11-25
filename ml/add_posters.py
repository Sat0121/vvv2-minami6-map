#!/usr/bin/env python3
"""AIPOS のポスター情報（画像URLだけ）を posters.json に追記する。画像は保存しない。
使い方: python3 add_posters.py 行ファイル   （- なら標準入力）
行の形（aipos_posters.js の出力）: YYYY-MM-DD 記事ID asabu=YYYYMM/ハッシュ[,…] eagle=… n24=… nishi=… minami6=…
このリポジトリで使う店だけを posters.json に入れ、同じ日付は上書きする。直近 KEEP_DAYS 日より古い日は消す。"""
import sys, json, re, pathlib, datetime as dt
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "posters.json"
HALLS = ("minami6",)
KEEP_DAYS = 90
IMG = "https://aipos.s3.ap-northeast-1.amazonaws.com/poster/{}"
POST = "https://hokkaido-aipos.com/archives/{}"
src = sys.stdin if len(sys.argv) < 2 or sys.argv[1] == "-" else open(sys.argv[1], encoding="utf-8")
data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
data.setdefault("source", "https://hokkaido-aipos.com/archives/category/ポスター情報")
days = data.setdefault("days", {})
n = 0
for line in src:
    t = line.split()
    if len(t) < 2 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", t[0]) or not t[1].isdigit(): continue
    day = days.setdefault(t[0], {})
    for kv in t[2:]:
        k, _, v = kv.partition("=")
        if k not in HALLS or not v: continue
        hashes = v.split(",")
        assert all(re.fullmatch(r"\d{6}/[0-9a-f]{32}(\.[a-z]+)?", x) for x in hashes), kv
        day[k] = {"post": POST.format(t[1]), "img": [IMG.format(x if "." in x else x + ".jpeg") for x in hashes]}
        n += 1
    if not day: del days[t[0]]
cut = (max(days) if days else "") and (dt.date.fromisoformat(max(days)) - dt.timedelta(days=KEEP_DAYS)).isoformat()
data["days"] = {d: days[d] for d in sorted(days) if d >= cut}
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"{n}件を反映。{OUT.name}: {len(data['days'])}日分（{min(data['days'])}〜{max(data['days'])}）")
