"""みんレポ(min-repo.com)の1日分を vvv2_1y.csv に追記する。
使い方: python3 ml/add_minrepo.py YYYY-MM-DD "1311|差枚|G数|BB|RB 1312|... ..."
  (みんレポの機種ページで取った「台番号|差枚|G数|BB|RB」を空白区切りで)

・slot-eye の回転数はボーナス・AT中のゲームを含まない数え方で、みんレポのG数より多い。
  2026/11/20・21 の全42台を照合して
      slot-eye の回転数 ≒ みんレポのG数 − 17.6×BB − 12.6×RB
  で換算できることを確かめた(換算後の誤差は±70G程度。差枚は±50枚以内、BB・RBは一致)。
・換算前の値は data/minrepo_raw.csv にそのまま残す。
・台数・台番号が現配置(42台)と合わなければ何も書かずにエラーで終わる。
"""
import sys, csv, os
K_BB, K_RB = 17.6, 12.6
EXPECT = [1311,1312,1313,1315,1316,1317,1333,1335,1336,1337,1338,1350,1351,1352,1353,1355,1356,1357,1358,1360,1361,1362,1363,1365,1366,1367,1368,1370,1371,1372,1373,1375,1376,1377,1378,1380,1381,1382,1383,1385,1386,2218]

def main(day, text):
    rows = {}
    for tok in text.split():
        p = tok.replace(",", "").replace("+", "").split("|")
        if len(p) != 5: sys.exit(f"形式が違う: {tok}")
        no, diff, g, bb, rb = (int(x) for x in p)
        rows[no] = (diff, g, bb, rb)
    if sorted(rows) != EXPECT:
        sys.exit(f"台番号が現配置と合わない(配置変更の可能性)。足りない: {sorted(set(EXPECT)-set(rows))} 余分: {sorted(set(rows)-set(EXPECT))}")
    have = {r[0] for r in csv.reader(open("vvv2_1y.csv"))}
    if day in have: sys.exit(f"{day} はすでに vvv2_1y.csv にある")
    out = []
    for no in EXPECT:
        diff, g, bb, rb = rows[no]
        games = 0 if g == 0 else max(0, round(g - K_BB * bb - K_RB * rb))
        out.append([day, no, diff, games, bb, rb])
    with open("vvv2_1y.csv", "a", newline="") as f: csv.writer(f, lineterminator="\r\n").writerows(out)
    raw = "data/minrepo_raw.csv"; new = not os.path.exists(raw)
    with open(raw, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new: w.writerow(["date", "machine", "diff", "games_minrepo", "bb", "rb"])
        w.writerows([[day, no, *rows[no]] for no in EXPECT])
    avg = sum(r[2] for r in out) / len(out)
    print(f"{day}: {len(out)}台を追記  平均差枚 {avg:+.0f}  平均G(換算後) {sum(r[3] for r in out)/len(out):.0f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
