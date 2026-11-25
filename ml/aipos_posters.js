// AIPOS（hokkaido-aipos.com）の「ポスター情報」から、対象店のポスター画像URLを集める。
// 使い方: 内蔵ブラウザで https://hokkaido-aipos.com/archives/category/%e3%83%9d%e3%82%b9%e3%82%bf%e3%83%bc%e6%83%85%e5%a0%b1 を開き、
// このファイルの中身をページで実行する。SINCE 以降の日付の「札幌近郊ポスター一覧」を読み、
// 1日1行「YYYY-MM-DD 記事ID asabu=YYYYMM/ハッシュ eagle=... n24=... nishi=... minami6=...」を返す。
// 返った文字列を add_posters.py に渡す。画像そのものは保存しない（URLだけ）。
const SINCE = (() => { const d = new Date(Date.now() + 9 * 3600e3); d.setUTCDate(d.getUTCDate() - 3); return d.toISOString().slice(0, 10); })();  // 既定: 3日前から
const STORES = { asabu: ["ハッピー麻生"], eagle: ["イーグル麻生"], n24: ["ハッピー二十四軒"], nishi: ["ハッピー西町"], minami6: ["ハッピー南六条", "ハッピー南6条"] };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const BASE = "/archives/category/%E3%83%9D%E3%82%B9%E3%82%BF%E3%83%BC%E6%83%85%E5%A0%B1";
const posts = [];
for (let p = 1; p <= 10; p++) {
  const d = new DOMParser().parseFromString(await fetch(p === 1 ? BASE : `${BASE}/page/${p}`).then(r => r.text()), "text/html");
  let older = false;
  for (const a of d.querySelectorAll('a[href*="/archives/"]')) {
    const m = a.textContent.trim().match(/(\d{4})年(\d{1,2})月(\d{1,2})日札幌近郊ポスター一覧/); if (!m) continue;
    const date = `${m[1]}-${m[2].padStart(2, "0")}-${m[3].padStart(2, "0")}`;
    if (date < SINCE) { older = true; continue; }
    if (!posts.some(x => x.url === a.href)) posts.push({ date, url: a.href });
  }
  if (older) break; await sleep(800);
}
const lines = [];
for (const p of posts.sort((a, b) => a.date.localeCompare(b.date))) {
  const d = new DOMParser().parseFromString(await fetch(p.url).then(r => r.text()), "text/html");
  const map = {}; let cur = null;
  for (const el of d.querySelector("section.entry-content").children) {
    if (el.tagName === "H2") { cur = el.textContent.normalize("NFKC").trim().replace(/^[^\p{L}\p{N}]+/u, ""); map[cur] = map[cur] || []; }  // 先頭の⭐などは外す
    else if (cur) el.querySelectorAll("img").forEach(i => map[cur].push(i.getAttribute("src")));
  }
  const cols = Object.entries(STORES).map(([k, names]) => { const n = names.find(n => map[n] && map[n].length); return n ? `${k}=${[...new Set(map[n])].map(u => u.replace(/^https:\/\/aipos\.s3\.ap-northeast-1\.amazonaws\.com\/poster\//, "").replace(/\.jpeg$/, "")).join(",")}` : null; }).filter(Boolean);
  lines.push(`${p.date} ${p.url.split("/").pop()} ${cols.join(" ")}`);
  await sleep(700);
}
lines.join("\n")
