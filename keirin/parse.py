"""レース詳細ページのHTMLから、選手・ワイドオッズ・結果を取り出す。"""
import gzip
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

TIP = {"◎": 1, "○": 2, "▲": 3, "△": 4, "×": 5, "注": 6}
_ROW = re.compile(r'<tr>\s*<th class="n\d">(\d)</th>(.*?)</tr>', re.S)
_TD = re.compile(r'<td(?: class="empty")?>\s*([\d.]*)\s*</td>')


def trifecta_pair_probs(html):
    """3連単オッズから「2車とも3着以内」の確率を逆算する。戻り値: {(a, b): 確率}。"""
    a, b = html.find("3連単オッズ表 Start"), html.find("3連単オッズ表 End")
    if a < 0:
        return {}
    trio = {}
    for chunk in html[a:b].split('<table class="odds_table bt5')[1:]:
        m = re.search(r'<span class="number">(\d)</span>', chunk)
        if not m:
            continue
        cols = [int(x) for x in re.findall(r'<th class="n\d">(\d)</th>', chunk[:chunk.find('<td class="rider')])]
        for r, body in _ROW.findall(chunk):
            for c, v in zip(cols, _TD.findall(body)):
                if v and float(v) > 0:
                    key = tuple(sorted((int(m.group(1)), c, int(r))))
                    trio[key] = trio.get(key, 0) + 1 / float(v)
    total = sum(trio.values())
    out = {}
    for (i, j, k), q in trio.items():
        for pair in ((i, j), (i, k), (j, k)):
            out[pair] = out.get(pair, 0) + q / total
    return out


def _txt(el):
    return unicodedata.normalize("NFKC", el.get_text(" ", strip=True)) if el else ""


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def parse(html, race_id):
    """戻り値: {"race": {...}, "riders": [...], "odds": [...], "payouts": [...]}。出走表が無ければ None。"""
    so = BeautifulSoup(html, "lxml")
    tables = so.find_all("table", class_="racecard_table")
    if not tables:
        return None

    title = _txt(so.title)
    m = re.search(r"(\d+)R\s*(\S+)", title)
    bank = re.search(r"同走路年間勝利度数【(\d+)】", unicodedata.normalize("NFKC", html))
    race = {
        "race_id": race_id,
        "venue": race_id[:2],
        "date": (lambda d: f"{d[1]}-{int(d[2]):02d}-{int(d[3]):02d}" if d else None)(
            re.search(r"(\d{4})年(\d+)月(\d+)日", title)),
        "race_no": int(m.group(1)) if m else None,
        "race_class": m.group(2) if m else "",
        "grade": _txt(so.find(class_="icon_grade")),
        "bank": int(bank.group(1)) if bank else None,
    }

    # 出走表（1つ目の表）
    riders = {}
    for tr in tables[0].find_all("tr"):
        num_td, rider_td = tr.find("td", class_="num"), tr.find("td", class_="rider")
        if not num_td or not rider_td:
            continue
        num = int(_txt(num_td))
        home = _txt(rider_td.find(class_="home"))
        hp = [x.strip() for x in home.split("/")]
        after = [_txt(td) for td in rider_td.find_next_siblings("td")]
        v = [_num(x) for x in after[2:]]
        v += [None] * (15 - len(v))
        riders[num] = {
            "race_id": race_id, "num": num,
            "name": _txt(rider_td).replace(home, "").strip(),
            "pref": hp[0].replace(" ", "") if hp else "",
            "age": _num(hp[1]) if len(hp) > 1 else None,
            "term": _num(hp[2]) if len(hp) > 2 else None,
            "tip": TIP.get(_txt(tr.find("td", class_="tip")), 7),
            "evaluation": _num(_txt(tr.find("td", class_="evaluation"))),
            "rank_class": after[0] if after else "",
            "style": after[1] if len(after) > 1 else "",
            "gear": v[0], "score": v[1], "s_cnt": v[2], "b_cnt": v[3],
            "nige": v[4], "makuri": v[5], "sashi": v[6], "mark": v[7],
            "win1": v[8], "win2": v[9], "win3": v[10], "out": v[11],
            "win_rate": v[12], "top2_rate": v[13], "top3_rate": v[14],
        }
    if len(riders) < 5:
        return None

    # 直近の着順（今場所・前場所・前々場所）
    recent = so.find("table", class_="past_racecard_table")
    if recent:
        for tr in recent.find_all("tr"):
            num_td = tr.find("td", class_="num")
            if not num_td or int(_txt(num_td)) not in riders:
                continue
            t = _txt(tr)
            fin = [int(x) for x in re.findall(r"(\d)着", t)]
            r = riders[int(_txt(num_td))]
            r["recent_n"] = len(fin)
            r["recent_mean"] = sum(fin) / len(fin) if fin else None
            r["recent_top3"] = sum(f <= 3 for f in fin) / len(fin) if fin else None

    # 並び予想（ライン）
    line_id, pos = 0, 0
    lp = so.find(class_="line_position")
    if lp:
        for sp in lp.find_all("span", class_="icon_p"):
            t = _txt(sp)
            m = re.match(r"(\d)\s*(\S*)", t)
            if "space" in sp.get("class", []) or not m:
                line_id, pos = line_id + 1, 0
                continue
            num = int(m.group(1))
            if num in riders and "line_id" not in riders[num]:
                pos += 1
                riders[num].update(line_id=line_id, line_pos=pos, line_role=m.group(2))
    for r in riders.values():
        if "line_id" not in r:
            line_id += 1
            r.update(line_id=line_id, line_pos=1, line_role="")
    for r in riders.values():
        r["line_size"] = sum(x["line_id"] == r["line_id"] for x in riders.values())

    # 着順
    res = so.find("table", class_="result_table")
    if res:
        for tr in res.find_all("tr")[1:]:
            tds = tr.find_all("td")
            num_td = tr.find("td", class_="num")
            if len(tds) < 3 or not num_td or not _txt(num_td).isdigit():
                continue
            num = int(_txt(num_td))
            if num in riders:
                riders[num]["finish"] = _num(_txt(tds[1])) or 99  # 落車・失格などは99

    # ワイドのオッズ（「1=3 3.2～4.4」の形の行）
    odds = {}
    for m in re.finditer(r"(\d)=(\d) ([\d.]+) ?~ ?([\d.]+)", _txt(so)):
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        odds[(a, b)] = {"race_id": race_id, "a": a, "b": b,
                        "odds_min": float(m.group(3)), "odds_max": float(m.group(4))}

    # ワイドの払戻
    payouts = []
    w = so.select_one("table.refund_table td.wide")
    if w:
        for dl in w.find_all("dl"):
            m = re.fullmatch(r"(\d)=(\d)", _txt(dl.find("dt")))
            y = re.search(r"([\d,]+)円", _txt(dl.find("dd")))
            if m and y:
                a, b = sorted((int(m.group(1)), int(m.group(2))))
                payouts.append({"race_id": race_id, "a": a, "b": b, "payout": _num(y.group(1))})

    race["n_riders"] = len(riders)
    race["finished"] = any("finish" in r for r in riders.values())
    return {"race": race, "riders": list(riders.values()),
            "odds": list(odds.values()), "payouts": payouts, "p_tan": trifecta_pair_probs(html)}


def parse_file(path):
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return parse(f.read(), path.name[:16])
