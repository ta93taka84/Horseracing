"""キャッシュしたレース結果HTMLを data/results.csv（1頭1行）と data/races.csv に変換"""
import os, re, glob, html, csv

CACHE = "/tmp/nk_cache"
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def txt(x):
    return html.unescape(re.sub(r"<[^>]+>", "", x)).replace("\xa0", " ").strip()


def parse(path):
    s = open(path).read()
    rid = os.path.basename(path)[:12]
    if "race_table_01" not in s:
        return None, []
    intro = s[s.find("data_intro"):s.find("race_table_01")]
    name = txt(re.search(r"<h1>(.*?)</h1>", intro, re.S).group(1))
    span = txt(re.search(r"<span>(.*?)</span>", intro, re.S).group(1))
    small = txt(re.search(r'class="smalltxt">(.*?)</p>', intro, re.S).group(1))
    m = re.search(r"(芝|ダ|障)(右|左|直線)?\s*(?:外|内|2周)?.*?(\d+)m", span)
    surf, turn, dist = m.group(1), m.group(2) or "", int(m.group(3))
    inner = "外" if re.search(r"(芝|ダ)(右|左)?\s*外", span) else ""
    weather = (re.search(r"天候 : (\S+)", span) or [None, ""])[1]
    going = (re.search(r"(?:芝|ダート|障害)\s*:\s*(\S+)", span) or [None, ""])[1]
    date = re.search(r"(\d{4})年(\d+)月(\d+)日", small)
    date = f"{date.group(1)}-{int(date.group(2)):02d}-{int(date.group(3)):02d}"
    race = dict(race_id=rid, date=date, place=rid[4:6], rno=int(rid[10:]), name=name, surface=surf,
                turn=turn, outer=inner, dist=dist, weather=weather, going=going, cond=small)
    tbl = s[s.find("race_table_01"):]
    tbl = tbl[:tbl.find("</table>")]
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", tbl, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(tds) < 20:
            continue
        t = [txt(x) for x in tds]
        horse = re.search(r"/horse/(\d+)/", tr)
        jockey = re.search(r"/jockey/result/recent/(\w+)/", tr)
        trainer = re.search(r"/trainer/result/recent/(\w+)/", tr)
        owner = re.search(r"/owner/result/recent/(\w+)/", tr)
        # 列: 着,枠,馬番,馬名,性齢,斤量,騎手,タイム,着差,指数x5,通過,上り,単勝,人気,馬体重,調教,コメント,備考,調教師,馬主,賞金
        bw = re.match(r"(\d+)\(([+-]?\d+)\)", t[18])
        rows.append(dict(
            race_id=rid, rank=t[0], waku=t[1], umaban=t[2], horse=t[3], horse_id=horse.group(1) if horse else "",
            sexage=t[4], kinryo=t[5], jockey=t[6], jockey_id=jockey.group(1) if jockey else "",
            time=t[7], margin=t[8], passing=t[14], last3f=t[15], odds=t[16], pop=t[17],
            weight=bw.group(1) if bw else "", wdiff=bw.group(2) if bw else "",
            trainer=re.sub(r"^\[.\]\s*", "", t[22]), trainer_area=(re.match(r"\[(.)\]", t[22]) or [None, ""])[1],
            trainer_id=trainer.group(1) if trainer else "", owner_id=owner.group(1) if owner else "",
            prize=t[24].replace(",", "")))
    # 複勝払戻
    pay = {}
    for tr in re.findall(r"<tr>(.*?)</tr>", s[s.find("pay_table_01"):s.find("pay_table_01") + 8000], re.S):
        th = re.search(r"<th[^>]*>(.*?)</th>", tr, re.S)
        if th and txt(th.group(1)) == "複勝":
            tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
            nums = re.split(r"<br\s*/?>", tds[0]); amts = re.split(r"<br\s*/?>", tds[1])
            for n, a in zip(nums, amts):
                if txt(n):
                    pay[txt(n)] = txt(a).replace(",", "")
    for r in rows:
        r["fukusho_pay"] = pay.get(r["umaban"], "0")
    race["n_runners"] = len(rows)
    return race, rows


if __name__ == "__main__":
    races, rows = [], []
    for p in sorted(glob.glob(CACHE + "/*.html")):
        try:
            a, b = parse(p)
        except Exception as e:
            print("ERR", p, e); continue
        if a:
            races.append(a); rows += b
    for name, data in (("races.csv", races), ("results.csv", rows)):
        with open(os.path.join(DATA, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print(len(races), "races", len(rows), "rows")
