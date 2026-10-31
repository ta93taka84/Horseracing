"""netkeibaから開催日→レースID→レース結果ページを取得してキャッシュする"""
import os, re, sys, time, random, json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
CACHE = "/tmp/nk_cache"
os.makedirs(CACHE, exist_ok=True)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def get(url, enc="euc-jp", tries=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode(enc, errors="replace")
        except Exception as e:
            time.sleep(min(2 ** i, 30) + random.random())
    raise RuntimeError("fail " + url)


def dates(start_ym, end_ym):
    out = []
    y, m = start_ym
    while (y, m) <= end_ym:
        s = get(f"https://race.netkeiba.com/top/calendar.html?year={y}&month={m}")
        out += [d for d in set(re.findall(r"kaisai_date=(\d{8})", s)) if d[:6] == f"{y}{m:02d}"]
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return sorted(out)


def race_ids(d):
    s = get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={d}", enc="utf-8")
    return sorted(set(re.findall(r"race_id=(\d{12})", s)))


def fetch_race(rid):
    p = os.path.join(CACHE, rid + ".html")
    if os.path.exists(p) and os.path.getsize(p) > 20000:
        return
    s = get(f"https://db.netkeiba.com/race/{rid}/")
    open(p, "w").write(s)
    time.sleep(0.5 + random.random() * 0.5)


if __name__ == "__main__":
    idx = os.path.join(DATA, "race_ids.json")
    if os.path.exists(idx):
        ids = json.load(open(idx))
    else:
        ds = [d for d in dates((2025, 8), (2026, 10)) if "20250801" <= d <= "20261025"]
        ids = {}
        for d in ds:
            ids[d] = [r for r in race_ids(d) if int(r[4:6]) <= 10]  # JRA場コード01-10
            time.sleep(0.3)
        json.dump(ids, open(idx, "w"), indent=0)
    allr = [r for v in ids.values() for r in v]
    print(len(ids), "days", len(allr), "races", flush=True)
    done = 0
    with ThreadPoolExecutor(3) as ex:
        for _ in ex.map(fetch_race, allr):
            done += 1
            if done % 200 == 0:
                print(done, flush=True)
    print("done")
