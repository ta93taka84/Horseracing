"""楽天Kドリームスからレース詳細ページを収集する。

使い方: python -m keirin.scrape 2026-08-15 2026-11-14
取得済みのページは飛ばすので、途中で止めても再実行すれば続きから進む。
"""
import datetime as dt
import gzip
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://keirin.kdreams.jp"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
INTERVAL = 1.0  # 1ページごとの待ち時間（秒）
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64)"


def get(url):
    for i in range(4):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (i + 1))
    return None


def race_urls(day):
    html = get(f"{BASE}/racecard/{day:%Y/%m/%d}/")
    time.sleep(INTERVAL)
    if not html:
        return []
    return sorted(set(re.findall(r"https://keirin\.kdreams\.jp/[a-z]+/racedetail/\d{16}/", html)))


def fetch_race(url):
    """1レース分を取得して保存する。保存先のパスを返す。"""
    rid = re.search(r"(\d{16})", url).group(1)
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{rid}.html.gz"
    if not path.exists():
        html = get(url)
        time.sleep(INTERVAL)
        if not html:
            return None
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(html)
    return path


def main(start, end):
    day = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    while day <= last:
        urls = race_urls(day)
        n = sum(fetch_race(u) is not None for u in urls)
        print(f"{day} {n}/{len(urls)}", flush=True)
        day += dt.timedelta(days=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
