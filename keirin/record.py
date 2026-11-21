"""締切2分前の買い目を記録し、確定後の結果と突き合わせる（実戦に近い条件での成績測定）。

使い方:
  python -m keirin.record run [分]   # 指定した分数（既定110分）だけ記録して終了する。再実行で続きから進む
  python -m keirin.record report     # これまでの記録を集計する

記録は data/live/ に保存する（リポジトリには入れない）。
"""
import datetime as dt
import gzip
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import value
from .features import pairs
from .parse import parse
from .scrape import INTERVAL, get, race_urls

JST = ZoneInfo("Asia/Tokyo")
OUT = Path(__file__).resolve().parent.parent / "data" / "live"
SNAP_BEFORE = dt.timedelta(seconds=120)  # 締切の何秒前に記録するか
RESULT_AFTER = dt.timedelta(minutes=12)  # 締切の何分後から結果を取りに行くか
RESULT_GIVEUP = dt.timedelta(hours=3)


def now():
    return dt.datetime.now(JST)


def rid_of(url):
    return re.search(r"(\d{16})", url).group(1)


def schedule(day):
    """その日のレースの締切時刻。初回だけ全レースのページを取得し、以後はファイルから読む。"""
    path = OUT / f"schedule_{day:%Y%m%d}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = []
    for u in race_urls(day):
        html = get(u)
        time.sleep(INTERVAL)
        if not html:
            continue
        m = re.search(r"投票締切[\s\S]{0,60}?(\d{1,2}):(\d{2})", unicodedata.normalize("NFKC", html))
        if not m:
            continue
        h, mi = int(m.group(1)), int(m.group(2))
        close = dt.datetime(day.year, day.month, day.day, h % 24, mi, tzinfo=JST)
        if h >= 24:
            close += dt.timedelta(days=1)
        rows.append({"url": u, "rid": rid_of(u), "close": close.isoformat()})
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def grab(r, kind, calib):
    html = get(r["url"])
    time.sleep(INTERVAL)
    d = parse(html, r["rid"]) if html else None
    if not d:
        return None
    if kind == "final" and not d["race"]["finished"]:
        return None
    if kind == "snap":  # 後から別の判定方法でも検証できるよう、締切2分前のページも残す
        (OUT / "html").mkdir(exist_ok=True)
        with gzip.open(OUT / "html" / f"snap_{r['rid']}.html.gz", "wt", encoding="utf-8") as f:
            f.write(html)
    df = pairs(d)
    if df["odds_min"].isna().all() or df["p_tan"].isna().all():
        return None
    df = value.add_ev(df, calib)
    df["at"] = now().isoformat()
    df["close"] = r["close"]
    df["slug"] = r["url"].split("/")[3]
    df["race_no"] = d["race"]["race_no"]
    df.to_parquet(OUT / f"{kind}_{r['rid']}.parquet")
    return df


def run(minutes=110):
    end = now() + dt.timedelta(minutes=minutes)
    calib = value.load()
    OUT.mkdir(parents=True, exist_ok=True)
    tried = {}
    while now() < end:
        day = now().date()
        prev = OUT / f"schedule_{day - dt.timedelta(days=1):%Y%m%d}.json"
        races = schedule(day) + (json.loads(prev.read_text()) if prev.exists() else [])
        for r in races:
            r["close_dt"] = dt.datetime.fromisoformat(r["close"])
        # 結果の取得（締切12分後〜3時間後、1レース5分おき）
        for r in races:
            snap, final = OUT / f"snap_{r['rid']}.parquet", OUT / f"final_{r['rid']}.parquet"
            t = now()
            if snap.exists() and not final.exists() and r["close_dt"] + RESULT_AFTER < t < r["close_dt"] + RESULT_GIVEUP:
                if t - tried.get(r["rid"], t - dt.timedelta(hours=1)) >= dt.timedelta(minutes=5):
                    tried[r["rid"]] = t
                    ok = grab(r, "final", calib) is not None
                    print(f"{t:%m/%d %H:%M} 結果 {r['rid']} {'取得' if ok else '未確定'}", flush=True)
        # 締切2分前の記録
        due = [r for r in races
               if not (OUT / f"snap_{r['rid']}.parquet").exists()
               and r["close_dt"] - SNAP_BEFORE <= now() < r["close_dt"] - dt.timedelta(seconds=30)]
        for r in sorted(due, key=lambda r: r["close_dt"]):
            df = grab(r, "snap", calib)
            n = 0 if df is None else len(value.select(df))
            print(f"{now():%m/%d %H:%M:%S} 記録 {r['rid']} 締切{r['close_dt']:%H:%M} 買い目{n}点", flush=True)
        time.sleep(15)
    print("区切りの時間に達したので終了", flush=True)


def report():
    snaps = sorted(OUT.glob("snap_*.parquet"))
    finals = {p.name[6:]: p for p in OUT.glob("final_*.parquet")}
    rows = []
    for p in snaps:
        f = finals.get(p.name[5:])
        if not f:
            continue
        s = pd.read_parquet(p).drop(columns=["hit", "payout"])
        fin = pd.read_parquet(f)[["a", "b", "hit", "payout", "ev", "odds_min"]].rename(
            columns={"ev": "ev_final", "odds_min": "odds_min_final"})
        rows.append(s.merge(fin, on=["a", "b"]))
    if not rows:
        print("結果のそろったレースがまだありません")
        return
    m = pd.concat(rows, ignore_index=True)
    m["day"] = m["at"].str[:10]
    print(f"記録 {len(snaps)}R / 結果あり {m['race_id'].nunique()}R")

    def line(name, s):
        n = len(s)
        roi = s["payout"].sum() / (100 * n) if n else float("nan")
        hit = s["hit"].mean() if n else float("nan")
        print(f"{name:<28} {n:>5}点 {s['race_id'].nunique():>4}R  的中{int(s['hit'].sum()):>4}  "
              f"的中率 {hit:6.1%}  回収率 {roi:6.1%}")

    print("\n--- 締切2分前の判定 ---")
    for min_ev, max_odds, name in [(1.0, 10.0, "標準（期待値1.0以上・10倍以下）"),
                                   (1.1, 10.0, "期待値1.1以上・10倍以下"),
                                   (1.0, 5.0, "期待値1.0以上・5倍以下"),
                                   (1.0, 1e9, "期待値1.0以上・上限なし")]:
        line(name, value.select(m, min_ev, max_odds))
    std = value.select(m)
    print("\n--- 日別（標準設定）---")
    for day, s in std.groupby("day"):
        line(day, s)
    if len(std):
        print(f"\n標準の買い目の期待値: 2分前 平均 {std['ev'].mean():.3f} → 確定 平均 {std['ev_final'].mean():.3f}")
        print(f"下限オッズ: 2分前 平均 {std['odds_min'].mean():.2f} → 確定 平均 {std['odds_min_final'].mean():.2f}")
    fx = m[(m["ev_final"] >= 1.0) & (m["odds_min_final"] <= 10.0)]
    line("\n参考: 同じレースを確定オッズで判定", fx)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 110)
    else:
        report()
