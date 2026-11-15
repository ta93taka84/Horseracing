"""指定したレースのワイド買い目を出す。

使い方: python -m keirin.predict <レース詳細ページのURL> [--min-odds 1.5] [--min-prob 0.4] [--top 5]
URL例: https://keirin.kdreams.jp/matsudo/racedetail/3120261109020007/
"""
import argparse
import re

import lightgbm as lgb

from .features import FEATURES, pairs
from .parse import parse
from .scrape import get
from .train import MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--min-odds", type=float, default=1.5, help="買い目に選ぶ最低オッズ")
    ap.add_argument("--min-prob", type=float, default=0.4, help="買い目に選ぶ最低の的中確率")
    ap.add_argument("--top", type=int, default=5, help="表示する組み合わせの数")
    a = ap.parse_args()

    html = get(a.url)
    d = parse(html, re.search(r"(\d{16})", a.url).group(1)) if html else None
    if not d:
        raise SystemExit("出走表を取得できませんでした。URLを確認してください。")
    df = pairs(d)
    df["p"] = lgb.Booster(model_file=str(MODEL)).predict(df[FEATURES])
    df = df.sort_values("p", ascending=False)

    r = d["race"]
    print(f"{r['date']} 場コード{r['venue']} {r['race_no']}R {r['grade']} {r['race_class']} {r['n_riders']}車立て")
    print("組み合わせ  的中確率  オッズ（現時点）")
    for _, x in df.head(a.top).iterrows():
        o = f"{x['odds_min']:.1f}〜{x['odds_max']:.1f}" if x["odds_min"] == x["odds_min"] else "未発表"
        print(f"  {int(x['a'])}={int(x['b'])}     {x['p']:.1%}    {o}")

    c = df[(df["odds_min"].fillna(0) >= a.min_odds) & (df["p"] >= a.min_prob)]
    if df["odds_min"].isna().all():
        print("\nオッズが未発表のため、買い目の判定はできません。確率の順位だけ参考にしてください。")
    elif c.empty:
        print(f"\n買い目: 見送り（オッズ{a.min_odds}倍以上・確率{a.min_prob:.0%}以上の組み合わせなし）")
    else:
        x = c.iloc[0]
        print(f"\n買い目: {int(x['a'])}={int(x['b'])}（的中確率 {x['p']:.1%}、オッズ {x['odds_min']:.1f}〜{x['odds_max']:.1f}）")


if __name__ == "__main__":
    main()
