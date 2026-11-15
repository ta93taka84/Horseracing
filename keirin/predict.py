"""指定したレースのワイド買い目を出す。

使い方: python -m keirin.predict <レース詳細ページのURL> [--min-ev 1.0] [--max-odds 10] [--top 8]
URL例: https://keirin.kdreams.jp/matsudo/racedetail/3120261109020007/

買い目は「3連単オッズから逆算した的中確率 × ワイドの下限オッズ」が基準以上の組み合わせ。
オッズは実行した時点のものなので、締切の直前に実行するほど検証時の条件に近づく。
"""
import argparse
import re

import lightgbm as lgb

from . import value
from .features import FEATURES, pairs
from .parse import parse
from .scrape import get
from .train import MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--min-ev", type=float, default=1.0, help="買い目に選ぶ最低の期待値（1.0 = 賭け金と同額の見込み）")
    ap.add_argument("--max-odds", type=float, default=10.0, help="買い目に選ぶ最高オッズ（下限オッズで判定）")
    ap.add_argument("--top", type=int, default=8, help="表示する組み合わせの数")
    a = ap.parse_args()

    html = get(a.url)
    d = parse(html, re.search(r"(\d{16})", a.url).group(1)) if html else None
    if not d:
        raise SystemExit("出走表を取得できませんでした。URLを確認してください。")
    df = pairs(d)
    df["p_model"] = lgb.Booster(model_file=str(MODEL)).predict(df[FEATURES])

    r = d["race"]
    print(f"{r['date']} 場コード{r['venue']} {r['race_no']}R {r['grade']} {r['race_class']} {r['n_riders']}車立て")
    if df["odds_min"].isna().all() or df["p_tan"].isna().all():
        print("オッズが未発表のため、買い目の判定はできません。出走表だけで出したモデルの確率を表示します。")
        for _, x in df.sort_values("p_model", ascending=False).head(a.top).iterrows():
            print(f"  {int(x['a'])}={int(x['b'])}  {x['p_model']:.1%}")
        return

    df = value.add_ev(df, value.load()).sort_values("ev", ascending=False)
    print("組み合わせ  期待値  的中確率  ワイドオッズ   （参考）出走表モデルの確率")
    for _, x in df.head(a.top).iterrows():
        print(f"  {int(x['a'])}={int(x['b'])}     {x['ev']:.2f}   {x['p_market']:>5.1%}    "
              f"{x['odds_min']:>5.1f}〜{x['odds_max']:<5.1f}  {x['p_model']:.1%}")

    buy = value.select(df, a.min_ev, a.max_odds)
    if buy.empty:
        print(f"\n買い目: 見送り（期待値{a.min_ev}以上・オッズ{a.max_odds:g}倍以下の組み合わせなし）")
    else:
        print("\n買い目: " + "、".join(f"{int(x['a'])}={int(x['b'])}" for _, x in buy.iterrows()))


if __name__ == "__main__":
    main()
