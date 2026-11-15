"""収集済みページから学習データを作り、モデルを学習・検証して保存する。

使い方: python -m keirin.train
"""
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .features import FEATURES, pairs
from . import value
from .parse import parse_file

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "wide.txt"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)


def _one(f):
    d = parse_file(f)
    if d and d["race"]["finished"] and d["payouts"] and d["race"]["date"]:
        return pairs(d)


def build():
    files = sorted((ROOT / "data" / "raw").glob("*.html.gz"))
    with Pool(4) as pool:
        frames = [x for x in pool.imap(_one, files, chunksize=20) if x is not None]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(ROOT / "data" / "pairs.parquet")
    return df


def value_backtest(df):
    """割安ワイド買いの検証。補正と基準は前半70%で決め、後半30%で確かめる。"""
    df = df.dropna(subset=["odds_min", "p_tan"])
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * 0.7)]
    a, b = df[df["date"] < cut], df[df["date"] >= cut]
    calib = value.fit(a["p_tan"], a["hit"])
    print(f"\n--- 割安ワイド買い（3連単オッズから逆算した確率 × ワイドオッズ）---")
    print(f"前半 {a['race_id'].nunique()}R（〜{cut}の前日） / 後半 {b['race_id'].nunique()}R（{cut}〜{dates[-1]}）")
    for max_odds in [5.0, 10.0, 1e9]:
        for min_ev in [0.9, 1.0, 1.1]:
            for name, d in [("前半", a), ("後半", b)]:
                sel = value.select(value.add_ev(d, calib), min_ev, max_odds)
                cap = "上限なし" if max_odds > 1e6 else f"{max_odds:.0f}倍以下"
                report(f"{name} 期待値{min_ev:.1f}以上 / オッズ{cap}", sel, d["race_id"].nunique())
    value.save(value.fit(df["p_tan"], df["hit"]))  # 全データで補正を作り直して保存
    print(f"保存: {value.CALIB}")


def pick(df, min_odds=1.0, min_prob=0.0):
    """各レースで「オッズ下限以上」の中から的中確率が最も高い1点を選ぶ。"""
    c = df[(df["odds_min"].fillna(0) >= min_odds) & (df["p"] >= min_prob)]
    return c.loc[c.groupby("race_id")["p"].idxmax()]


def report(name, sel, n_races):
    print(f"{name:<34} 購入{len(sel):>5}点 ({sel['race_id'].nunique() / n_races:>4.0%}のレース)  的中率 {sel['hit'].mean():.1%}"
          f"  回収率 {sel['payout'].sum() / (100 * len(sel)):.1%}  平均配当 {sel.loc[sel['hit'] == 1, 'payout'].mean():.0f}円")


def main():
    df = build()
    value_backtest(df)
    dates = sorted(df["date"].unique())
    d_va, d_te = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.8)]
    tr, va, te = df[df["date"] < d_va], df[(df["date"] >= d_va) & (df["date"] < d_te)], df[df["date"] >= d_te]
    print(f"レース数 学習{tr['race_id'].nunique()} 検証{va['race_id'].nunique()} テスト{te['race_id'].nunique()}"
          f"  (テスト期間 {d_te}〜{dates[-1]})")

    m = lgb.train(PARAMS, lgb.Dataset(tr[FEATURES], tr["hit"]), 3000,
                  valid_sets=[lgb.Dataset(va[FEATURES], va["hit"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    te = te.assign(p=m.predict(te[FEATURES]))
    print(f"木の本数 {m.best_iteration}  テスト AUC {roc_auc_score(te['hit'], te['p']):.3f}"
          f"  logloss {log_loss(te['hit'], te['p']):.4f}")

    n = te["race_id"].nunique()
    print("\n--- 1レース1点買いの比較（テスト期間）---")
    report("モデル: 確率最大の1点", pick(te), n)
    report("比較: 最低オッズの1点（1番人気）", te.loc[te.dropna(subset=["odds_min"]).groupby("race_id")["odds_min"].idxmin()], n)
    report("比較: 競走得点 上位2車", te.loc[te.groupby("race_id")["score_sum_z"].idxmax()], n)
    report("比較: 記者印 上位2車", te.loc[te.groupby("race_id")["tip_sum"].idxmin()], n)

    print("\n--- オッズ下限 × 確率下限（テスト期間）---")
    for mo in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for mp in [0.0, 0.4, 0.5, 0.6]:
            sel = pick(te, mo, mp)
            if len(sel) >= 30:
                report(f"オッズ{mo:.1f}倍以上 / 確率{mp:.0%}以上", sel, n)

    imp = pd.Series(m.feature_importance("gain"), FEATURES).sort_values(ascending=False)
    print("\n重要な特徴量 上位15:", ", ".join(imp.index[:15]))

    # 全データで学習し直して保存する
    final = lgb.train(PARAMS, lgb.Dataset(df[FEATURES], df["hit"]), int(m.best_iteration * 1.2))
    MODEL.parent.mkdir(exist_ok=True)
    final.save_model(str(MODEL))
    print(f"\n保存: {MODEL}")


if __name__ == "__main__":
    main()
