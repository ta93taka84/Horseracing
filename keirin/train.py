"""収集済みページから学習データを作り、モデルを学習・検証して保存する。

使い方: python -m keirin.train
"""
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .features import FEATURES, pairs
from .parse import parse_file

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "wide.txt"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)


def build():
    frames = []
    for f in sorted((ROOT / "data" / "raw").glob("*.html.gz")):
        d = parse_file(f)
        if d and d["race"]["finished"] and d["payouts"] and d["race"]["date"]:
            frames.append(pairs(d))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(ROOT / "data" / "pairs.parquet")
    return df


def pick(df, min_odds=1.0, min_prob=0.0):
    """各レースで「オッズ下限以上」の中から的中確率が最も高い1点を選ぶ。"""
    c = df[(df["odds_min"].fillna(0) >= min_odds) & (df["p"] >= min_prob)]
    return c.loc[c.groupby("race_id")["p"].idxmax()]


def report(name, sel, n_races):
    print(f"{name:<34} 購入{len(sel):>5}R ({len(sel) / n_races:>4.0%})  的中率 {sel['hit'].mean():.1%}"
          f"  回収率 {sel['payout'].sum() / (100 * len(sel)):.1%}  平均配当 {sel.loc[sel['hit'] == 1, 'payout'].mean():.0f}円")


def main():
    df = build()
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
