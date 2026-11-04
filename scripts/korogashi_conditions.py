"""複勝転がし用：案3モデルの推奨馬（各レースで補正後3着内確率が最大の馬）について、
レース前に分かる条件別に 複勝率・的中時平均払戻・払戻1.2倍以上の割合 を集計する。
補正係数は前半/後半の2分割で互いに推定（推定に使っていない期間で評価）"""
import os, sys, json, itertools
import numpy as np, pandas as pd, lightgbm as lgb, statsmodels.api as sm
from scipy.special import logit, expit
sys.path.insert(0, os.path.dirname(__file__))
from features import build
from fit_model import adj_factors, BASE_FEATS

OUT = os.path.join(os.path.dirname(__file__), "..", "analysis")
df = build()
df = df[df.finished & df.odds.notna()].copy()
df["log_odds"] = np.log(df.odds)
df["month"] = df.date.dt.to_period("M")
df["exp3"] = np.nan
for m in df.month.unique():  # 市場ベースライン：月単位 out-of-fold
    tr, te = df.month != m, df.month == m
    mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200, verbose=-1)
    mdl.fit(df.loc[tr, BASE_FEATS], df.loc[tr, "top3"])
    df.loc[te, "exp3"] = mdl.predict_proba(df.loc[te, BASE_FEATS])[:, 1]
an = df[(df.date >= "2025-11-01") & (df.date < "2026-10-26")].copy()
an["half"] = np.where(an.date < "2026-05-01", "H1", "H2")
F = sm.add_constant(adj_factors(an))
off = logit(an.exp3.clip(1e-4, 1 - 1e-4))
an["p_adj"] = np.nan
for fit_h, ev_h in (("H1", "H2"), ("H2", "H1")):
    a, b = an.half == fit_h, an.half == ev_h
    g = sm.GLM(an.top3[a], F[a], family=sm.families.Binomial(), offset=off[a]).fit()
    an.loc[b, "p_adj"] = expit(off[b] + F[b] @ g.params)

pick = an.loc[an.groupby("race_id").p_adj.idxmax()].copy()
pick["2nd_gap"] = pick.p_adj - an.groupby("race_id").p_adj.apply(lambda x: x.nlargest(2).iloc[-1]).reindex(pick.race_id).values
pick["確率帯"] = pd.cut(pick.p_adj, [0, .6, .7, .75, .8, .85, 1], labels=["<60%", "60-70", "70-75", "75-80", "80-85", "85%+"])
pick["単勝帯"] = pd.cut(pick.odds, [1, 1.3, 1.5, 1.7, 2.0, 2.5, 99], labels=["~1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0", "2.0-2.5", "2.5+"])
pick["頭数帯"] = pd.cut(pick.n, [0, 10, 13, 16, 18], labels=["~10", "11-13", "14-16", "17-18"])
pick["芝ダ"] = pick.surface
pick["距離帯"] = pick.dist_band if "dist_band" in pick else pd.cut(pick.dist, [0, 1400, 1800, 2200, 4000], labels=["~1400", "1500-1800", "1900-2200", "2300+"])
pick["競馬場"] = pick.place_name
pick["クラス"] = pick.klass.map({-1: "新馬", 0: "未勝利", 1: "1勝", 2: "2勝", 3: "3勝", 4: "OP/L", 5: "重賞", 6: "重賞", 7: "重賞"})
pick["馬場"] = pick.going
pick["人気"] = np.where(pick["pop"] == 1, "1番人気", "2番人気以下")
pick.to_pickle("/tmp/korogashi_pick.pkl")


def summ(d):
    hit = d[d.top3 == 1]
    return pd.Series(dict(件数=len(d), 複勝率=d.top3.mean(), 的中時平均払戻=hit.fukusho_pay.mean() / 100,
                          払戻120以上の割合=(hit.fukusho_pay >= 120).mean(), 回収率=d.fukusho_pay.mean() / 100,
                          複勝率_前半=d[d.half == "H1"].top3.mean(), 複勝率_後半=d[d.half == "H2"].top3.mean()))


rows = []
conds = ["確率帯", "単勝帯", "頭数帯", "芝ダ", "距離帯", "競馬場", "クラス", "馬場", "人気"]
for c in conds:
    for lv, d in pick.groupby(c, observed=True):
        rows.append(dict(条件=c, 水準=str(lv), **summ(d)))
for c1, c2 in itertools.combinations(conds, 2):
    for (l1, l2), d in pick.groupby([c1, c2], observed=True):
        if len(d) >= 40:
            rows.append(dict(条件=f"{c1}×{c2}", 水準=f"{l1}×{l2}", **summ(d)))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "korogashi_conditions.csv"), index=False)
print("推奨馬の総計", summ(pick).round(3).to_dict())
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(t[t.条件.isin(["確率帯", "単勝帯", "人気"])].round(3).to_string(index=False))
cand = t[(t.複勝率 >= 0.8) & (t.件数 >= 40)].sort_values("回収率", ascending=False)
print("\n複勝率80%以上(件数40以上):", len(cand))
print(cand.head(40).round(3).to_string(index=False))
