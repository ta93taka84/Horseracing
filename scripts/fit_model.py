"""予想用モデルを全期間で学習して models/ に保存
- market_base.txt : オッズ→3着内確率（LightGBM）
- adjust.json     : 市場確率に掛ける補正係数（ロジスティック回帰、offset=市場確率）"""
import os, json
import numpy as np, pandas as pd, lightgbm as lgb, statsmodels.api as sm
from scipy.special import logit

ROOT = os.path.join(os.path.dirname(__file__), "..")
BASE_FEATS = ["log_odds", "mkt_p", "pop", "n"]


def adj_factors(d):
    """補正要因（予想時にも同じ定義で使う）"""
    f = pd.DataFrame(index=d.index)
    f["2走前6-8着"] = d.p2_rank.between(6, 8)
    f["前走4角中団"] = d.p_pos4_rel.between(0.134, 0.533)
    f["芝ダ替わり"] = d.surf_chg == 1
    f["斤量軽(平均-1kg以下)"] = d.kinryo_rel <= -1
    f["昇級初戦"] = d.klass_chg > 0
    f["休み明け78日以上"] = d.days > 77
    return f.astype(float)


if __name__ == "__main__":
    df = pd.read_pickle("/tmp/features.pkl")
    df = df[df.finished & df.odds.notna()].copy()
    df["log_odds"] = np.log(df.odds)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200, verbose=-1)
    m.fit(df[BASE_FEATS], df.top3)
    m.booster_.save_model(os.path.join(ROOT, "models", "market_base.txt"))

    an = pd.read_pickle("/tmp/an.pkl")  # exp3 は月単位 out-of-fold
    X = sm.add_constant(adj_factors(an))
    g = sm.GLM(an.top3, X, family=sm.families.Binomial(), offset=logit(an.exp3.clip(1e-4, 1 - 1e-4))).fit()
    out = {k: round(float(v), 4) for k, v in g.params.items()}
    json.dump(dict(coef=out, pvalues={k: round(float(v), 4) for k, v in g.pvalues.items()},
                   period="2025-11-01..2026-10-25", n=int(len(an))),
              open(os.path.join(ROOT, "models", "adjust.json"), "w"), ensure_ascii=False, indent=1)
    print(g.summary2().tables[1][["Coef.", "P>|z|"]].round(3))
