"""市場確率に少数の補正要因を掛ける簡易モデル：H1で係数推定→H2で検証"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.special import logit, expit
from sklearn.metrics import log_loss
an = pd.read_pickle("/tmp/an.pkl")
d = pd.DataFrame(index=an.index)
d["p2_rank_6_8"] = an.p2_rank.between(6, 8)
d["surf_chg"] = an.surf_chg == 1
d["career_le1"] = an.n_prev <= 1
d["career_ge7"] = an.n_prev >= 7
d["light_kinryo"] = an.kinryo_rel <= -1
d["class_up"] = an.klass_chg > 0
d["long_rest"] = an.days > 77
d["prev_front4"] = an.p_pos4_rel <= 0.133
d["prev_mid4"] = an.p_pos4_rel.between(0.134, 0.533)
d["waku8"] = an.waku == 8
d["prev_beat_pop"] = an.p_pop_minus_rank > 3
d["jockey_hot"] = an.jockey_ae > 1.073
d = d.astype(float)
off = logit(an.exp3.clip(1e-4, 1 - 1e-4))
h1, h2 = an.half == "H1", an.half == "H2"
m = sm.GLM(an.top3[h1], sm.add_constant(d[h1]), family=sm.families.Binomial(), offset=off[h1]).fit()
print(m.summary2().tables[1][["Coef.", "P>|z|"]].round(3))
p = expit(off[h2] + sm.add_constant(d[h2]) @ m.params)
t = an[h2].assign(p_adj=p)
print("H2 logloss market", round(log_loss(t.top3, t.exp3), 5), "adjusted", round(log_loss(t.top3, t.p_adj), 5))
for c in ["exp3", "p_adj"]:
    top = t.loc[t.groupby("race_id")[c].idxmax()]
    print(c, "本命3着内率", round(top.top3.mean(), 4), "複勝回収", round(top.fukusho_pay.mean() / 100, 3))
# 補正で本命が入れ替わったレースのみ比較
a = t.loc[t.groupby("race_id").exp3.idxmax()].set_index("race_id")
b = t.loc[t.groupby("race_id").p_adj.idxmax()].set_index("race_id")
ch = a.index[a.umaban != b.umaban]
print("本命が変わったレース", len(ch), "市場本命", a.loc[ch].top3.mean().round(3), "補正後本命", b.loc[ch].top3.mean().round(3))
# 人気別の実績
an["popc"] = an["pop"].clip(upper=10)
print(an.groupby("popc").agg(n=("top3", "size"), top3=("top3", "mean"), roi=("fukusho_pay", lambda x: x.mean() / 100)).round(3).T)
