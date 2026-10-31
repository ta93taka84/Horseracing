"""市場ベースライン（オッズから見た3着内確率）に対する上乗せ要因を総当たりで探す"""
import os, sys, itertools, json
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

OUT = os.path.join(os.path.dirname(__file__), "..", "analysis")
os.makedirs(OUT, exist_ok=True)
START, MID, END = os.environ.get("START", "2025-11-01"), os.environ.get("MID", "2026-05-01"), os.environ.get("END", "2026-10-26")

df = pd.read_pickle("/tmp/features.pkl")
df = df[df.finished & df.odds.notna()].copy()

# ---------- 1. 市場ベースライン：月単位の out-of-fold で P(3着内 | オッズ, 頭数) ----------
df["log_odds"] = np.log(df.odds)
df["month"] = df.date.dt.to_period("M")
base_feats = ["log_odds", "mkt_p", "pop", "n"]
df["exp3"] = np.nan
for m in df.month.unique():
    tr, te = df.month != m, df.month == m
    mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200, verbose=-1)
    mdl.fit(df.loc[tr, base_feats], df.loc[tr, "top3"])
    df.loc[te, "exp3"] = mdl.predict_proba(df.loc[te, base_feats])[:, 1]

# ---------- 騎手・調教師・種牡馬などの「過去のみ」A/E（前日までの累積）----------
df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
df["resid"] = df.top3 - df.exp3
for key in ["jockey_id", "trainer_id", "owner_id"]:
    daily = df.groupby([key, "date"]).agg(o=("top3", "sum"), e=("exp3", "sum"), c=("top3", "size")).reset_index()
    daily = daily.sort_values([key, "date"])
    for c in "oec":
        daily["cum_" + c] = daily.groupby(key)[c].cumsum() - daily[c]
    daily[key[:-3] + "_ae"] = (daily.cum_o + 20 * 1) / (daily.cum_e + 20)  # 縮小推定（事前A/E=1, 20頭分）
    daily[key[:-3] + "_starts"] = daily.cum_c
    df = df.merge(daily[[key, "date", key[:-3] + "_ae", key[:-3] + "_starts"]], on=[key, "date"], how="left")
# 騎手×コース(場・芝ダ)
key = ["jockey_id", "place", "surface"]
daily = df.groupby(key + ["date"]).agg(o=("top3", "sum"), e=("exp3", "sum")).reset_index().sort_values(key + ["date"])
for c in "oe":
    daily["cum_" + c] = daily.groupby(key)[c].cumsum() - daily[c]
daily["jockey_course_ae"] = (daily.cum_o + 20) / (daily.cum_e + 20)
df = df.merge(daily[key + ["date", "jockey_course_ae"]], on=key + ["date"], how="left")

# 種牡馬・母父（あれば）：全体A/E と 芝ダ×距離帯別A/E（前日までの累積）
PED = os.path.join(os.path.dirname(__file__), "..", "data", "pedigree.csv")
df["dist_band"] = pd.cut(df.dist, [0, 1400, 1800, 2200, 4000], labels=["短", "マイル", "中", "長"]).astype(str)
if os.path.exists(PED) and not os.environ.get("NOPED"):
    pe = pd.read_csv(PED, dtype=str).drop_duplicates("horse_id")
    df = df.merge(pe[["horse_id", "sire", "damsire", "sire_sire"]], on="horse_id", how="left")
    for name, key in [("sire_ae", ["sire"]), ("damsire_ae", ["damsire"]), ("sire_cond_ae", ["sire", "surface", "dist_band"]),
                      ("sire_going_ae", ["sire", "surface", "going"]), ("damsire_cond_ae", ["damsire", "surface", "dist_band"])]:
        daily = df.groupby(key + ["date"]).agg(o=("top3", "sum"), e=("exp3", "sum")).reset_index().sort_values(key + ["date"])
        for c in "oe":
            daily["cum_" + c] = daily.groupby(key)[c].cumsum() - daily[c]
        daily[name] = (daily.cum_o + 20) / (daily.cum_e + 20)
        df = df.merge(daily[key + ["date", name]], on=key + ["date"], how="left")
    PED_FEATS = ["sire_ae", "damsire_ae", "sire_cond_ae", "sire_going_ae", "damsire_cond_ae"]
else:
    PED_FEATS = []

an = df[(df.date >= START) & (df.date < END)].copy()
an["half"] = np.where(an.date < MID, "H1", "H2")
print("analysis rows", len(an), "races", an.race_id.nunique())

# ---------- 2. 単変量スクリーニング ----------
num_feats = ["waku", "umaban", "uma_rel", "n", "age", "kinryo", "kinryo_rel", "kinryo_chg", "weight", "weight_rel", "wdiff",
             "p_rank", "p2_rank", "p_pop", "p_pop_minus_rank", "p_l3f_rank", "p_tdiff", "p2_tdiff", "p_pos1_rel", "p_pos4_rel",
             "days", "dist_chg", "klass_chg", "n_prev", "cum_top3_rate", "same_cond_starts", "jockey_ae", "trainer_ae",
             "owner_ae", "jockey_course_ae", "jockey_starts", "trainer_starts", "dist", "rno", "klass"] + PED_FEATS
cat_feats = ["sex", "surface", "going", "place_name", "weather", "turn", "outer", "trainer_area", "surf_chg", "jockey_chg",
             "same_place", "handicap", "age_cond", "female_only", "p_going", "p_top3"]


def binned(s, q=5):
    if s.nunique() <= 8:
        return s.astype(str)
    try:
        return pd.qcut(s, q, duplicates="drop").astype(str)
    except Exception:
        return s.astype(str)


def ae_table(d, col):
    g = d.groupby(col, observed=True).agg(n=("top3", "size"), o=("top3", "sum"), e=("exp3", "sum"),
                                          v=("exp3", lambda p: (p * (1 - p)).sum()), pay=("fukusho_pay", "mean"))
    g["ae"] = g.o / g.e
    g["z"] = (g.o - g.e) / np.sqrt(g.v)
    g["roi"] = g.pay / 100
    return g


for c in num_feats:
    an["b_" + c] = binned(an[c]).where(an[c].notna(), "NA")
for c in cat_feats:
    an["b_" + c] = an[c].astype(str)
bcols = ["b_" + c for c in num_feats + cat_feats]

rows = []
for c in bcols:
    t = ae_table(an, c)
    t1 = ae_table(an[an.half == "H1"], c); t2 = ae_table(an[an.half == "H2"], c)
    for lvl, r in t.iterrows():
        if r.n < 300: continue
        rows.append(dict(feature=c[2:], level=lvl, n=int(r.n), top3_rate=r.o / r.n, exp_rate=r.e / r.n, ae=r.ae, z=r.z,
                         roi=r.roi, ae_H1=t1.ae.get(lvl, np.nan), ae_H2=t2.ae.get(lvl, np.nan),
                         z_H1=t1.z.get(lvl, np.nan), z_H2=t2.z.get(lvl, np.nan)))
uni = pd.DataFrame(rows)
uni["p"] = 2 * norm.sf(uni.z.abs())
uni["q"] = multipletests(uni.p, method="fdr_bh")[1]
uni["replicated"] = (np.sign(uni.z_H1) == np.sign(uni.z_H2)) & (uni.z_H1.abs() > 1.64) & (uni.z_H2.abs() > 1.64)
uni.sort_values("z", key=abs, ascending=False).to_csv(os.path.join(OUT, "univariate.csv"), index=False)

# ---------- 3. 2要因の組み合わせ（交互作用）：単独効果を差し引いた上乗せ ----------
rows = []
for a, b in itertools.combinations(bcols, 2):
    g = an.groupby([a, b], observed=True).agg(n=("top3", "size"), o=("top3", "sum"), e=("exp3", "sum"),
                                              v=("exp3", lambda p: (p * (1 - p)).sum()), pay=("fukusho_pay", "mean"))
    g = g[g.n >= 400]
    if g.empty: continue
    ta = ae_table(an, a).ae; tb = ae_table(an, b).ae
    for (la, lb), r in g.iterrows():
        exp_add = r.e * ta[la] * tb[lb]  # 単独効果の積を期待値とする
        z = (r.o - exp_add) / np.sqrt(r.v)
        d1 = an[(an[a] == la) & (an[b] == lb)]
        rows.append(dict(f1=a[2:], l1=la, f2=b[2:], l2=lb, n=int(r.n), top3_rate=r.o / r.n, ae=r.o / r.e,
                         ae_vs_marginals=r.o / exp_add, z_inter=z, roi=r.pay / 100,
                         ae_H1=d1[d1.half == "H1"].top3.sum() / d1[d1.half == "H1"].exp3.sum(),
                         ae_H2=d1[d1.half == "H2"].top3.sum() / d1[d1.half == "H2"].exp3.sum()))
inter = pd.DataFrame(rows)
inter["p"] = 2 * norm.sf(inter.z_inter.abs())
inter["q"] = multipletests(inter.p, method="fdr_bh")[1]
inter.sort_values("z_inter", key=abs, ascending=False).to_csv(os.path.join(OUT, "interactions.csv"), index=False)

# ---------- 4. 機械学習：オッズのみ vs 全特徴量（後半期間で評価）----------
from sklearn.metrics import log_loss, roc_auc_score
feats = list(dict.fromkeys(base_feats + num_feats + cat_feats))
X = an[feats].copy()
for c in cat_feats:
    X[c] = X[c].astype("category")
tr, te = an.half == "H1", an.half == "H2"
res = {}
for name, fs in [("odds_only", base_feats), ("all", feats), ("no_odds", [f for f in feats if f not in base_feats])]:
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X.loc[tr, fs], an.loc[tr, "top3"])
    p = m.predict_proba(X.loc[te, fs])[:, 1]
    an.loc[te, "p_" + name] = p
    res[name] = dict(logloss=log_loss(an.loc[te, "top3"], p), auc=roc_auc_score(an.loc[te, "top3"], p))
    if name == "all":
        imp = pd.Series(m.booster_.feature_importance("gain"), index=fs).sort_values(ascending=False)
        imp.to_csv(os.path.join(OUT, "lgbm_gain.csv"))
# 市場確率(exp3)を初期値にして「上乗せ分」だけ学習するモデル（H1末尾2か月で早期打ち切り）
from scipy.special import logit, expit
fs = [f for f in feats if f not in base_feats]
dates_h1 = an.loc[tr, "date"]
cut = dates_h1.quantile(0.75)  # H1の最後の1/4で早期打ち切り
fit, val = tr & (an.date <= cut), tr & (an.date > cut)
init = logit(an.exp3.clip(1e-4, 1 - 1e-4))
m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.01, num_leaves=15, min_child_samples=300, reg_lambda=10,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
m.fit(X.loc[fit, fs], an.loc[fit, "top3"], init_score=init[fit], eval_set=[(X.loc[val, fs], an.loc[val, "top3"])],
      eval_init_score=[init[val]], callbacks=[lgb.early_stopping(100, verbose=False)])
p = expit(init[te] + m.predict(X.loc[te, fs], raw_score=True))
an.loc[te, "p_market_plus"] = p
p0 = an.loc[te, "exp3"]
res["market_plus"] = dict(logloss=log_loss(an.loc[te, "top3"], p), auc=roc_auc_score(an.loc[te, "top3"], p), trees=int(m.best_iteration_ or 0))
res["market_base"] = dict(logloss=log_loss(an.loc[te, "top3"], p0), auc=roc_auc_score(an.loc[te, "top3"], p0))
an.loc[te, "p_market_base"] = p0
pd.Series(m.booster_.feature_importance("gain"), index=fs).sort_values(ascending=False).to_csv(os.path.join(OUT, "market_plus_gain.csv"))
# 各レースで予測1位の馬の3着内率（=本命の的中率）
t = an[te]
for name in list(res):
    top = t.loc[t.groupby("race_id")["p_" + name].idxmax()]
    res[name]["pick1_top3"] = top.top3.mean(); res[name]["pick1_roi"] = top.fukusho_pay.mean() / 100
    top3 = t[t.groupby("race_id")["p_" + name].rank(ascending=False, method="first") <= 3]
    res[name]["pick3_top3"] = top3.top3.mean()
fav = t[t["pop"] == 1]
res["1番人気"] = dict(pick1_top3=fav.top3.mean(), pick1_roi=fav.fukusho_pay.mean() / 100)
json.dump(res, open(os.path.join(OUT, "model_compare.json"), "w"), indent=1, ensure_ascii=False, default=float)
an.to_pickle("/tmp/an.pkl")
print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
