"""results.csv / races.csv から特徴量テーブル data/features.parquet を作る（各馬の過去情報は当該レースより前のデータのみ使用）"""
import os, re
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def klass(row):
    c, n = row["cond"], row["name"]
    if "G1" in n or "GI)" in n: return 7
    if "G2" in n or "GII)" in n: return 6
    if "G3" in n or "GIII)" in n: return 5
    if "オープン" in c or "(L)" in n or "OP" in n: return 4
    if "3勝" in c: return 3
    if "2勝" in c: return 2
    if "1勝" in c: return 1
    if "新馬" in c: return -1
    return 0  # 未勝利


def tsec(t):
    m = re.match(r"(\d+):(\d+\.\d)", str(t))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else np.nan


def build():
    ra = pd.read_csv(os.path.join(DATA, "races.csv"), dtype={"race_id": str, "place": str})
    rs = pd.read_csv(os.path.join(DATA, "results.csv"), dtype={"race_id": str, "horse_id": str, "jockey_id": str, "trainer_id": str, "owner_id": str})
    ra = ra[ra.surface != "障"].copy()
    ra["klass"] = ra.apply(klass, axis=1)
    ra["place_name"] = ra.place.map(PLACE)
    ra["handicap"] = ra.cond.str.contains("ハンデ").astype(int)
    ra["age_cond"] = np.where(ra.cond.str.contains("2歳"), "2歳", np.where(ra.cond.str.contains("3歳以上|4歳以上"), "古馬混", "3歳"))
    ra["female_only"] = ra.cond.str.contains("牝").astype(int)
    df = rs.merge(ra, on="race_id")
    df["rank_n"] = pd.to_numeric(df["rank"], errors="coerce")
    df["finished"] = df.rank_n.notna()
    df["top3"] = (df.rank_n <= 3).astype(int)
    df["win"] = (df.rank_n == 1).astype(int)
    for c in ["waku", "umaban", "kinryo", "odds", "pop", "weight", "wdiff", "last3f", "fukusho_pay"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sex"] = df.sexage.str[0]
    df["age"] = pd.to_numeric(df.sexage.str[1:], errors="coerce")
    df["date"] = pd.to_datetime(df.date)
    df["tsec"] = df.time.map(tsec)
    # レース内の相対値
    g = df.groupby("race_id")
    df["n"] = g.umaban.transform("count")
    df["uma_rel"] = (df.umaban - 1) / (df.n - 1).clip(lower=1)
    df["l3f_rank"] = g.last3f.rank(method="min")
    df["tdiff"] = df.tsec - g.tsec.transform("min")
    first = df.passing.astype(str).str.split("-").str[0]
    df["pos1"] = pd.to_numeric(first, errors="coerce")
    last = df.passing.astype(str).str.split("-").str[-1]
    df["pos4"] = pd.to_numeric(last, errors="coerce")
    df["pos1_rel"] = (df.pos1 - 1) / (df.n - 1).clip(lower=1)
    df["pos4_rel"] = (df.pos4 - 1) / (df.n - 1).clip(lower=1)
    df["kinryo_rel"] = df.kinryo - g.kinryo.transform("mean")
    df["weight_rel"] = df.weight - g.weight.transform("mean")
    df["odds_rank_in_race"] = df["pop"]
    # 市場の暗黙確率（単勝オッズの逆数をレース内で正規化）
    df["mkt_p"] = (1 / df.odds) / g.odds.transform(lambda x: (1 / x).sum())

    # ---- 馬の前走情報（時系列で shift）----
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    h = df.groupby("horse_id")
    prev_cols = {"rank_n": "p_rank", "pop": "p_pop", "odds": "p_odds", "l3f_rank": "p_l3f_rank", "tdiff": "p_tdiff",
                 "pos1_rel": "p_pos1_rel", "pos4_rel": "p_pos4_rel", "dist": "p_dist", "surface": "p_surface",
                 "klass": "p_klass", "jockey_id": "p_jockey", "place": "p_place", "date": "p_date", "kinryo": "p_kinryo",
                 "n": "p_n", "going": "p_going", "top3": "p_top3", "last3f": "p_last3f", "weight": "p_weight"}
    for a, b in prev_cols.items():
        df[b] = h[a].shift(1)
    df["p2_rank"] = h.rank_n.shift(2)
    df["p2_tdiff"] = h.tdiff.shift(2)
    df["n_prev"] = h.cumcount()
    df["days"] = (df.date - df.p_date).dt.days
    df["dist_chg"] = df.dist - df.p_dist
    df["surf_chg"] = np.where(df.p_surface.isna(), np.nan, (df.surface != df.p_surface).astype(float))
    df["klass_chg"] = df.klass - df.p_klass
    df["jockey_chg"] = np.where(df.p_jockey.isna(), np.nan, (df.jockey_id != df.p_jockey).astype(float))
    df["same_place"] = np.where(df.p_place.isna(), np.nan, (df.place == df.p_place).astype(float))
    df["kinryo_chg"] = df.kinryo - df.p_kinryo
    df["p_pop_minus_rank"] = df.p_pop - df.p_rank  # 前走で人気以上に走ったか
    # 過去（ウィンドウ内）の通算
    df["cum_top3_rate"] = (h.top3.cumsum() - df.top3) / df.n_prev.replace(0, np.nan)
    df["same_cond_starts"] = df.groupby(["horse_id", "surface", "dist"]).cumcount()
    df["same_cond_top3"] = df.groupby(["horse_id", "surface", "dist"]).top3.cumsum() - df.top3

    # ---- 騎手・調教師の過去成績（当日より前のみ、市場比 A/E）----
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["exp3"] = np.nan  # 後で市場ベースラインを入れる
    return df


if __name__ == "__main__":
    df = build()
    df.to_pickle("/tmp/features.pkl")
    print(df.shape)
