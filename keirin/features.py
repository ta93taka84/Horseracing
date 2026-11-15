"""解析結果から「2車の組み合わせ1つ = 1行」の表を作る。オッズは特徴量に入れない。"""
from itertools import combinations

import pandas as pd

STYLE = {"逃": 0, "両": 1, "追": 2}
CLASS = {"SS": 6, "S1": 5, "S2": 4, "A1": 3, "A2": 2, "A3": 1, "L1": 0}
GRADE = {"GP": 6, "G1": 5, "G2": 4, "G3": 3, "F1": 2, "F2": 1}

RIDER = ["score", "score_z", "score_rank", "tip", "evaluation", "style", "rank_class", "age", "term",
         "gear", "b_cnt", "s_cnt", "nige", "makuri", "sashi", "mark", "win_rate", "top2_rate",
         "top3_rate", "recent_mean", "recent_top3", "recent_n", "line_size", "line_pos",
         "line_score", "line_head_score"]
RACE = ["n_riders", "grade", "bank", "girls", "score_std", "n_lines", "race_no"]
PAIR = ["same_line", "line_adjacent", "same_pref", "score_sum_z", "tip_sum", "tip_min"]
FEATURES = [f"x_{c}" for c in RIDER] + [f"y_{c}" for c in RIDER] + PAIR + RACE


def pairs(d):
    """parse() の戻り値 → 組み合わせの DataFrame。x が競走得点の高い方、y が低い方。"""
    r = pd.DataFrame(d["riders"])
    for c in ["recent_mean", "recent_top3", "recent_n", "finish"]:
        if c not in r:
            r[c] = float("nan")
    r["style"] = r["style"].map(STYLE)
    r["rank_class"] = r["rank_class"].map(CLASS)
    r["score_z"] = r["score"] - r["score"].mean()
    r["score_rank"] = r["score"].rank(ascending=False, method="min")
    r["line_score"] = r.groupby("line_id")["score"].transform("mean") - r["score"].mean()
    head = r[r["line_pos"] == 1].set_index("line_id")["score"]
    r["line_head_score"] = r["line_id"].map(head) - r["score"].mean()

    race = d["race"]
    base = {
        "race_id": race["race_id"], "date": race["date"],
        "n_riders": race["n_riders"], "grade": GRADE.get(race["grade"]), "bank": race["bank"],
        "girls": int(race["race_class"].startswith("L")), "score_std": r["score"].std(),
        "n_lines": r["line_id"].nunique(), "race_no": race["race_no"],
    }
    odds = {(o["a"], o["b"]): o for o in d["odds"]}
    pay = {(p["a"], p["b"]): p["payout"] for p in d["payouts"]}
    rows = []
    for i, j in combinations(range(len(r)), 2):
        x, y = r.iloc[i], r.iloc[j]
        if (y["score"], -y["num"]) > (x["score"], -x["num"]):
            x, y = y, x
        key = tuple(sorted((int(x["num"]), int(y["num"]))))
        same = x["line_id"] == y["line_id"]
        row = dict(base, a=key[0], b=key[1])
        row.update({f"x_{c}": x[c] for c in RIDER})
        row.update({f"y_{c}": y[c] for c in RIDER})
        row.update(
            same_line=int(same),
            line_adjacent=int(same and abs(x["line_pos"] - y["line_pos"]) == 1),
            same_pref=int(x["pref"] == y["pref"]),
            score_sum_z=x["score_z"] + y["score_z"],
            tip_sum=x["tip"] + y["tip"], tip_min=min(x["tip"], y["tip"]),
            odds_min=odds.get(key, {}).get("odds_min"), odds_max=odds.get(key, {}).get("odds_max"),
            payout=pay.get(key, 0.0),
            hit=int(x["finish"] <= 3 and y["finish"] <= 3),
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df[FEATURES] = df[FEATURES].astype(float)
    return df
