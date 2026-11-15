"""割安なワイドを探す。

3連単は発売票数がワイドよりはるかに多く、オッズが実力を正確に映しやすい。
3連単オッズから逆算した「2車とも3着以内」の確率を補正し、ワイドのオッズと掛けて期待値を出す。
"""
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

CALIB = Path(__file__).resolve().parent.parent / "models" / "market_calib.json"


def fit(p_tan, hit):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_tan, hit)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def add_ev(df, calib):
    """p_market（補正後の確率）と ev（確率 × ワイドの下限オッズ）の列を足す。"""
    df = df.copy()
    df["p_market"] = np.interp(df["p_tan"].astype(float), calib["x"], calib["y"])
    df["ev"] = df["p_market"] * df["odds_min"]
    return df


def select(df, min_ev=1.0, max_odds=10.0):
    return df[(df["ev"] >= min_ev) & (df["odds_min"] <= max_odds)]


def save(calib):
    CALIB.parent.mkdir(exist_ok=True)
    CALIB.write_text(json.dumps(calib))


def load():
    return json.loads(CALIB.read_text())
