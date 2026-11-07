"""複勝転がし：提示レースが条件リストを満たすかチェックする
使い方: python3 scripts/korogashi_check.py RACE_ID [RACE_ID ...]"""
import os, sys, json, io, contextlib
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
import predict
from fetch import get

ROOT = os.path.join(os.path.dirname(__file__), "..")
P_MIN, FUKU_LOW_MIN = 0.65, 1.2  # 条件リスト v3


def fukusho(rid):
    j = json.loads(get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=1&action=update", enc="utf-8"))
    return {int(k): (float(v[0]), float(v[1])) for k, v in j["data"]["odds"].get("2", {}).items() if v[0].replace(".", "").isdigit()}


if __name__ == "__main__":
    base = lgb.Booster(model_file=os.path.join(ROOT, "models", "market_base.txt"))
    coef = json.load(open(os.path.join(ROOT, "models", "adjust.json")))["coef"]
    hist = predict.history()
    for rid in sys.argv[1:]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            e = predict.predict(rid, hist, base, coef)
            race, _ = predict.entry(rid)
        print(buf.getvalue())
        top = e.iloc[0]; fk = fukusho(rid).get(int(top.umaban), (float("nan"), float("nan")))
        checks = [("障害戦ではない", race["surface"] != "障", race["surface"]),
                  ("推奨馬の補正後3着内確率 65%以上", top.p_adj >= P_MIN, f"{top.p_adj:.1%}"),
                  ("推奨馬の複勝オッズ下限 1.2倍以上", fk[0] >= FUKU_LOW_MIN, f"{fk[0]}〜{fk[1]}倍")]
        print(f"推奨馬: {top.umaban}番 {top.horse}（{top['pop']}番人気）  補正要因: {top['要因'] or 'なし'}")
        for name, ok, val in checks:
            print(f"  {'○' if ok else '×'} {name}: {val}")
        print("  判定:", "条件をすべて満たす" if all(c[1] for c in checks) else "見送り")
