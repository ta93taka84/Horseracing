"""案3：オッズ起点の3着内確率に補正を掛け、市場と差が大きい馬を「要詳細チェック」として出す
使い方: python3 scripts/predict.py RACE_ID [RACE_ID ...]
前提: data/ が前日までの結果で更新済み（python3 scripts/fetch.py YYYYMMDD → parse.py）"""
import os, re, sys, json, html, unicodedata
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit, expit

sys.path.insert(0, os.path.dirname(__file__))
from fetch import get
from features import build, klass
from fit_model import adj_factors, BASE_FEATS

ROOT = os.path.join(os.path.dirname(__file__), "..")
FLAG = 0.2  # 補正幅(logit)がこれ以上なら要詳細チェック
GRADE = {"1": 7, "2": 6, "3": 5, "15": 4, "16": 4, "5": 4}


def txt(x):
    return unicodedata.normalize("NFKC", html.unescape(re.sub(r"<[^>]+>", " ", x))).strip()


def entry(rid):
    s = get(f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}", enc="utf-8")
    d1 = txt(re.search(r"RaceData01.*?</div>", s, re.S).group(0))
    d2 = re.sub(r"\s+", " ", txt(re.search(r"RaceData02.*?</div>", s, re.S).group(0)))
    head = re.search(r'class="RaceName">(.*?)</h1>', s, re.S).group(1)
    name = txt(head)
    gm = re.search(r"Icon_GradeType(\d+)", head)
    m = re.search(r"(芝|ダ|障)(\d+)m", d1)
    race = dict(race_id=rid, name=name, surface=m.group(1), dist=int(m.group(2)), cond=d2)
    race["klass"] = GRADE.get(gm.group(1), 4) if gm else klass(dict(cond=d2, name=name))
    rows = []
    for tr in re.findall(r'<tr class="HorseList[^"]*"[^>]*>(.*?)</tr>', s, re.S):
        if "Cancel" in tr or "取消" in tr or "除外" in tr:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(tds) < 7 or not txt(tds[1]).isdigit():
            continue  # 出走馬以外の行
        hid = re.search(r"/horse/(\d+)", tr)
        rows.append(dict(waku=int(txt(tds[0])), umaban=int(txt(tds[1])), horse_id=hid.group(1),
                         horse=txt(re.search(r'class="HorseName">(.*?)</span>', tr, re.S).group(1)),
                         kinryo=float(txt(tds[5])), jockey=txt(tds[6])))
    return race, pd.DataFrame(rows)


def odds(rid):
    j = json.loads(get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=1&action=update", enc="utf-8"))
    o = j["data"]["odds"]["1"]
    return {int(k): (float(v[0]), int(v[2])) for k, v in o.items() if re.match(r"^\d+(\.\d+)?$", v[0])}, j["data"].get("official_datetime", "")


def history():
    df = build().sort_values(["horse_id", "date", "race_id"])
    last = df.groupby("horse_id").tail(1)  # 最新の出走 = 今回から見た前走
    return last.set_index("horse_id")[["date", "rank_n", "p_rank", "surface", "klass", "pos4_rel"]]


def predict(rid, hist, base, coef):
    race, e = entry(rid)
    od, ts = odds(rid)
    e = e[e.umaban.isin(od)].copy()
    e["odds"] = e.umaban.map(lambda u: od[u][0]); e["pop"] = e.umaban.map(lambda u: od[u][1])
    e["n"] = len(e); e["log_odds"] = np.log(e.odds)
    e["mkt_p"] = (1 / e.odds) / (1 / e.odds).sum()
    e["p_mkt"] = base.predict(e[BASE_FEATS])
    h = hist.reindex(e.horse_id)
    today = pd.Timestamp.today().normalize()
    e["p2_rank"] = h.p_rank.values
    e["p_pos4_rel"] = h.pos4_rel.values
    e["surf_chg"] = np.where(h.surface.isna(), np.nan, (h.surface.values != race["surface"]).astype(float))
    e["kinryo_rel"] = e.kinryo - e.kinryo.mean()
    e["klass_chg"] = race["klass"] - h.klass.values
    e["days"] = (today - pd.to_datetime(h.date.values)).days
    f = adj_factors(e)
    adj = coef["const"] + sum(f[k] * v for k, v in coef.items() if k != "const")
    e["p_adj"] = expit(logit(e.p_mkt.clip(1e-4, 1 - 1e-4)) + adj)
    e["補正"] = adj
    e["要因"] = f.apply(lambda r: "・".join(f"{k}({'+' if coef[k] > 0 else '-'})" for k in f.columns if r[k] == 1), axis=1)
    e = e.sort_values("p_adj", ascending=False).reset_index(drop=True)
    mkt_top3 = set(e.nlargest(3, "p_mkt").umaban); adj_top3 = set(e.head(3).umaban)
    e["チェック"] = np.where((e.補正.abs() >= FLAG) | (e.umaban.isin(mkt_top3 ^ adj_top3)), "★", "")
    print(f"\n■ {rid} {race['name']} {race['surface']}{race['dist']}m  オッズ時刻 {ts}")
    out = e[["チェック", "umaban", "horse", "jockey", "odds", "pop", "p_mkt", "p_adj", "要因"]].copy()
    out[["p_mkt", "p_adj"]] = (out[["p_mkt", "p_adj"]] * 100).round(1)
    out.columns = ["★", "馬番", "馬名", "騎手", "単勝", "人気", "市場3着内%", "補正後%", "補正要因"]
    print(out.to_string(index=False))
    return e


if __name__ == "__main__":
    base = lgb.Booster(model_file=os.path.join(ROOT, "models", "market_base.txt"))
    coef = json.load(open(os.path.join(ROOT, "models", "adjust.json")))["coef"]
    hist = history()
    for rid in sys.argv[1:]:
        predict(rid, hist, base, coef)
