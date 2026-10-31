"""results.csv の全馬について血統ページから父・母父を取得し data/pedigree.csv に追記"""
import os, re, csv, time, random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from fetch import get, DATA

OUT = os.path.join(DATA, "pedigree.csv")


def nm(x):
    m = re.search(r"<a[^>]*>([^<]+)<", x)
    return re.sub(r"&nbsp;.*", "", m.group(1)).strip() if m else ""


def ped(hid):
    s = get(f"https://db.netkeiba.com/horse/ped/{hid}/")
    i = s.find("blood_table"); t = s[i:s.find("</table>", i)]
    g1 = [nm(x) for x in re.findall(r'<td[^>]*rowspan="16"[^>]*>(.*?)</td>', t, re.S)]
    g2 = [nm(x) for x in re.findall(r'<td[^>]*rowspan="8"[^>]*>(.*?)</td>', t, re.S)]
    time.sleep(0.5 + random.random() * 0.5)
    return dict(horse_id=hid, sire=g1[0] if g1 else "", dam=g1[1] if len(g1) > 1 else "",
                sire_sire=g2[0] if g2 else "", damsire=g2[2] if len(g2) > 2 else "")


def safe(hid):
    try:
        return ped(hid)
    except Exception:
        return dict(horse_id=hid, sire="", dam="", sire_sire="", damsire="")


if __name__ == "__main__":
    rs = pd.read_csv(os.path.join(DATA, "results.csv"), dtype={"horse_id": str})
    ids = sorted(set(rs.horse_id.dropna()))
    done = set(pd.read_csv(OUT, dtype=str).horse_id) if os.path.exists(OUT) else set()
    todo = [h for h in ids if h not in done]
    print(len(ids), "horses", len(todo), "todo", flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f, ThreadPoolExecutor(3) as ex:
        w = csv.DictWriter(f, fieldnames=["horse_id", "sire", "dam", "sire_sire", "damsire"])
        if new: w.writeheader()
        for k, r in enumerate(ex.map(safe, todo)):
            w.writerow(r)
            if k % 1000 == 0: f.flush(); print(k, flush=True)
    print("ped done", flush=True)
