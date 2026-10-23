"""Download a Binance symbol's history into data/coins/<SYM>/.

spot 1h klines (monthly), USDⓈ-M perp 1h klines (monthly), funding (monthly),
futures metrics (daily, open interest + long/short ratios). Archives are read
in memory; only extracted CSVs are retained. Existing CSVs are skipped.

From the repo root:
    python ppo/src/download_coin.py BTCUSDT --end 2026-08
"""

import argparse
import http.client
import io
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from project_paths import project_path

BASE = "https://data.binance.vision/data"


def months(start, end):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def fetch(url, out_csv):
    if os.path.exists(out_csv):
        return "skip"
    for attempt in range(5):                  # flaky connections reset mid-run; retry, then give up
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            break
        except urllib.error.HTTPError as e:
            return f"{e.code}"
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            time.sleep(2 ** attempt)
    else:
        return "neterr"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = z.namelist()[0]
        open(out_csv, "wb").write(z.read(name))
    return "ok"


def jobs(sym, end, root):
    out = []
    for kind, path, start in [("raw", f"spot/monthly/klines/{sym}/1h/{sym}-1h-{{m}}", "2017-08"),
                              ("perp", f"futures/um/monthly/klines/{sym}/1h/{sym}-1h-{{m}}", "2019-09"),
                              ("funding", f"futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{{m}}", "2019-09")]:
        os.makedirs(f"{root}/{kind}", exist_ok=True)
        for m in months(start, end):
            p = path.format(m=m)
            out.append((f"{BASE}/{p}.zip", f"{root}/{kind}/{os.path.basename(p)}.csv"))
    os.makedirs(f"{root}/metrics", exist_ok=True)
    for d in pd.date_range("2021-12-01", pd.Period(end, "M").end_time.normalize(), freq="D"):
        name = f"{sym}-metrics-{d:%Y-%m-%d}"
        out.append((f"{BASE}/futures/um/daily/metrics/{sym}/{name}.zip", f"{root}/metrics/{name}.csv"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--end", default="2026-09")
    args = ap.parse_args()
    for sym in args.symbols:
        todo = jobs(sym, args.end, project_path(f"data/coins/{sym}"))
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda j: fetch(*j), todo))
        bad = [j[0] for j, r in zip(todo, res) if r not in ("ok", "skip")]
        print(f"{sym}: {res.count('ok')} downloaded, {res.count('skip')} cached, {len(bad)} missing")
        open(project_path(f"data/coins/{sym}/missing.txt"), "w").write("\n".join(bad))


if __name__ == "__main__":
    main()
