"""One coin's hourly bars with every input the V7 coevolution search may use.

Engine features (build_features, same as BTC), order flow from the klines'
taker volume, Binance futures positioning (open interest, long/short ratios)
and the perp funding paid per bar. Works for BTC (the original files) and for
the confirmation coins downloaded by download_coin.py.

    from coin_data import load_coin, fold_windows
"""

import glob
import os
import json

import numpy as np
import pandas as pd

from build_features import COLUMNS, FEATURE_COLUMNS, add_features

FLOW = ["ofi_1h", "ofi_4h", "ofi_24h", "ofi_z"]
POS = ["oi_d1", "oi_d24", "ls_top", "ls_all", "taker_ls"]
SEALED_FROM = pd.Timestamp("2026-05-01 07:00")    # BTC bars from here on are spent / never dev


def _paths(sym):
    # BTC: the original hand-assembled files if present, else the download_coin.py layout
    if sym == "BTCUSDT" and os.path.isdir("../data/external/metrics/csv"):
        return "../data/raw", "../data/external/metrics/csv", None
    root = f"../data/coins/{sym}"
    return f"{root}/raw", f"{root}/metrics", f"{root}/funding"


def _raw(raw_dir):
    raw = pd.concat([pd.read_csv(f, header=None, names=COLUMNS) for f in sorted(glob.glob(f"{raw_dir}/*.csv"))],
                    ignore_index=True)
    raw = raw[pd.to_numeric(raw.open_time, errors="coerce").notna()]      # drop header rows
    ts = raw.open_time.astype("int64")
    ts = ts.where(ts.astype(str).str.len() != 16, ts // 1000)            # us -> ms from Jan 2025
    raw["open_time"] = pd.to_datetime(ts, unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base_volume"]:
        raw[c] = pd.to_numeric(raw[c])
    return raw.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def _funding(sym, fund_dir):
    if fund_dir is None:
        f = pd.read_parquet("../data/funding/BTCUSDT_funding.parquet")
        t, r = f.time, f.funding_rate
    else:
        f = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{fund_dir}/*.csv"))])
        t, r = pd.to_datetime(f.calc_time, unit="ms"), f.last_funding_rate
    s = pd.Series(r.values, index=pd.DatetimeIndex(t).astype("datetime64[ns]").round("h"))
    return s.groupby(level=0).last()


def load_coin(sym="BTCUSDT"):
    raw_dir, met_dir, fund_dir = _paths(sym)
    raw = _raw(raw_dir)
    df = add_features(raw[["open_time", "open", "high", "low", "close", "volume"]])
    t = df.open_time.astype("datetime64[ns]")
    r = raw.assign(open_time=raw.open_time.dt.tz_localize(None).astype("datetime64[ns]")).set_index("open_time")
    r = r.reindex(t.values)
    tb, v = r.taker_buy_base_volume.values, r.volume.values
    ofi = pd.Series(2 * tb / np.where(v > 0, v, np.nan) - 1).fillna(0.0)
    df["ofi_1h"] = ofi.values
    for n in (4, 24):
        df[f"ofi_{n}h"] = (2 * pd.Series(tb).rolling(n).sum() / pd.Series(v).rolling(n).sum() - 1).values
    df["ofi_z"] = ((ofi - ofi.rolling(168).mean()) / ofi.rolling(168).std()).values

    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{met_dir}/*.csv"))])
    m["t"] = pd.to_datetime(m.create_time).astype("datetime64[ns]")
    m = m.drop_duplicates("t").sort_values("t")
    close_t = pd.DataFrame({"t": (t + pd.Timedelta(hours=1)).values})
    p = pd.merge_asof(close_t, m, on="t", direction="backward", tolerance=pd.Timedelta(minutes=30))
    oi = np.log(p.sum_open_interest.where(p.sum_open_interest > 0))
    df["oi_d1"], df["oi_d24"] = oi.diff(1).values, oi.diff(24).values
    df["ls_top"] = np.log(p.sum_toptrader_long_short_ratio.where(lambda x: x > 0)).values
    df["ls_all"] = np.log(p.count_long_short_ratio.where(lambda x: x > 0)).values
    df["taker_ls"] = np.log(p.sum_taker_long_short_vol_ratio.where(lambda x: x > 0)).values

    # funding paid while holding close[i] -> close[i+1] (settles at open_time[i] + 2h)
    rate = _funding(sym, fund_dir)
    df["funding_paid"] = rate.reindex((t + pd.Timedelta(hours=2)).values).fillna(0.0).values
    df["open_time"] = t
    return df.replace([np.inf, -np.inf], np.nan)


def fold_windows(df, folds_path="../runs/walkforward/folds.json", train_span=40000, min_train=8000):
    """The six BTC walk-forward folds as CALENDAR windows mapped onto df.

    Test windows cover the same dates as the BTC dev folds; train = the
    train_span bars before (fewer if the coin's history is shorter).
    """
    btc_t = pd.read_parquet("../data/features/btcusdt_features_ext2.parquet", columns=["open_time"]).open_time
    btc_t = btc_t.astype("datetime64[ns]")
    t = df.open_time.values
    out = []
    for f in json.load(open(folds_path))["folds"]:
        t0, t1 = btc_t.iloc[f["test_start"]], btc_t.iloc[f["test_end"] - 1]
        ts, te = int(np.searchsorted(t, t0.to_datetime64())), int(np.searchsorted(t, t1.to_datetime64(), side="right"))
        tr = max(0, ts - train_span)
        assert ts - tr >= min_train, f"fold {f['fold']}: only {ts - tr} train bars"
        assert t1 < SEALED_FROM
        out.append({"fold": f["fold"], "train_start": tr, "train_end": ts, "test_start": ts, "test_end": te})
    return out


if __name__ == "__main__":
    b = load_coin("BTCUSDT")
    ext = pd.read_parquet("../data/features/btcusdt_features_ext2.parquet")
    assert len(b) == len(ext) and np.allclose(b[FEATURE_COLUMNS].values, ext[FEATURE_COLUMNS].values)
    folds = json.load(open("../runs/walkforward/folds.json"))["folds"]
    w = fold_windows(b)
    assert [(x["test_start"], x["test_end"]) for x in w] == [(f["test_start"], f["test_end"]) for f in folds], w
    print("BTC reproduces ext2 features and fold indices;", len(b), "rows")
    print(b[FLOW + POS + ["funding_paid"]].iloc[w[0]["test_start"]:].describe().T[["count", "mean", "std"]])
