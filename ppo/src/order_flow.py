"""Order-flow / positioning test (pre-registered in the experiment ledger, "ORDER-FLOW TEST").

Do taker buy/sell imbalance (klines) or Binance futures positioning (open interest,
long/short ratios) predict BTC direction at 1h / 4h well enough for a long/flat
timer to beat buy-and-hold after costs? Same machinery as signal_hunt.py.

Examples from ppo/src:
    python order_flow.py
    python order_flow.py --confirm flow@4h     # once, fold 6 (2026-05 -> 2026-08)

The --confirm examples reproduce the original one-time protocol. The
May–September 2026 evaluation windows are already spent historical data.
The CLI does not enforce one-time use; see experiment notes section 9i.
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd

import signal_hunt
from build_features import COLUMNS, FEATURE_COLUMNS
from signal_probe import COST
from project_paths import project_path

FLOW = ["ofi_1h", "ofi_4h", "ofi_24h", "ofi_z"]
POS = ["oi_d1", "oi_d24", "ls_top", "ls_all", "taker_ls"]
SETS = {"flow": FLOW, "pos": POS, "flowpos": FLOW + POS, "all": list(FEATURE_COLUMNS) + FLOW + POS}
RULES = ["ofi_1h", "ofi_4h"]


def taker_flow():
    raw = pd.concat([pd.read_csv(f, header=None, names=COLUMNS, usecols=["open_time", "volume", "taker_buy_base_volume"])
                     for f in sorted(glob.glob(project_path("data/raw/*.csv")))], ignore_index=True)
    ts = raw.open_time.astype("int64")
    ts = ts.where(ts.astype(str).str.len() != 16, ts // 1000)      # us -> ms from Jan 2025, as build_features
    raw["open_time"] = pd.to_datetime(ts, unit="ms")
    return raw.drop_duplicates("open_time").set_index("open_time")


def positioning():
    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(project_path("data/external/metrics/csv/*.csv")))])
    m["t"] = pd.to_datetime(m.create_time).astype("datetime64[ns]")
    return m.drop_duplicates("t").sort_values("t")


def load(path=project_path("data/features/btcusdt_features_ext.parquet")):
    df = pd.read_parquet(path)
    t = df.open_time.astype("datetime64[ns]")
    raw = taker_flow().reindex(t.values)
    assert raw.volume.notna().all(), "raw klines missing bars"
    tb, v = raw.taker_buy_base_volume.values, raw.volume.values
    ofi = pd.Series(2 * tb / np.where(v > 0, v, np.nan) - 1).fillna(0.0)
    df["ofi_1h"] = ofi.values
    for n in (4, 24):
        df[f"ofi_{n}h"] = (2 * pd.Series(tb).rolling(n).sum() / pd.Series(v).rolling(n).sum() - 1).values
    df["ofi_z"] = ((ofi - ofi.rolling(168).mean()) / ofi.rolling(168).std()).values

    m = positioning()
    close_t = pd.DataFrame({"t": (t + pd.Timedelta(hours=1)).values})
    p = pd.merge_asof(close_t, m, on="t", direction="backward", tolerance=pd.Timedelta(minutes=30))
    oi = np.log(p.sum_open_interest.where(p.sum_open_interest > 0))   # a few snapshots report OI = 0: missing
    df["oi_d1"], df["oi_d24"] = oi.diff(1).values, oi.diff(24).values
    df["ls_top"] = np.log(p.sum_toptrader_long_short_ratio).values
    df["ls_all"] = np.log(p.count_long_short_ratio).values
    df["taker_ls"] = np.log(p.sum_taker_long_short_vol_ratio).values
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", nargs="*")
    args = ap.parse_args()
    signal_hunt.HORIZONS = {1: 1, 4: 4}

    df = load()
    folds = json.load(open(project_path("runs/walkforward/folds.json")))["folds"]
    sets, rules = SETS, RULES
    if args.confirm:
        n = len(df)
        folds = [{"fold": 6, "train_start": n - 2945 - 40000, "train_end": n - 2945, "test_start": n - 2945,
                  "test_end": n}]
        names = {c.split("@")[0] for c in args.confirm}
        sets, rules = {k: v for k, v in SETS.items() if k in names}, [r for r in RULES if r in names]
    te = np.concatenate([np.arange(f["test_start"], f["test_end"]) for f in folds])
    print("test-span NaN share (pos):", df[POS].iloc[te].isna().mean().round(4).to_dict())

    res, seg = signal_hunt.run(df, folds, sets, rules, COST)
    res2, _ = signal_hunt.run(df, folds, sets, rules, 2 * COST)
    res0, _ = signal_hunt.run(df, folds, sets, rules, 0.0)
    if args.confirm:
        res = {k: v for k, v in res.items() if k in args.confirm}
    for k, v in res.items():
        v["sharpe_2x"], v["buy_hold_2x"] = res2[k]["sharpe"], res2[k]["buy_hold"]
        v["sharpe_0x"], v["buy_hold_0x"] = res0[k]["sharpe"], res0[k]["buy_hold"]
        v["pass"] = bool(v["pass"] and v["sharpe_2x"] > v["buy_hold_2x"])
        a = v["auc"]
        auc_s = f"AUC {a['auc_pooled']:.3f} folds {a['per_fold']}" if a else "rule timer"
        print(f"  {k:12s} {auc_s:58s} Sharpe {v['sharpe']:+.3f} vs B&H {v['buy_hold']:+.3f}  "
              f"2x {v['sharpe_2x']:+.3f}  0x {v['sharpe_0x']:+.3f} (B&H {v['buy_hold_0x']:+.3f})  "
              f"P {v['p_timer_le_bh']:.3f}  wins {v['wins_vs_bh']}/{len(seg)}  expo {v['exposure']:.2f}  "
              f"timing {v['timing']:+.3f}  {'PASS' if v['pass'] else '-'}")
    out = project_path("runs/walkforward/order_flow") + ("_confirm" if args.confirm else "") + ".json"
    json.dump({"n_segments": len(seg), "results": res}, open(out, "w"), indent=2)
    print("written", out)


if __name__ == "__main__":
    main()
