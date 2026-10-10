"""Step 1 signal hunt (pre-registered in docs/loop/LEDGER.md, "STEP 1").

Does new information (cross-asset, on-chain, weekly momentum) predict BTC
direction well enough for a long/flat timer to beat buy-and-hold after costs?
Same machinery as signal_probe.py (logistic C=0.1, scaler fit per fold, purged
labels, 0.15% per unit position change), on the 6 walk-forward folds.

    python signal_hunt.py                       # dev: 12 trials on folds 0-5
    python signal_hunt.py --confirm all@168h    # once, fold 6 (2026-05 -> 2026-08)
"""

import argparse
import json

import numpy as np
import pandas as pd

from build_features import FEATURE_COLUMNS
from evaluate import paired_bootstrap
from metrics import sharpe_ratio
from signal_probe import SEG, auc_table, fit_predict, run_position

EXT = "../data/external"
CROSS = ["spx_r1", "spx_r5", "spx_r20", "vix", "vix_d5", "dxy_r5", "dxy_r20", "tnx_d5", "tnx_d20"]
ONCHAIN = ["mvrv", "mvrv_d7", "adr_ratio", "tx_ratio", "netflow_7d", "splyex_d30", "hash_ratio"]
MOM = ["mom_1w", "mom_4w"]
SETS = {"cross": CROSS, "onchain": ONCHAIN, "momentum": MOM,
        "all": list(FEATURE_COLUMNS) + CROSS + ONCHAIN + MOM}
HORIZONS = {24: 24, 168: 168}          # horizon -> rebalance period (bars)


def yahoo(name):
    r = json.load(open(f"{EXT}/yahoo_{name}.json"))["chart"]["result"][0]
    tz = r["meta"]["exchangeTimezoneName"]
    day = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    s = pd.Series(r["indicators"]["quote"][0]["close"], index=day, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def daily_external():
    """Daily features, each indexed by the UTC time it becomes known."""
    spx, vix, dxy, tnx = (yahoo(n) for n in ("spx", "vix", "dxy", "us10y"))
    lspx, ldxy = np.log(spx), np.log(dxy)
    cross = pd.DataFrame({"spx_r1": lspx.diff(), "spx_r5": lspx.diff(5), "spx_r20": lspx.diff(20)})
    cross = cross.join(pd.DataFrame({"vix": vix, "vix_d5": vix.diff(5)}), how="outer")
    cross = cross.join(pd.DataFrame({"dxy_r5": ldxy.diff(5), "dxy_r20": ldxy.diff(20)}), how="outer")
    cross = cross.join(pd.DataFrame({"tnx_d5": tnx.diff(5), "tnx_d20": tnx.diff(20)}), how="outer")
    cross.index = cross.index + pd.Timedelta(days=1)          # US close <= 21:00 UTC

    cm = pd.DataFrame(json.load(open(f"{EXT}/coinmetrics_btc.json"))["data"])
    cm.index = pd.to_datetime(cm["time"]).dt.tz_localize(None)
    m = cm[["CapMVRVCur", "AdrActCnt", "FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "TxCnt", "HashRate"]].astype(float)
    onchain = pd.DataFrame({
        "mvrv": m.CapMVRVCur,
        "mvrv_d7": np.log(m.CapMVRVCur).diff(7),
        "adr_ratio": m.AdrActCnt.rolling(7).mean() / m.AdrActCnt.rolling(28).mean(),
        "tx_ratio": m.TxCnt.rolling(7).mean() / m.TxCnt.rolling(28).mean(),
        "netflow_7d": (m.FlowInExNtv - m.FlowOutExNtv).rolling(7).sum() / m.SplyExNtv,
        "splyex_d30": np.log(m.SplyExNtv).diff(30),
        "hash_ratio": m.HashRate.rolling(30).mean() / m.HashRate.rolling(60).mean(),
    })
    onchain.index = onchain.index + pd.Timedelta(days=1, hours=6)   # flash lands ~02:10 next day
    return cross.sort_index(), onchain.sort_index()


def load(features_path):
    df = pd.read_parquet(features_path)
    lc = np.log(df["close"])
    df["mom_1w"], df["mom_4w"] = lc.diff(168), lc.diff(672)
    known_at = pd.DataFrame({"t": (df["open_time"] + pd.Timedelta(hours=1)).astype("datetime64[ns]")})
    for ext in daily_external():
        ext = ext.rename_axis("t").reset_index()
        ext["t"] = ext["t"].astype("datetime64[ns]")
        m = pd.merge_asof(known_at, ext, on="t", direction="backward")
        for c in ext.columns.drop("t"):
            df[c] = m[c].values
    return df


def hold(signal, k):
    """Decide at segment bars 0, k, 2k, ... and hold."""
    return signal[(np.arange(len(signal)) // k) * k]


def timers(df, folds, sets, rules):
    """Per-segment Sharpe of B&H and every timer (logistic sets x horizons, rule timers x rebalances)."""
    rows = []
    for fold in folds:
        te = slice(fold["test_start"], fold["test_end"])
        close = df["close"].values[te]
        sig = {f"{n}@{H}h": (fit_predict(df, fold, f, H)[0] > 0.5, HORIZONS[H]) for n, f in sets.items()
               for H in HORIZONS}
        sig.update({f"{r}@{k}h": (df[r].values[te] > 0, k) for r in rules for k in HORIZONS.values()})
        for s in range(0, len(close) - SEG + 1, SEG):
            c, sl = close[s:s + SEG], slice(s, s + SEG)
            row = {"fold": fold["fold"], "segment_start": s, "bh_move": float(np.log(c[-1] / c[0])),
                   "buy_hold": sharpe_ratio(run_position(c, np.ones(SEG)))}
            for name, (x, k) in sig.items():
                w = hold(x[sl].astype(float), k)
                row[name] = sharpe_ratio(run_position(c, w))
                row[f"{name}_exposure"] = float(w.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def run(df, folds, sets, rules):
    auc = auc_table(df, folds, sets, list(HORIZONS)) if sets else {}
    seg = timers(df, folds, sets, rules)
    res = {}
    for name in [c for c in seg.columns if "@" in c and not c.endswith("_exposure")]:
        a = auc.get(name)
        auc_ok = a is None or (a["auc_pooled"] >= 0.55 and sum(x > 0.52 for x in a["per_fold"]) >= 5)
        p = paired_bootstrap(seg[name].values, seg["buy_hold"].values)
        up, down = seg.bh_move > 0, seg.bh_move < 0
        res[name] = {"auc": a, "sharpe": float(seg[name].mean()), "buy_hold": float(seg["buy_hold"].mean()),
                     "p_timer_le_bh": p, "wins_vs_bh": int((seg[name] > seg["buy_hold"]).sum()),
                     "exposure": float(seg[f"{name}_exposure"].mean()),
                     "timing": float(seg.loc[up, f"{name}_exposure"].mean() - seg.loc[down, f"{name}_exposure"].mean()),
                     "pass": bool(auc_ok and seg[name].mean() > seg["buy_hold"].mean() and p <= 0.10)}
    return res, seg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="../data/features/btcusdt_features_ext.parquet")
    ap.add_argument("--folds", default="../runs/walkforward/folds.json")
    ap.add_argument("--confirm", nargs="*", help="configs to score ONCE on fold 6 (e.g. all@168h mom_4w@168h)")
    ap.add_argument("--out", default="../runs/walkforward/signal_hunt.json")
    args = ap.parse_args()

    df = load(args.features)
    folds = json.load(open(args.folds))["folds"]
    ref = pd.read_parquet("../data/features/btcusdt_features.parquet", columns=["open_time"])
    assert (df["open_time"].values[:len(ref)] == ref["open_time"].values).all(), "fold indices misaligned"

    if args.confirm:
        n = len(df)
        folds = [{"fold": 6, "train_start": n - 2945 - 40000, "train_end": n - 2945, "test_start": n - 2945,
                  "test_end": n}]
        names = {c.split("@")[0] for c in args.confirm}
        res, seg = run(df, folds, {k: v for k, v in SETS.items() if k in names},
                       [r for r in ("mom_1w", "mom_4w") if r in names])
        res = {k: v for k, v in res.items() if k in args.confirm}
        args.out = args.out.replace(".json", "_confirm.json")
    else:
        res, seg = run(df, folds, SETS, ["mom_1w", "mom_4w"])

    base = auc_table(df, folds, {"features": list(FEATURE_COLUMNS)}, list(HORIZONS))
    print(f"{len(seg)} segments | replication, price features only: "
          + "  ".join(f"{k} AUC {v['auc_pooled']:.3f}" for k, v in base.items()))
    for k, v in res.items():
        a = v["auc"]
        auc_s = f"AUC {a['auc_pooled']:.3f} folds {a['per_fold']}" if a else "rule timer"
        print(f"  {k:16s} {auc_s:60s} timer {v['sharpe']:+.3f} vs B&H {v['buy_hold']:+.3f}  "
              f"P {v['p_timer_le_bh']:.3f}  wins {v['wins_vs_bh']}/{len(seg)}  expo {v['exposure']:.2f}  "
              f"timing {v['timing']:+.3f}  {'PASS' if v['pass'] else '-'}")
    json.dump({"n_segments": len(seg), "replication": base, "results": res}, open(args.out, "w"), indent=2)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
