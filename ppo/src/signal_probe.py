"""Is there a timing signal in the real BTC features? (Task A, the null result.)

Reproduces the ad-hoc checks behind docs/experiment-notes.md §9e on the six
walk-forward folds (runs/walkforward/folds.json), with no lookahead:

1. Direction AUC: logistic regression (C=0.1, StandardScaler fit on the fold's
   train span) predicting sign(forward H-bar log return) from FEATURE_COLUMNS
   (+ optional 30/90-day trend, + optional Binance funding rate). Training
   labels that would peek into the test span are purged (last H train bars).
2. Ceilings on the 48 test segments (8 x 720 bars per fold), daily rebalance,
   COST per unit of position change, Sharpe per segment then averaged:
   logistic long/flat and long/short timers, volatility targeting
   (train-median 7d vol / trailing 7d vol, capped at 1), 30-day trend filter
   (close > 720-bar SMA), buy-and-hold. Two cost conventions are reported:
   `sharpe` charges the entry from cash, `sharpe_no_entry` does not (the
   original ad-hoc timer check used the latter: B&H +0.97 vs +0.93).
3. Timing score of every trained arm from runs/walkforward/compare.csv
   (exposure on up minus down segments, pooled corr(exposure, segment move)).

    python signal_probe.py                         # all three, prints + JSON
    python signal_probe.py --funding ../data/funding/BTCUSDT_funding.parquet
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from build_features import FEATURE_COLUMNS
from compare_arms import timing_score
from metrics import sharpe_ratio
from project_paths import project_path

SEG, DAY, WEEK = 720, 24, 168
COST = 0.0015   # fee 0.1% + slippage 0.05% per unit of position change


def load(features_path, funding_path=None):
    df = pd.read_parquet(features_path)
    lc = np.log(df["close"])
    df["trend_30d"] = lc - lc.shift(720)
    df["trend_90d"] = lc - lc.shift(2160)
    if funding_path:
        f = pd.read_parquet(funding_path).sort_values("time")
        close_time = (df["open_time"] + pd.Timedelta(hours=1)).astype("datetime64[ns]")
        f["time"] = f["time"].astype("datetime64[ns]")
        m = pd.merge_asof(pd.DataFrame({"t": close_time}), f.rename(columns={"time": "t"}),
                          on="t", direction="backward")
        df["funding_rate"] = m["funding_rate"].values
    return df


def fit_predict(df, fold, feats, H):
    """Fit on the fold's train span (labels purged before test), predict test."""
    fwd = np.log(df["close"].shift(-H) / df["close"])
    tr = df.iloc[fold["train_start"]:fold["train_end"] - H]
    ytr = fwd.iloc[fold["train_start"]:fold["train_end"] - H]
    ok = tr[feats].notna().all(axis=1) & ytr.notna()
    sc = StandardScaler().fit(tr.loc[ok, feats])
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(tr.loc[ok, feats]),
                                                      (ytr[ok] > 0).astype(int))
    te = df.iloc[fold["test_start"]:fold["test_end"]]
    p = clf.predict_proba(sc.transform(te[feats].fillna(0)))[:, 1]
    return p, fwd.iloc[fold["test_start"]:fold["test_end"]].values


def auc_table(df, folds, feature_sets, horizons):
    out = {}
    for name, feats in feature_sets.items():
        for H in horizons:
            ps, ys = [], []
            for fold in folds:
                p, y = fit_predict(df, fold, feats, H)
                ok = ~np.isnan(y)
                ps.append(p[ok]); ys.append(y[ok] > 0)
            per_fold = [roc_auc_score(y, p) for p, y in zip(ps, ys)]
            out[f"{name}@{H}h"] = {"auc_pooled": float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))),
                                    "auc_fold_mean": float(np.mean(per_fold)),
                                    "per_fold": [round(a, 4) for a in per_fold]}
    return out


def run_position(close, target, cost=COST, charge_entry=True):
    """Hourly returns of a position path held over bar i -> i+1, paying `cost`
    per unit change (the first position taken from cash only if charge_entry)."""
    r = close[1:] / close[:-1] - 1
    pos = target[:-1]
    change = np.abs(np.diff(np.concatenate([[0.0 if charge_entry else pos[0]], pos])))
    return pos * r - change * cost


def daily(signal):
    """Hold each daily decision for 24 bars (rebalance at bar 0, 24, 48...)."""
    idx = (np.arange(len(signal)) // DAY) * DAY
    return signal[idx]


def ceilings(df, folds, H=24):
    feats = list(FEATURE_COLUMNS)
    lr = np.log(df["close"]).diff()
    vol7 = lr.rolling(WEEK).std()
    rows = []
    for fold in folds:
        p, _ = fit_predict(df, fold, feats, H)
        train_med_vol = vol7.iloc[fold["train_start"]:fold["train_end"]].median()
        te = slice(fold["test_start"], fold["test_end"])
        close = df["close"].values[te]
        v = vol7.values[te]
        trend = (df["close"] / df["close"].rolling(720).mean()).values[te]
        for s in range(0, len(close) - SEG + 1, SEG):
            c = close[s:s + SEG]
            sl = slice(s, s + SEG)
            strat = {
                "buy_hold": np.ones(SEG),
                "timer_long_flat": daily((p[sl] > 0.5).astype(float)),
                "timer_long_short": daily(np.where(p[sl] > 0.5, 1.0, -1.0)),
                "vol_target": daily(np.minimum(1.0, train_med_vol / v[sl])),
                "trend_30d": daily((trend[sl] > 1).astype(float)),
            }
            rows.append({"fold": fold["fold"], "segment_start": s,
                         "bh_move": float(np.log(c[-1] / c[0])),
                         **{k: sharpe_ratio(run_position(c, w)) for k, w in strat.items()},
                         **{f"{k}_no_entry": sharpe_ratio(run_position(c, w, charge_entry=False))
                            for k, w in strat.items()},
                         **{f"{k}_exposure": float(w.mean()) for k, w in strat.items()}})
    seg = pd.DataFrame(rows)
    names = ["buy_hold", "timer_long_flat", "timer_long_short", "vol_target", "trend_30d"]
    summary = {k: {"sharpe": float(seg[k].mean()),
                   "sharpe_no_entry": float(seg[f"{k}_no_entry"].mean()),
                   "wins_vs_bh": int((seg[k] > seg["buy_hold"]).sum()) if k != "buy_hold" else None,
                   "timing": float(seg.loc[seg.bh_move > 0, f"{k}_exposure"].mean()
                                   - seg.loc[seg.bh_move < 0, f"{k}_exposure"].mean())}
               for k in names}
    return summary, seg


def arm_timing(compare_csv, arms=("coevolution", "single", "population"), kind="best"):
    c = pd.read_csv(compare_csv)
    return {a: timing_score(c, a, kind) for a in arms if (c.arm == a).any()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=project_path("data/features/btcusdt_features.parquet"))
    ap.add_argument("--folds", default=project_path("runs/walkforward/folds.json"))
    ap.add_argument("--compare", default=project_path("runs/walkforward/compare.csv"))
    ap.add_argument("--funding", default=None,
                    help="parquet with columns time (UTC-naive) and funding_rate")
    ap.add_argument("--out", default=project_path("runs/walkforward/signal_probe.json"))
    args = ap.parse_args()

    df = load(args.features, args.funding)
    folds = json.load(open(args.folds))["folds"]
    sets = {"features": list(FEATURE_COLUMNS),
            "features+trend": list(FEATURE_COLUMNS) + ["trend_30d", "trend_90d"]}
    if args.funding:
        sets["features+funding"] = list(FEATURE_COLUMNS) + ["funding_rate"]
        sets["funding_only"] = ["funding_rate"]
    res = {"auc": auc_table(df, folds, sets, [24, 168])}
    res["ceilings"], seg = ceilings(df, folds)
    res["n_segments"] = len(seg)
    res["arm_timing"] = arm_timing(args.compare)

    print("direction AUC (pooled over folds | mean of per-fold)")
    for k, v in res["auc"].items():
        print(f"  {k:26s} {v['auc_pooled']:.3f} | {v['auc_fold_mean']:.3f}   {v['per_fold']}")
    print(f"\nceilings on {len(seg)} segments (daily rebalance, cost {COST:.2%} per unit change)")
    for k, v in res["ceilings"].items():
        print(f"  {k:18s} sharpe {v['sharpe']:+.3f} (no entry cost {v['sharpe_no_entry']:+.3f})  "
              f"wins vs B&H {v['wins_vs_bh']}  timing {v['timing']:+.3f}")
    print("\narm timing (best_trader, walk-forward)")
    for a, t in res["arm_timing"].items():
        print(f"  {a:12s} exposure up {t['exposure_up']:.2f} / down {t['exposure_down']:.2f}  "
              f"timing {t['timing']:+.3f}  corr {t['corr_exposure_move']:+.3f}")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
