"""Volatility-targeting check (pre-registered in the experiment ledger, "VOL TARGET CHECK").

Long-only BTC sized as min(1, target / trailing sigma), rebalanced every 24 bars.
Target = median trailing sigma over the fold's train window. No fitting on test.

    python vol_target.py              # trials 36-37 on folds 0-5
    python vol_target.py --confirm vt_168h   # once, fold 6 (2026-05 -> 2026-08)
"""

import argparse
import json

import numpy as np
import pandas as pd

from evaluate import paired_bootstrap
from metrics import max_drawdown, sharpe_ratio
from signal_probe import COST, DAY, SEG, run_position
from project_paths import project_path

WINDOWS = {"vt_168h": 168, "vt_24h": 24}


def exposure(sigma, target, k=DAY):
    """min(1, target/sigma), decided at bars 0, k, 2k, ... of the segment and held."""
    w = np.minimum(1.0, target / sigma)
    return w[(np.arange(len(w)) // k) * k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", nargs="*")
    args = ap.parse_args()

    # _ext is btcusdt_features extended to 2026-08 (identical prefix), needed for fold 6
    close = pd.read_parquet(project_path("data/features/btcusdt_features_ext.parquet"), columns=["close"]).close
    lr = np.log(close).diff()
    # sigma at bar i uses returns up to close[i] only
    sigma = {name: lr.rolling(n).std().values for name, n in WINDOWS.items()}
    folds = json.load(open(project_path("runs/walkforward/folds.json")))["folds"]
    names = list(WINDOWS)
    if args.confirm:
        n = len(close)
        folds = [{"fold": 6, "train_start": n - 2945 - 40000, "train_end": n - 2945,
                  "test_start": n - 2945, "test_end": n}]
        names = args.confirm

    c = close.values
    rows = []
    for fold in folds:
        tr = slice(fold["train_start"], fold["train_end"])
        target = {nm: np.nanmedian(sigma[nm][tr]) for nm in names}
        for s in range(fold["test_start"], fold["test_end"] - SEG + 1, SEG):
            seg = slice(s, s + SEG)
            for mult in (1.0, 2.0):
                paths = {"buy_hold": np.ones(SEG)}
                paths.update({nm: exposure(sigma[nm][seg], target[nm]) for nm in names})
                for nm, w in paths.items():
                    r = run_position(c[seg], w, cost=COST * mult)
                    rows.append({"fold": fold["fold"], "seg": s, "cost": mult, "rule": nm,
                                 "sharpe": sharpe_ratio(r), "ret": float(np.prod(1 + r) - 1),
                                 "mdd": max_drawdown(np.cumprod(np.concatenate([[1.0], 1 + r]))),
                                 "expo": float(w.mean())})
    long = pd.DataFrame(rows)
    n_seg = long.seg.nunique()
    out = {}
    for mult in (1.0, 2.0):
        w = long[long.cost == mult].pivot(index=["fold", "seg"], columns="rule")
        bh = w["sharpe"]["buy_hold"].values
        for nm in ["buy_hold"] + names:
            x = w["sharpe"][nm].values
            out.setdefault(nm, {})[f"cost{mult:g}x"] = {
                "sharpe": float(x.mean()), "ret_mean": float(w["ret"][nm].mean()),
                "mdd_mean": float(w["mdd"][nm].mean()), "expo": float(w["expo"][nm].mean()),
                "wins_vs_bh": int((x > bh).sum()), "p_le_bh": paired_bootstrap(x, bh)}
    bh1, bh2 = out["buy_hold"]["cost1x"]["sharpe"], out["buy_hold"]["cost2x"]["sharpe"]
    for nm, v in out.items():
        a, b = v["cost1x"], v["cost2x"]
        v["pass"] = nm != "buy_hold" and a["sharpe"] > bh1 and a["p_le_bh"] <= 0.10 and b["sharpe"] > bh2
        print(f"{nm:10s} Sharpe {a['sharpe']:+.3f} (2x cost {b['sharpe']:+.3f})  ret {a['ret_mean']:+.2%}  "
              f"maxDD {a['mdd_mean']:.1%}  expo {a['expo']:.2f}  wins {a['wins_vs_bh']}/{n_seg}  "
              f"P {a['p_le_bh']:.3f}  {'PASS' if v['pass'] else '-'}")
    per_fold = long[long.cost == 1.0].groupby(["fold", "rule"]).sharpe.mean().unstack().round(2)
    print("\nper-fold mean Sharpe (cost 1x)\n", per_fold.to_string())
    path = project_path("runs/walkforward/vol_target") + ("_confirm" if args.confirm else "") + ".json"
    json.dump({"n_segments": n_seg, "results": out, "per_fold": per_fold.to_dict()}, open(path, "w"), indent=2)
    print("written", path)


if __name__ == "__main__":
    main()
