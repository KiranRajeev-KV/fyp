"""Funding carry check (pre-registered in docs/loop/LEDGER.md, "CARRY CHECK").

Can a rule that holds long spot + short USDT-M perp (earning funding) beat
buy-and-hold Sharpe on the 48 real-BTC walk-forward test segments? No fitting.

    python carry_check.py
"""

import glob
import json

import numpy as np
import pandas as pd

from evaluate import paired_bootstrap
from metrics import sharpe_ratio
from signal_probe import SEG

SPOT_COST, PERP_COST = 0.0015, 0.0010     # per unit change: fee + slippage
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def load(features_path="../data/features/btcusdt_features.parquet"):
    spot = pd.read_parquet(features_path, columns=["open_time", "close"])
    perp = pd.concat([pd.read_csv(f, header=None, names=COLS, dtype=str)
                      for f in sorted(glob.glob("../data/external/perp/BTCUSDT-1h-*.csv"))])
    perp = perp[perp.open_time != "open_time"]
    perp = pd.Series(perp.close.astype(float).values,
                     index=pd.to_datetime(perp.open_time.astype("int64"), unit="ms"))
    perp = perp[~perp.index.duplicated()]
    t = spot.open_time.astype("datetime64[ns]")
    df = pd.DataFrame({"t": t, "spot": spot.close.values,
                       "perp": perp.reindex(t.values).values})

    f = pd.read_parquet("../data/funding/BTCUSDT_funding.parquet")
    f["time"] = f["time"].astype("datetime64[ns]").dt.round("h")
    rate = f.groupby("time").funding_rate.last()
    close_t = df.t + pd.Timedelta(hours=1)               # bar i closes at open_time + 1h
    # paid while holding close[i] -> close[i+1], i.e. at time open_time[i] + 2h
    df["paid"] = rate.reindex(close_t + pd.Timedelta(hours=1)).fillna(0.0).values
    # last rate already paid when deciding at close[i]
    df["last_rate"] = rate.reindex(close_t, method="ffill").values
    df["decide"] = close_t.dt.hour.isin([0, 8, 16]).values
    return df


def returns(seg, s, p, cost_mult=1.0):
    """Hourly returns of spot weight s and perp weight p (p = -1 is a full short)."""
    rs = seg.spot.values[1:] / seg.spot.values[:-1] - 1
    rp = seg.perp.values[1:] / seg.perp.values[:-1] - 1
    s, p = s[:-1], p[:-1]
    ds = np.abs(np.diff(np.concatenate([[0.0], s])))
    dp = np.abs(np.diff(np.concatenate([[0.0], p])))
    return s * rs + p * rp - p * seg.paid.values[:-1] - cost_mult * (ds * SPOT_COST + dp * PERP_COST)


def hold_at_decisions(x, decide):
    """Value of x at the latest decision bar (bar 0 always decides)."""
    idx = np.where(decide | (np.arange(len(x)) == 0), np.arange(len(x)), 0)
    return x[np.maximum.accumulate(idx)]


def rules(seg):
    pos = hold_at_decisions((seg.last_rate.values > 0).astype(float), seg.decide.values)
    one, zero = np.ones(len(seg)), np.zeros(len(seg))
    return {"buy_hold": (one, zero),
            "carry_always": (one, -one),
            "carry_if_pos": (pos, -pos),
            "carry_if_pos_else_bh": (one, -pos)}


def main():
    df = load()
    folds = json.load(open("../runs/walkforward/folds.json"))["folds"]
    rows = []
    for fold in folds:
        for s in range(fold["test_start"], fold["test_end"] - SEG + 1, SEG):
            seg = df.iloc[s:s + SEG]
            assert seg.perp.notna().all(), f"missing perp bars in fold {fold['fold']} @ {s}"
            for mult in (1.0, 2.0):
                for name, (sw, pw) in rules(seg).items():
                    r = returns(seg, sw, pw, mult)
                    rows.append({"fold": fold["fold"], "seg": s, "cost": mult, "rule": name,
                                 "sharpe": sharpe_ratio(r), "ret": float(np.prod(1 + r) - 1),
                                 "carry_share": float((pw < 0).mean())})
    long = pd.DataFrame(rows)
    out = {}
    for mult in (1.0, 2.0):
        w = long[long.cost == mult].pivot(index=["fold", "seg"], columns="rule")
        bh = w["sharpe"]["buy_hold"].values
        for name in ["buy_hold", "carry_always", "carry_if_pos", "carry_if_pos_else_bh"]:
            x = w["sharpe"][name].values
            out.setdefault(name, {})[f"cost{mult:g}x"] = {
                "sharpe": float(x.mean()), "ret_mean": float(w["ret"][name].mean()),
                "carry_share": float(w["carry_share"][name].mean()),
                "wins_vs_bh": int((x > bh).sum()), "p_le_bh": paired_bootstrap(x, bh)}
    for name, v in out.items():
        a, b = v["cost1x"], v["cost2x"]
        v["pass"] = name != "buy_hold" and a["sharpe"] > out["buy_hold"]["cost1x"]["sharpe"] \
            and a["p_le_bh"] <= 0.10 and b["sharpe"] > out["buy_hold"]["cost2x"]["sharpe"]
        print(f"{name:22s} Sharpe {a['sharpe']:+.3f} (2x cost {b['sharpe']:+.3f})  "
              f"mean seg return {a['ret_mean']:+.2%}  carry {a['carry_share']:.0%}  "
              f"wins {a['wins_vs_bh']}/48  P {a['p_le_bh']:.3f}  {'PASS' if v.get('pass') else '-'}")
    per_fold = long[long.cost == 1.0].groupby(["fold", "rule"]).sharpe.mean().unstack().round(2)
    print("\nper-fold mean Sharpe (cost 1x)\n", per_fold.to_string())
    json.dump({"results": out, "per_fold": per_fold.to_dict()},
              open("../runs/walkforward/carry_check.json", "w"), indent=2)
    print("written ../runs/walkforward/carry_check.json")


if __name__ == "__main__":
    main()
