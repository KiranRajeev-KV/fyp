"""Task B: controlled synthetic market with a hidden, learnable regime.

A separate benchmark motivated by weak timing signals in the historical BTC
experiments (feature AUC ~0.53 under those probes). Here timing is possible by
construction, and the training/test regime mix is shifted on purpose. Task B
and R-number labels refer to the original local experiment ledger. Metadata
records the benchmark name and parameters; source hashes are not recorded.

Generator: semi-Markov regimes (bull / bear / sideways). At each switch the
next regime is drawn from a target mix, durations are geometric with the same
mean for every regime, so the long-run occupancy equals the mix. Hourly log
returns are Gaussian with regime-dependent drift and volatility. One extra
observable feature, `regime_signal`, is an EMA of (regime code + noise): it
reveals the regime with noise and lag. The usual 19 features are computed with
build_features.add_features, so the engine runs unchanged. Every bar keeps its
TRUE regime in `true_regime` (segments.build_segment_pool uses it if present).

Splits (each a parquet in --out, features standardised with train stats):
    train.parquet        one series, TRAIN mix (bear rare), seed 0
    val.parquet          one series, TRAIN mix, seed 1 (engine's external val)
    dev_shift.parquet    seeds 1000-1099, SHIFTED mix (bear ~50%)
    dev_control.parquet  seeds 2000-2099, TRAIN mix (the control)
    sealed_shift.parquet seeds 9000-9099, SHIFTED mix  -- only with --sealed
Test files are 100 independent blocks of 2 x 720 bars, so stride-720 segments
never cross a block: 200 non-overlapping segments per test set.

Examples from ppo/src:
    python synth_market.py --out ../data/synth_v1            # build dev splits
    python synth_market.py --out ../data/synth_v1 --check    # + cheap check (R4)
    python synth_market.py --out ../data/synth_v1 --sealed   # once, at the end
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from build_features import FEATURE_COLUMNS, add_features
from metrics import sharpe_ratio

REGIMES = ["bull", "sideways", "bear"]
CODE = {"bull": 1.0, "sideways": 0.0, "bear": -1.0}

# Historical SYNTH_V1 parameters. Treat parameter changes as a new benchmark.
PARAMS = {
    "drift": {"bull": 0.0004, "sideways": 0.0, "bear": -0.0008},   # per hour
    "vol": {"bull": 0.006, "sideways": 0.006, "bear": 0.009},      # per hour
    "mean_duration": 480,          # bars (~20 days)
    "train_mix": {"bull": 0.47, "sideways": 0.43, "bear": 0.10},
    "shift_mix": {"bull": 0.25, "sideways": 0.25, "bear": 0.50},
    "signal_noise": 2.0,           # sd of per-bar noise on the regime code
    "signal_span": 48,             # EMA span of the observed signal
    "warmup": 400,                 # bars dropped after feature computation
    "train_bars": 64353, "val_bars": 6552,
    "block_bars": 1440, "n_blocks": 100,
    "seeds": {"train": 0, "val": 1, "dev_shift": 1000, "dev_control": 2000,
              "sealed_shift": 9000},
}
SIGNAL = "regime_signal"
FEATURES = FEATURE_COLUMNS + [SIGNAL]


def simulate(n, mix, seed, p=PARAMS):
    """n bars of OHLCV + true_regime + regime_signal (raw, unscaled)."""
    rng = np.random.default_rng(seed)
    probs = np.array([mix[r] for r in REGIMES])
    reg = np.empty(n, dtype=object)
    t = 0
    while t < n:
        r = REGIMES[rng.choice(3, p=probs)]
        d = rng.geometric(1.0 / p["mean_duration"])
        reg[t:t + d] = r
        t += d
    mu = np.array([p["drift"][r] for r in reg])
    sd = np.array([p["vol"][r] for r in reg])
    log_ret = mu + sd * rng.standard_normal(n)
    close = 20000 * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[20000.0], close[:-1]])
    wick = sd * np.abs(rng.standard_normal((2, n)))
    df = pd.DataFrame({
        "open_time": pd.date_range("2020-01-01", periods=n, freq="1h"),
        "open": open_,
        "high": np.maximum(open_, close) * (1 + wick[0]),
        "low": np.minimum(open_, close) * (1 - wick[1]),
        "close": close,
        "volume": rng.lognormal(7, 0.5, n),
        "true_regime": reg,
    })
    code = np.array([CODE[r] for r in reg]) + p["signal_noise"] * rng.standard_normal(n)
    df[SIGNAL] = pd.Series(code).ewm(span=p["signal_span"]).mean().values
    return df


def featurize(raw, keep):
    """add_features drops warm-up NaN rows; keep exactly the last `keep` bars."""
    extra = raw[["open_time", "true_regime", SIGNAL]]
    f = add_features(raw.drop(columns=["true_regime", SIGNAL]))
    f["open_time"] = pd.to_datetime(f["open_time"])
    extra = extra.assign(open_time=extra["open_time"].dt.tz_localize(None))
    f = f.merge(extra, on="open_time", how="left")
    assert len(f) >= keep, (len(f), keep)
    return f.iloc[-keep:].reset_index(drop=True)


def series(n, mix, seed, p=PARAMS):
    return featurize(simulate(n + p["warmup"], mix, seed, p), n)


def blocks(first_seed, mix, p=PARAMS):
    return pd.concat([series(p["block_bars"], mix, first_seed + i, p)
                      for i in range(p["n_blocks"])], ignore_index=True)


def build(out, sealed=False, p=PARAMS):
    os.makedirs(out, exist_ok=True)
    s = p["seeds"]
    splits = {"train": series(p["train_bars"], p["train_mix"], s["train"], p),
              "val": series(p["val_bars"], p["train_mix"], s["val"], p)}
    if sealed:
        splits = {"sealed_shift": blocks(s["sealed_shift"], p["shift_mix"], p)}
    else:
        splits["dev_shift"] = blocks(s["dev_shift"], p["shift_mix"], p)
        splits["dev_control"] = blocks(s["dev_control"], p["train_mix"], p)
    scaler_path = os.path.join(out, "feature_scaler.pkl")
    if sealed:
        scaler = joblib.load(scaler_path)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(splits["train"][FEATURES])
        joblib.dump(scaler, scaler_path)
        joblib.dump(FEATURES, os.path.join(out, "feature_columns.pkl"))
        with open(os.path.join(out, "params.json"), "w") as metadata:
            json.dump({"benchmark": "SYNTH_V1", "params": p}, metadata, indent=2)
    for name, df in splits.items():
        path = os.path.join(out, f"{name}.parquet")
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists; a benchmark is never rebuilt in place")
        df.loc[:, FEATURES] = scaler.transform(df[FEATURES])
        df.to_parquet(path, index=False)
        print(f"{name:13s} {len(df):7d} bars  regime share "
              f"{df.true_regime.value_counts(normalize=True).round(3).to_dict()}")


# ---------------------------------------------------------------------------
# cheap check (R4): is the benchmark winnable?
# ---------------------------------------------------------------------------

def segment_regimes(df, L=720):
    return [df.true_regime.iloc[i:i + L].mode().iloc[0] for i in range(0, len(df) - L + 1, L)]


def timer_vs_bh(df, prob_up, L=720, rebalance=24, cost=0.0015):
    """Per-segment Sharpe of a long/flat timer (long when P(up) > 0.5, decided
    every `rebalance` bars) and of buy-and-hold, same cost per position change."""
    close = df.close.values
    out = []
    for st in range(0, len(df) - L + 1, L):
        r = close[st + 1:st + L] / close[st:st + L - 1] - 1
        pos = np.zeros(L - 1)
        cur = 0
        for i in range(L - 1):
            if i % rebalance == 0:
                cur = int(prob_up[st + i] > 0.5)
            pos[i] = cur
        changes = np.abs(np.diff(np.concatenate([[0], pos])))
        timer = pos * r - changes * cost
        bh = r.copy()
        bh[0] -= cost
        out.append((sharpe_ratio(timer), sharpe_ratio(bh), pos.mean(),
                    np.log(close[st + L - 1] / close[st])))
    return np.array(out)


def check(out, horizon=24):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    feats = joblib.load(os.path.join(out, "feature_columns.pkl"))

    def xy(df):
        fwd = np.log(df.close.shift(-horizon) / df.close)
        ok = fwd.notna().values
        return df[feats].values[ok], (fwd[ok] > 0).astype(int).values, ok

    tr = pd.read_parquet(os.path.join(out, "train.parquet"))
    X, y, _ = xy(tr)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(X, y)
    res = {}
    for name in ["dev_shift", "dev_control"]:
        df = pd.read_parquet(os.path.join(out, f"{name}.parquet"))
        Xt, yt, _ = xy(df)
        p_all = clf.predict_proba(df[feats].values)[:, 1]
        auc = roc_auc_score(yt, clf.predict_proba(Xt)[:, 1])
        # signal-only and price-only AUCs: which part carries the timing
        auc_sig = roc_auc_score(yt, Xt[:, feats.index(SIGNAL)])
        tb = timer_vs_bh(df, p_all)
        regs = segment_regimes(df)
        res[name] = {
            "auc": round(float(auc), 4), "auc_signal_only": round(float(auc_sig), 4),
            "n_segments": len(tb),
            "bear_segment_share": round(regs.count("bear") / len(regs), 3),
            "timer_sharpe": round(float(tb[:, 0].mean()), 3),
            "bh_sharpe": round(float(tb[:, 1].mean()), 3),
            "timer_wins": int((tb[:, 0] > tb[:, 1]).sum()),
            "timer_exposure": round(float(tb[:, 2].mean()), 3),
            # timing score (R5): exposure in up-segments minus down-segments
            "timer_timing": round(float(tb[tb[:, 3] > 0, 2].mean() - tb[tb[:, 3] < 0, 2].mean()), 3),
            "timer_corr_exposure_move": round(float(np.corrcoef(tb[:, 2], tb[:, 3])[0, 1]), 3),
        }
    res["passed"] = (res["dev_shift"]["timer_sharpe"] > res["dev_shift"]["bh_sharpe"]
                     and 0.58 <= res["dev_shift"]["auc"] <= 0.67)
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(out, "cheap_check.json"), "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="run the cheap check after building")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--sealed", action="store_true", help="build the sealed holdout (once)")
    args = ap.parse_args()
    if not args.check_only:
        build(args.out, sealed=args.sealed)
    if args.check or args.check_only:
        check(args.out)


if __name__ == "__main__":
    main()
