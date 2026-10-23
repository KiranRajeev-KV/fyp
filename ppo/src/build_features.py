"""Extend and numerically compare the original notebook feature pipeline.

Requires raw CSVs, the existing base feature file, a fitted feature scaler,
and prepared data/cleaned, data/features, and data/processed directories.
This is an extension/reproduction tool, not a bootstrap for the three notebooks.
It leaves the saved scaler fitted on the original pre-2025 training data.

Default new outputs under ppo/:
    data/cleaned/btcusdt_1h_clean_ext.parquet
    data/features/btcusdt_features_ext.parquet
    data/processed/test_ext.parquet      (>= --test-from, transformed by saved scaler)

Existing output paths are refused. The overlap check compares feature values,
but its current left-join/NaN handling does not fully validate missing matches
or nonfinite values. The indicator formulas match feature_engineering.ipynb.

From the repo root:
    python ppo/src/build_features.py
    python ppo/src/build_features.py --check-only
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands
from feature_columns import FEATURE_COLUMNS
from project_paths import project_path

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
           "taker_buy_quote_volume", "ignore"]


def load_raw(raw_dir):
    """data_collect.ipynb: merge monthly CSVs, normalise ms/us timestamps, dedupe."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    df = pd.concat([pd.read_csv(f, header=None, names=COLUMNS) for f in files],
                   ignore_index=True)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    # Binance switched open_time from milliseconds (13 digits) to microseconds
    # (16 digits) in Jan 2025
    ts = df["open_time"].astype("int64")
    ts = ts.where(ts.astype(str).str.len() != 16, ts // 1000)
    df["open_time"] = pd.to_datetime(ts, unit="ms", utc=True)
    df = (df.sort_values("open_time").drop_duplicates(subset=["open_time"])
            .reset_index(drop=True))
    expected = pd.date_range(df["open_time"].min(), df["open_time"].max(), freq="1h", tz="UTC")
    missing = expected.difference(df["open_time"])
    print(f"raw: {len(files)} files, {len(df)} candles, {df['open_time'].min()} .. "
          f"{df['open_time'].max()}, missing candles: {len(missing)}")
    return df


def add_features(df):
    """feature_engineering.ipynb, verbatim."""
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
    df["open_close_spread"] = (df["close"] - df["open"]) / df["open"]
    for span in (20, 50, 100, 200):
        df[f"ema{span}"] = df["close"].ewm(span=span).mean()
    df["ema20_ema50_ratio"] = df["ema20"] / df["ema50"]
    df["ema50_ema200_ratio"] = df["ema50"] / df["ema200"]
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"],
                                 window=14).average_true_range()
    df["volatility_24h"] = df["return"].rolling(24).std()
    df["volatility_168h"] = df["return"].rolling(168).std()
    df["volume_ma24"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma24"]
    df["volume_change"] = df["volume"].pct_change()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["price_vs_ema50"] = df["close"] / df["ema50"]
    df["price_vs_ema200"] = df["close"] / df["ema200"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df["open_time"] = df["open_time"].dt.tz_localize(None)
    return df


def check_reproduction(new, existing_path):
    old = pd.read_parquet(existing_path)
    merged = old.merge(new, on="open_time", suffixes=("_old", "_new"), how="left")
    assert len(merged) == len(old), "rows of the existing feature file are missing"
    worst = 0.0
    for c in FEATURE_COLUMNS + ["open", "high", "low", "close", "volume"]:
        a, b = merged[f"{c}_old"].values, merged[f"{c}_new"].values
        rel = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12))
        worst = max(worst, rel)
    print(f"reproduction check vs {os.path.basename(existing_path)}: {len(old)} rows, "
          f"worst relative difference {worst:.2e}")
    assert worst < 1e-9, "the scripted pipeline does not reproduce the notebook features"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=project_path("data/raw"))
    ap.add_argument("--existing-features", default=project_path("data/features/btcusdt_features.parquet"))
    ap.add_argument("--scaler", default=project_path("models/feature_scaler.pkl"))
    ap.add_argument("--test-from", default="2025-10-01",
                    help="start of the held-out window (same as train_test.ipynb)")
    ap.add_argument("--suffix", default="_ext")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    raw = load_raw(args.raw_dir)
    feats = add_features(raw)
    print(f"features: {len(feats)} rows, {feats['open_time'].min()} .. {feats['open_time'].max()}")
    check_reproduction(feats, args.existing_features)
    if args.check_only:
        return

    cleaned_path = project_path(f"data/cleaned/btcusdt_1h_clean{args.suffix}.parquet")
    feat_path = project_path(f"data/features/btcusdt_features{args.suffix}.parquet")
    test_path = project_path(f"data/processed/test{args.suffix}.parquet")
    for p in (cleaned_path, feat_path, test_path):
        if os.path.exists(p):
            raise FileExistsError(f"{p} exists; delete it first or use another --suffix")

    raw.to_parquet(cleaned_path, index=False)
    feats.to_parquet(feat_path, index=False)

    scaler = joblib.load(args.scaler)
    test = feats[feats["open_time"] >= args.test_from].copy()
    test.loc[:, FEATURE_COLUMNS] = scaler.transform(test[FEATURE_COLUMNS])
    test = test.reset_index(drop=True)
    test.to_parquet(test_path, index=False)
    print(f"written {cleaned_path}, {feat_path}, {test_path} "
          f"({len(test)} test rows, {test['open_time'].min()} .. {test['open_time'].max()})")


if __name__ == "__main__":
    main()
