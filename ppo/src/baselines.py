"""Buy-and-hold and moving-average crossover baselines, using the same fee
model as TradingEnv, so PPO results have a real, reproducible comparison
point instead of the unreproducible figures cited in the PR2 report.
"""

import argparse

import numpy as np
import pandas as pd

from metrics import summarize


def buy_and_hold(df, initial_balance=10000.0, trading_fee=0.001, slippage=0.0):
    """Buy once at the first close (filled above close by `slippage`, same as
    TradingEnv), never sell. Default slippage 0.0 reproduces the published
    V1-V4 baseline numbers; pass the env's slippage for a like-for-like fee model."""
    price = df["close"].values
    btc = (initial_balance * (1 - trading_fee)) / (price[0] * (1 + slippage))
    return btc * price


def ma_crossover(df, initial_balance=10000.0, trading_fee=0.001, fast=20, slow=50,
                 slippage=0.0):
    """20/50 MA crossover. Buys fill above close and sells below close by
    `slippage`, mirroring TradingEnv.step, so the rule-based baseline pays the
    same execution cost as the agent. Default 0.0 reproduces V1-V4 numbers."""
    price = df["close"].values
    fast_ma = df["close"].rolling(fast).mean().values
    slow_ma = df["close"].rolling(slow).mean().values

    cash = initial_balance
    btc = 0.0
    position = 0
    portfolio_values = []

    for i in range(len(price)):
        if i >= slow:
            signal_long = fast_ma[i] > slow_ma[i]
            if signal_long and position == 0:
                cash_to_spend = cash * (1 - trading_fee)
                btc = cash_to_spend / (price[i] * (1 + slippage))
                cash = 0.0
                position = 1
            elif not signal_long and position == 1:
                cash = btc * price[i] * (1 - slippage) * (1 - trading_fee)
                btc = 0.0
                position = 0

        portfolio_values.append(cash + btc * price[i])

    return np.array(portfolio_values)


def report(name, portfolio_values):
    result = summarize(portfolio_values)
    print(f"\n=== {name} ===")
    for key, value in result.items():
        print(f"{key}: {value:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run buy-and-hold and MA-crossover baselines on a held-out split"
    )
    parser.add_argument("--data", default="../data/processed/test.parquet")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--trading-fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005,
                        help="per-side slippage, same as TradingEnv (0 = legacy numbers)")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    report("Buy & Hold", buy_and_hold(df, args.initial_balance, args.trading_fee, args.slippage))
    report("MA Crossover (20/50)", ma_crossover(df, args.initial_balance, args.trading_fee,
                                                slippage=args.slippage))


if __name__ == "__main__":
    main()
