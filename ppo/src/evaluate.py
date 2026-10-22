"""Evaluation harness — deliberately OUTSIDE the coevolutionary arms race.

Evaluates frozen models on held-out data cut into regime-labeled segments,
compares against buy-and-hold and MA-crossover on the exact same segments
and cost model, aggregates across seeds, and reports a paired-bootstrap
significance check so outperformance isn't just data-mined luck.

Example (from ppo/src):
    python evaluate.py \
        --models ../runs/coevolution_seed42/best_trader.zip \
                 ../runs/coevolution_seed43/best_trader.zip \
        --data ../data/processed/test.parquet
"""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from segments import build_segment_pool
from baselines import buy_and_hold, ma_crossover
from metrics import summarize
from coevo.trader_population import evaluate_on_segments

METRIC_KEYS = ["total_return_pct", "sharpe", "sortino", "max_drawdown_pct", "calmar"]


def eval_baselines_per_segment(pool, initial_balance, trading_fee, slippage=0.0):
    """Baselines on every segment under the SAME fee + slippage model as the agent.

    slippage defaults to 0.0 so older callers reproduce the V1-V4 numbers;
    evaluate.py's CLI passes the env's slippage.
    """
    rows = []
    for seg in pool.segments:
        seg_df = pool.slice(seg)
        rows.append({
            "regime": seg.regime,
            "buy_hold": summarize(buy_and_hold(
                seg_df, initial_balance, trading_fee, slippage)),
            "ma_crossover": summarize(ma_crossover(
                seg_df, initial_balance, trading_fee, slippage=slippage)),
        })
    return rows


def eval_model_per_segment(model_path, pool, env_kwargs, feature_columns):
    trader = SimpleNamespace(model=PPO.load(model_path))
    seg_ranges = [(s.start, s.end) for s in pool.segments]
    _, per_segment = evaluate_on_segments(
        trader, pool.df, feature_columns, seg_ranges, env_kwargs)
    return per_segment


def paired_bootstrap(model_scores, baseline_scores, n_boot=10000, seed=0):
    """P(model <= baseline) under a paired bootstrap over segments."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(model_scores) - np.asarray(baseline_scores)
    n = len(diffs)
    boot_means = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    return float((boot_means <= 0).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True,
                        help="one or more best_trader.zip paths (one per seed)")
    parser.add_argument("--data", default="../data/processed/test.parquet")
    parser.add_argument("--features", default="../models/feature_columns.pkl")
    parser.add_argument("--segment-length", type=int, default=720)
    parser.add_argument("--segment-stride", type=int, default=720,
                        help="non-overlapping by default for independent segments")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--trading-fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--out", default=None, help="optional JSON results path")
    args = parser.parse_args()

    import joblib
    feature_columns = joblib.load(args.features)

    df = pd.read_parquet(args.data)
    pool = build_segment_pool(
        df, segment_length=args.segment_length, stride=args.segment_stride)
    print(f"Evaluation segments: {len(pool)}, regimes: {pool.regimes()}")

    env_kwargs = {
        "initial_balance": args.initial_balance,
        "trading_fee": args.trading_fee,
        "slippage": args.slippage,
    }

    baseline_rows = eval_baselines_per_segment(
        pool, args.initial_balance, args.trading_fee, args.slippage)

    per_model = {}
    for path in args.models:
        print(f"Evaluating {path} ...")
        per_model[path] = eval_model_per_segment(
            path, pool, env_kwargs, feature_columns)

    regimes = sorted({s.regime for s in pool.segments})
    results = {"n_segments": len(pool), "per_regime": {}, "overall": {}}

    def aggregate(rows_per_model, key_filter=None):
        """mean +/- std across seeds of the per-segment mean of each metric."""
        agg = {}
        for metric in METRIC_KEYS:
            seed_means = []
            for rows in rows_per_model.values():
                vals = [
                    r[metric] for r, s in zip(rows, pool.segments)
                    if key_filter is None or s.regime == key_filter
                ]
                if vals:
                    seed_means.append(float(np.mean(vals)))
            if seed_means:
                agg[metric] = {
                    "mean": float(np.mean(seed_means)),
                    "std": float(np.std(seed_means)),
                }
        return agg

    results["overall"]["model"] = aggregate(per_model)
    for regime in regimes:
        results["per_regime"][regime] = {"model": aggregate(per_model, regime)}

    for name in ["buy_hold", "ma_crossover"]:
        rows = [r[name] for r in baseline_rows]
        results["overall"][name] = {
            m: {"mean": float(np.mean([r[m] for r in rows])), "std": 0.0}
            for m in METRIC_KEYS
        }
        for regime in regimes:
            vals = [
                r[name] for r, s in zip(baseline_rows, pool.segments)
                if s.regime == regime
            ]
            results["per_regime"][regime][name] = {
                m: {"mean": float(np.mean([v[m] for v in vals])), "std": 0.0}
                for m in METRIC_KEYS
            }

    # paired bootstrap on per-segment sharpe, model (mean across seeds) vs each baseline
    model_sharpes = np.mean(
        [[r["sharpe"] for r in rows] for rows in per_model.values()], axis=0)
    results["significance"] = {
        name: paired_bootstrap(
            model_sharpes, [r[name]["sharpe"] for r in baseline_rows])
        for name in ["buy_hold", "ma_crossover"]
    }

    print(f"\n{'':<14}{'sharpe':>12}{'return %':>12}{'max DD %':>12}")
    for name in ["model", "buy_hold", "ma_crossover"]:
        agg = results["overall"][name]
        print(
            f"{name:<14}"
            f"{agg['sharpe']['mean']:>12.3f}"
            f"{agg['total_return_pct']['mean']:>12.2f}"
            f"{agg['max_drawdown_pct']['mean']:>12.2f}"
        )
    print(f"\nSeeds: {len(per_model)}  (std across seeds in JSON output)")
    for name, p in results["significance"].items():
        print(f"P(model <= {name}) on per-segment Sharpe (paired bootstrap): {p:.4f}")
    for regime in regimes:
        m = results["per_regime"][regime]["model"].get("sharpe")
        b = results["per_regime"][regime]["buy_hold"]["sharpe"]
        if m:
            print(f"  {regime:<10} model sharpe {m['mean']:+.3f}±{m['std']:.3f}"
                  f" vs B&H {b['mean']:+.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
