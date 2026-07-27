"""Walk-forward ablation: train + evaluate across multiple rolling windows.

The chronological single split (train<2025 / test Oct2025+) has NO bull regime
in the test window, so it can't test bull-market generalization. This driver
rolls train/test folds across the FULL feature history, so different folds land
on different regimes (including bull), and runs the three-way ablation
(single / population / coevolution) per fold.

Per fold: the scaler is re-fit on that fold's train window only (no lookahead),
the trader trains on the train window, and the frozen best trader is evaluated
per-regime on the fold's held-out test window against buy-and-hold and
MA-crossover, with a paired-bootstrap significance check — reusing evaluate.py.

This RETRAINS per fold, so it is its own compute chunk: run with reduced
seeds/generations. Start tiny to validate, then scale.

Example (from ppo/src):
    python run_walkforward.py --seeds 42 43 44 --generations 20 \
        --train-span 40000 --test-span 6000 --val-span 4000
"""

import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from segments import build_segment_pool, walk_forward_folds
from coevo.engine import CoevolutionEngine, EngineConfig
import evaluate as ev


def load_feature_columns(path):
    try:
        import joblib
        return joblib.load(path)
    except FileNotFoundError:
        from train_ppo import FEATURE_COLUMNS
        return FEATURE_COLUMNS


def scale_fold(train_df, test_df, feature_columns):
    """Fit StandardScaler on the fold's train window only, transform both."""
    scaler = StandardScaler()
    train_s = train_df.copy()
    test_s = test_df.copy()
    train_s.loc[:, feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_s.loc[:, feature_columns] = scaler.transform(test_df[feature_columns])
    return train_s.reset_index(drop=True), test_s.reset_index(drop=True)


def evaluate_fold(models_by_mode, test_df, env_kwargs, feature_columns,
                  segment_length, segment_stride):
    """Per-regime metrics + paired bootstrap for each mode on the fold's test window.

    Mirrors evaluate.py's aggregation, reusing its helpers, but takes the models
    trained in-process for this fold.
    """
    pool = build_segment_pool(test_df, segment_length=segment_length, stride=segment_stride)
    regimes = sorted({s.regime for s in pool.segments})
    baseline_rows = ev.eval_baselines_per_segment(
        pool, env_kwargs["initial_balance"], env_kwargs["trading_fee"])

    out = {"n_segments": len(pool), "regimes": pool.regimes(),
           "per_regime": {}, "overall": {}, "significance": {}}

    def aggregate(rows_per_model, regime=None):
        agg = {}
        for metric in ev.METRIC_KEYS:
            seed_means = []
            for rows in rows_per_model:
                vals = [r[metric] for r, s in zip(rows, pool.segments)
                        if regime is None or s.regime == regime]
                if vals:
                    seed_means.append(float(np.mean(vals)))
            if seed_means:
                agg[metric] = {"mean": float(np.mean(seed_means)),
                               "std": float(np.std(seed_means))}
        return agg

    for mode, model_paths in models_by_mode.items():
        per_model = [ev.eval_model_per_segment(p, pool, env_kwargs, feature_columns)
                     for p in model_paths]
        out["overall"].setdefault(mode, aggregate(per_model))
        for regime in regimes:
            out["per_regime"].setdefault(regime, {})[mode] = aggregate(per_model, regime)
        # paired bootstrap on per-segment sharpe: model (mean across seeds) vs each baseline
        model_sharpes = np.mean([[r["sharpe"] for r in rows] for rows in per_model], axis=0)
        out["significance"][mode] = {
            name: ev.paired_bootstrap(model_sharpes, [r[name]["sharpe"] for r in baseline_rows])
            for name in ["buy_hold", "ma_crossover"]
        }

    for name in ["buy_hold", "ma_crossover"]:
        rows = [r[name] for r in baseline_rows]
        out["overall"][name] = {m: {"mean": float(np.mean([r[m] for r in rows])), "std": 0.0}
                                for m in ev.METRIC_KEYS}
        for regime in regimes:
            vals = [r[name] for r, s in zip(baseline_rows, pool.segments) if s.regime == regime]
            out["per_regime"].setdefault(regime, {})[name] = {
                m: {"mean": float(np.mean([v[m] for v in vals])), "std": 0.0}
                for m in ev.METRIC_KEYS}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="../data/features/btcusdt_features.parquet")
    parser.add_argument("--feature-cols", default="../models/feature_columns.pkl")
    parser.add_argument("--modes", nargs="+",
                        default=["single", "population", "coevolution"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out-root", default="../runs/walkforward")
    # fold geometry (rows are 1h bars)
    parser.add_argument("--train-span", type=int, default=40000)
    parser.add_argument("--test-span", type=int, default=6000)
    parser.add_argument("--val-span", type=int, default=4000,
                        help="held out from the tail of each train window for the yardstick")
    parser.add_argument("--step", type=int, default=None, help="defaults to test-span")
    parser.add_argument("--max-folds", type=int, default=None)
    # engine knobs (reduced defaults — walk-forward retrains per fold)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--n-traders", type=int, default=8)
    parser.add_argument("--n-env", type=int, default=12)
    parser.add_argument("--timesteps-per-match", type=int, default=8192)
    parser.add_argument("--segments-per-match", type=int, default=4)
    parser.add_argument("--env-weight-sigma", type=float, default=0.3)
    parser.add_argument("--env-temp-sigma", type=float, default=0.15)
    parser.add_argument("--segment-length", type=int, default=720)
    parser.add_argument("--segment-stride", type=int, default=168)
    parser.add_argument("--eval-segment-stride", type=int, default=720,
                        help="non-overlapping test segments for independent significance")
    args = parser.parse_args()

    df = pd.read_parquet(args.features).reset_index(drop=True)
    feature_columns = load_feature_columns(args.feature_cols)
    env_kwargs = {"initial_balance": 10000.0, "trading_fee": 0.001, "slippage": 0.0005}

    folds = walk_forward_folds(len(df), args.train_span, args.test_span, step=args.step)
    if args.max_folds:
        folds = folds[:args.max_folds]
    print(f"Walk-forward: {len(folds)} folds over {len(df)} rows "
          f"(train {args.train_span}, test {args.test_span})")

    os.makedirs(args.out_root, exist_ok=True)
    all_results = {"folds": [], "config": vars(args)}

    for k, fold in enumerate(folds):
        train_raw = df.iloc[fold.train_start:fold.train_end]
        test_raw = df.iloc[fold.test_start:fold.test_end]
        train_s, test_s = scale_fold(train_raw, test_raw, feature_columns)

        # carve a validation tail off the train window for the external yardstick;
        # the trader trains only on the earlier part so it never sees the val tail
        val_s = train_s.iloc[-args.val_span:].reset_index(drop=True)
        pool_train = train_s.iloc[:-args.val_span].reset_index(drop=True)
        pool = build_segment_pool(pool_train, segment_length=args.segment_length,
                                  stride=args.segment_stride)
        print(f"\n=== Fold {k}: train rows {len(pool_train)} "
              f"(pool {len(pool)} segments, {pool.regimes()}), "
              f"test rows {len(test_s)} ===")

        models_by_mode = {}
        for mode in args.modes:
            model_paths = []
            for seed in args.seeds:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                config = EngineConfig(
                    mode=mode, seed=seed, generations=args.generations,
                    n_traders=args.n_traders, n_env=args.n_env,
                    timesteps_per_match=args.timesteps_per_match,
                    segments_per_match=args.segments_per_match,
                    env_weight_sigma=args.env_weight_sigma,
                    env_temp_sigma=args.env_temp_sigma,
                )
                out_dir = f"{args.out_root}/fold{k}/{mode}_seed{seed}"
                engine = CoevolutionEngine(config, pool, feature_columns, val_s, out_dir)
                _, model_path = engine.run()
                model_paths.append(model_path + ".zip")
                print(f"  [fold{k} {mode} seed{seed}] done")
            models_by_mode[mode] = model_paths

        fold_eval = evaluate_fold(models_by_mode, test_s, env_kwargs, feature_columns,
                                  args.segment_length, args.eval_segment_stride)
        fold_eval["fold"] = k
        fold_eval["test_regimes"] = fold_eval["regimes"]
        all_results["folds"].append(fold_eval)

        print(f"  fold {k} test regimes: {fold_eval['regimes']}")
        for mode in args.modes:
            s = fold_eval["overall"][mode].get("sharpe", {})
            bh = fold_eval["overall"]["buy_hold"]["sharpe"]["mean"]
            if s:
                print(f"    {mode:12s} sharpe {s['mean']:+.3f}±{s['std']:.3f}  vs B&H {bh:+.3f}")

    out_path = os.path.join(args.out_root, "walkforward_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
