"""Walk-forward ablation: train + evaluate across multiple rolling windows.

The chronological single split (train<2025 / test Oct2025+) has NO bull regime
in the test window and only 9 non-overlapping test segments, so it can neither
test bull-market generalization nor reach significance at the observed effect
size. This driver rolls train/test folds across the FULL feature history, so
different folds land on different regimes (including bull), and runs the
ablation (single / population / coevolution, optionally a variant such as
coevolution with the regret adversary) per fold.

Per fold: the scaler is re-fit on that fold's train window only (no lookahead),
a validation tail is carved off the train window for the engine's external
yardstick, the trader trains on the rest, and the frozen traders are evaluated
per-regime on the fold's held-out test window against buy-and-hold and
MA-crossover under the same fee + slippage model.

Two phases, so the work can be sharded across processes (each fold x mode x
seed is one independent job; see run_walkforward.sh):

  train      python run_walkforward.py --folds 0 1 --modes coevolution --seeds 42 \
                 [any engine flag from engine_cli.py] --out-root ../runs/walkforward
  aggregate  python run_walkforward.py --aggregate-only --out-root ../runs/walkforward

Engine knobs are the SAME flags as run_coevolution.py (engine_cli.py), so a
walk-forward run with V5's flags uses V5's exact EngineConfig.
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from segments import build_segment_pool, walk_forward_folds
from coevo.engine import CoevolutionEngine
from engine_cli import add_engine_args, config_from_args
import evaluate as ev
from project_paths import project_path


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

    Mirrors evaluate.py's aggregation, reusing its helpers. Also returns the
    per-(mode, model, segment) Sharpe table so folds can be pooled later.
    """
    pool = build_segment_pool(test_df, segment_length=segment_length, stride=segment_stride)
    regimes = sorted({s.regime for s in pool.segments})
    baseline_rows = ev.eval_baselines_per_segment(
        pool, env_kwargs["initial_balance"], env_kwargs["trading_fee"],
        env_kwargs.get("slippage", 0.0))

    out = {"n_segments": len(pool), "regimes": pool.regimes(),
           "per_regime": {}, "overall": {}, "significance": {}}
    table = []   # long-format rows for compare_arms.py

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
        per_model = []
        for p in model_paths:
            rows = ev.eval_model_per_segment(p, pool, env_kwargs, feature_columns)
            per_model.append(rows)
            for i, (r, s) in enumerate(zip(rows, pool.segments)):
                table.append({"arm": mode, "model": p, "segment": i,
                              "regime": s.regime, "start": s.start,
                              "sharpe": r["sharpe"],
                              "total_return_pct": r["total_return_pct"],
                              "max_drawdown_pct": r["max_drawdown_pct"],
                              "trades": r["trades"], "exposure": r["exposure"]})
        out["overall"].setdefault(mode, aggregate(per_model))
        for regime in regimes:
            out["per_regime"].setdefault(regime, {})[mode] = aggregate(per_model, regime)
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
        for i, (r, s) in enumerate(zip(baseline_rows, pool.segments)):
            table.append({"arm": name, "model": name, "segment": i,
                          "regime": s.regime, "start": s.start,
                          "sharpe": r[name]["sharpe"],
                          "total_return_pct": r[name]["total_return_pct"],
                          "max_drawdown_pct": r[name]["max_drawdown_pct"],
                          "trades": float("nan"), "exposure": float("nan")})
    return out, table


def fold_dir(out_root, k):
    return os.path.join(out_root, f"fold{k}")


def describe_folds(df, folds, time_col="open_time"):
    rows = []
    for k, f in enumerate(folds):
        row = {"fold": k, "train_start": f.train_start, "train_end": f.train_end,
               "test_start": f.test_start, "test_end": f.test_end}
        if time_col in df.columns:
            row["train_from"] = str(df[time_col].iloc[f.train_start])
            row["test_from"] = str(df[time_col].iloc[f.test_start])
            row["test_to"] = str(df[time_col].iloc[f.test_end - 1])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=project_path("data/features/btcusdt_features.parquet"))
    parser.add_argument("--feature-cols", default=project_path("models/feature_columns.pkl"))
    parser.add_argument("--modes", nargs="+",
                        default=["single", "population", "coevolution"],
                        help="engine modes to train; append ':name' to give a run a "
                             "different directory name, e.g. coevolution:coevolution_regret")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out-root", default=project_path("runs/walkforward"))
    # fold geometry (rows are 1h bars)
    parser.add_argument("--train-span", type=int, default=40000)
    parser.add_argument("--test-span", type=int, default=6000)
    parser.add_argument("--val-span", type=int, default=4000,
                        help="held out from the tail of each train window for the yardstick")
    parser.add_argument("--step", type=int, default=None, help="defaults to test-span")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="fold indices to train (default: all)")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--segment-length", type=int, default=720)
    parser.add_argument("--segment-stride", type=int, default=168)
    parser.add_argument("--eval-segment-stride", type=int, default=720,
                        help="non-overlapping test segments for independent significance")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip training; evaluate every model found under --out-root")
    add_engine_args(parser)
    args = parser.parse_args()

    df = pd.read_parquet(args.features).reset_index(drop=True)
    feature_columns = load_feature_columns(args.feature_cols)
    env_kwargs = {"initial_balance": args.initial_balance,
                  "trading_fee": args.trading_fee, "slippage": args.slippage}

    folds = walk_forward_folds(len(df), args.train_span, args.test_span, step=args.step)
    if args.max_folds:
        folds = folds[:args.max_folds]
    print(f"Walk-forward: {len(folds)} folds over {len(df)} rows "
          f"(train {args.train_span}, test {args.test_span})", flush=True)
    os.makedirs(args.out_root, exist_ok=True)
    folds_path = os.path.join(args.out_root, "folds.json")
    if not os.path.exists(folds_path):
        with open(folds_path, "w") as f:
            json.dump({"geometry": {"train_span": args.train_span, "test_span": args.test_span,
                                    "val_span": args.val_span, "step": args.step,
                                    "features": args.features},
                       "folds": describe_folds(df, folds)}, f, indent=2)

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    if not args.aggregate_only:
        which = args.folds if args.folds else list(range(len(folds)))
        for k in which:
            fold = folds[k]
            train_raw = df.iloc[fold.train_start:fold.train_end]
            test_raw = df.iloc[fold.test_start:fold.test_end]
            train_s, test_s = scale_fold(train_raw, test_raw, feature_columns)
            fdir = fold_dir(args.out_root, k)
            os.makedirs(fdir, exist_ok=True)
            test_path = os.path.join(fdir, "test_scaled.parquet")
            if not os.path.exists(test_path):
                test_s.to_parquet(test_path)   # the exact window the models are judged on

            # carve a validation tail off the train window for the external yardstick;
            # the trader trains only on the earlier part so it never sees the val tail
            val_s = train_s.iloc[-args.val_span:].reset_index(drop=True)
            pool_train = train_s.iloc[:-args.val_span].reset_index(drop=True)
            pool = build_segment_pool(pool_train, segment_length=args.segment_length,
                                      stride=args.segment_stride)
            print(f"\n=== Fold {k}: train rows {len(pool_train)} "
                  f"(pool {len(pool)} segments, {pool.regimes()}), "
                  f"test rows {len(test_s)} ===", flush=True)

            for spec in args.modes:
                mode, _, name = spec.partition(":")
                name = name or mode
                for seed in args.seeds:
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    config = config_from_args(args, mode, seed)
                    out_dir = os.path.join(fdir, f"{name}_seed{seed}")
                    engine = CoevolutionEngine(config, pool, feature_columns, val_s, out_dir)
                    engine.run(on_generation=lambda g, r, n=name, s=seed, kk=k: print(
                        f"  [fold{kk} {n} seed{s}] gen {g:3d} internal "
                        f"{r['internal_trader_fitness_mean']:+.3f} external "
                        f"{r['external_val_sharpe']:+.3f}", flush=True))
                    print(f"  [fold{k} {name} seed{seed}] done", flush=True)
        if args.folds:
            print("\nShard finished. Run with --aggregate-only once every shard is done.")
            return

    # ------------------------------------------------------------------
    # aggregate: everything that exists under out_root
    # ------------------------------------------------------------------
    all_results = {"folds": [], "config": vars(args)}
    long_rows = []
    for k in range(len(folds)):
        fdir = fold_dir(args.out_root, k)
        test_path = os.path.join(fdir, "test_scaled.parquet")
        if not os.path.exists(test_path):
            print(f"fold {k}: no test_scaled.parquet, skipping")
            continue
        test_s = pd.read_parquet(test_path)
        models_by_mode = {}
        for p in sorted(glob.glob(os.path.join(fdir, "*_seed*", "best_trader.zip"))):
            name = os.path.basename(os.path.dirname(p)).rsplit("_seed", 1)[0]
            models_by_mode.setdefault(name, []).append(p)
        if not models_by_mode:
            print(f"fold {k}: no models, skipping")
            continue
        fold_eval, table = evaluate_fold(models_by_mode, test_s, env_kwargs, feature_columns,
                                         args.segment_length, args.eval_segment_stride)
        fold_eval["fold"] = k
        fold_eval["test_regimes"] = fold_eval["regimes"]
        all_results["folds"].append(fold_eval)
        for r in table:
            r["fold"] = k
        long_rows += table
        with open(os.path.join(fdir, "fold_eval.json"), "w") as f:
            json.dump(fold_eval, f, indent=2)

        print(f"  fold {k} test regimes: {fold_eval['regimes']}")
        bh = fold_eval["overall"]["buy_hold"]["sharpe"]["mean"]
        for mode in models_by_mode:
            s = fold_eval["overall"][mode].get("sharpe", {})
            if s:
                print(f"    {mode:20s} sharpe {s['mean']:+.3f}±{s['std']:.3f}  "
                      f"(n={len(models_by_mode[mode])}) vs B&H {bh:+.3f}")

    out_path = os.path.join(args.out_root, "walkforward_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    pd.DataFrame(long_rows).to_csv(os.path.join(args.out_root, "per_segment.csv"), index=False)
    print(f"\nResults written to {out_path} and per_segment.csv "
          f"(pool folds with compare_arms.py --walkforward-root)")


if __name__ == "__main__":
    main()
