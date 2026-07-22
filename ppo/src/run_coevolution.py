"""Entry point for the three-way ablation: single / population / coevolution.

Each (mode, seed) run gets its own output directory containing config.json,
generations.csv (internal + external ledgers), and best_trader.zip.

Examples (from ppo/src):
    python run_coevolution.py --mode single --seeds 42 43 44
    python run_coevolution.py --mode population --seeds 42 43 44
    python run_coevolution.py --mode coevolution --seeds 42 43 44
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch

from segments import build_segment_pool
from coevo.engine import CoevolutionEngine, EngineConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "population", "coevolution"],
                        default="coevolution")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-data", default="../data/processed/train.parquet")
    parser.add_argument("--val-data", default="../data/processed/val.parquet")
    parser.add_argument("--out-root", default="../runs")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--n-traders", type=int, default=8)
    parser.add_argument("--n-env", type=int, default=12)
    parser.add_argument("--timesteps-per-match", type=int, default=8192)
    parser.add_argument("--segment-length", type=int, default=720)
    parser.add_argument("--segment-stride", type=int, default=168)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train_data)
    val_df = pd.read_parquet(args.val_data)

    try:
        import joblib
        feature_columns = joblib.load("../models/feature_columns.pkl")
    except FileNotFoundError:
        from train_ppo import FEATURE_COLUMNS as feature_columns

    pool = build_segment_pool(
        train_df,
        segment_length=args.segment_length,
        stride=args.segment_stride,
    )
    print(f"Segment pool: {len(pool)} segments, regimes: {pool.regimes()}")

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = EngineConfig(
            mode=args.mode,
            seed=seed,
            generations=args.generations,
            n_traders=args.n_traders,
            n_env=args.n_env,
            timesteps_per_match=args.timesteps_per_match,
        )
        out_dir = f"{args.out_root}/{args.mode}_seed{seed}"
        engine = CoevolutionEngine(config, pool, feature_columns, val_df, out_dir)

        def report(gen, row):
            print(
                f"[{args.mode} seed={seed}] gen {gen:3d} | "
                f"internal fit (mean/max): "
                f"{row['internal_trader_fitness_mean']:+.3f}/"
                f"{row['internal_trader_fitness_max']:+.3f} | "
                f"external val sharpe: {row['external_val_sharpe']:+.3f} "
                f"(B&H {row['baseline_bh_sharpe']:+.3f}, "
                f"MA {row['baseline_ma_sharpe']:+.3f})"
            )

        _, model_path = engine.run(on_generation=report)
        print(f"[{args.mode} seed={seed}] done -> {model_path}.zip")


if __name__ == "__main__":
    main()
