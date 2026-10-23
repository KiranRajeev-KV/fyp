import argparse
import random

import pandas as pd
import numpy as np
import torch
import joblib

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from trading_env import TradingEnv
from metrics import summarize
from project_paths import project_path

# ==========================================
# FEATURE LIST
# ==========================================

FEATURE_COLUMNS = [
    "return",
    "log_return",
    "high_low_spread",
    "open_close_spread",

    "ema20_ema50_ratio",
    "ema50_ema200_ratio",

    "rsi",

    "macd",
    "macd_signal",
    "macd_hist",

    "atr",

    "volatility_24h",
    "volatility_168h",

    "volume_ratio",
    "volume_change",

    "bb_width",
    "bb_position",

    "price_vs_ema50",
    "price_vs_ema200"
]


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, df, initial_balance, trading_fee):
    env = TradingEnv(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        initial_balance=initial_balance,
        trading_fee=trading_fee
    )

    obs, _ = env.reset()

    portfolio_values = [initial_balance]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

    return portfolio_values, info["trade_count"]


def main():
    parser = argparse.ArgumentParser(description="Train a single-agent PPO trading baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--data-dir", default=project_path("data/processed"))
    parser.add_argument("--models-dir", default=project_path("models"))
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--trading-fee", type=float, default=0.001)
    args = parser.parse_args()

    set_global_seed(args.seed)

    train_df = pd.read_parquet(f"{args.data_dir}/train.parquet")
    val_df = pd.read_parquet(f"{args.data_dir}/val.parquet")

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))

    env = TradingEnv(
        df=train_df,
        feature_columns=FEATURE_COLUMNS,
        initial_balance=args.initial_balance,
        trading_fee=args.trading_fee
    )
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        tensorboard_log=project_path("src/tensorboard_logs")
    )

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True
    )

    model_path = f"{args.models_dir}/ppo_btc_seed{args.seed}"
    model.save(model_path)
    print(f"\nModel saved to {model_path}.zip")

    joblib.dump(FEATURE_COLUMNS, f"{args.models_dir}/feature_columns.pkl")
    print("Feature list saved")

    # ==========================================
    # HELD-OUT EVALUATION (val split, not train)
    # ==========================================

    portfolio_values, trade_count = evaluate(
        model, val_df, args.initial_balance, args.trading_fee
    )

    metrics = summarize(portfolio_values)

    print("\n==========================")
    print(f"Validation Evaluation (seed={args.seed})")
    print("==========================")
    print(f"Final Portfolio: ${portfolio_values[-1]:,.2f}")
    print(f"Total Trades: {trade_count}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    portfolio_df = pd.DataFrame({"portfolio_value": portfolio_values})
    curve_path = f"{args.models_dir}/val_portfolio_curve_seed{args.seed}.csv"
    portfolio_df.to_csv(curve_path, index=False)
    print(f"Portfolio curve saved to {curve_path}")


if __name__ == "__main__":
    main()
