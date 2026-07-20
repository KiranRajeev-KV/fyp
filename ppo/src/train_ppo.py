import pandas as pd
import numpy as np
import joblib

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ==========================================
# IMPORT YOUR ENVIRONMENT
# ==========================================

from trading_env import TradingEnv

# ==========================================
# LOAD TRAIN DATA
# ==========================================

train_df = pd.read_parquet(
    "../data/processed/train.parquet"
)

print("Train rows:", len(train_df))

# ==========================================
# FEATURE LIST
# ==========================================

feature_columns = [
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

# ==========================================
# CREATE ENVIRONMENT
# ==========================================

env = TradingEnv(
    df=train_df,
    feature_columns=feature_columns,
    initial_balance=10000,
    trading_fee=0.001
)

env = Monitor(env)

# ==========================================
# PPO MODEL
# ==========================================

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

    tensorboard_log="./tensorboard_logs/"
)

# ==========================================
# TRAIN
# ==========================================

TOTAL_TIMESTEPS = 500_000

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    "../models/ppo_btc"
)

print("\nModel Saved")

# ==========================================
# SAVE FEATURES
# ==========================================

joblib.dump(
    feature_columns,
    "../models/feature_columns.pkl"
)

print("Feature List Saved")

# ==========================================
# QUICK TRAINING EVALUATION
# ==========================================

obs, _ = env.reset()

portfolio_values = []

actions_taken = []

done = False

while not done:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, done, truncated, info = env.step(action)

    actions_taken.append(action)

    portfolio_values.append(
        info["portfolio_value"]
    )

# ==========================================
# RESULTS
# ==========================================

final_portfolio = portfolio_values[-1]

print("\n==========================")
print("Training Evaluation")
print("==========================")

print(
    f"Final Portfolio: "
    f"${final_portfolio:,.2f}"
)

print(
    f"Total Trades: "
    f"{info['trade_count']}"
)

print(
    f"Return: "
    f"{(final_portfolio/10000-1)*100:.2f}%"
)

# ==========================================
# SAVE PORTFOLIO CURVE
# ==========================================

portfolio_df = pd.DataFrame({
    "portfolio_value":
    portfolio_values
})

portfolio_df.to_csv(
    "../models/train_portfolio_curve.csv",
    index=False
)

print(
    "Portfolio Curve Saved"
)
