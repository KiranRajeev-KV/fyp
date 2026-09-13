# Bitcoin Trading Agent using PPO

This project implements a custom algorithmic trading environment using OpenAI's `gymnasium` and trains a Reinforcement Learning agent to trade Bitcoin using Proximal Policy Optimization (PPO) via `stable-baselines3`.

## The Project Approach

The project aims to create a profitable trading strategy by providing an RL agent with historical Bitcoin data combined with various technical indicators. The agent learns when to buy, sell, or hold its position to maximize the portfolio's value over time. 

The project is split into an exploratory phase (Jupyter Notebooks) and a script-based execution phase.

> **Note:** the single-agent PPO here is the *control* for the project's
> real contribution — the competitive coevolutionary engine in `src/coevo/`
> (trader population evolved via PBT vs. an adversarial environment
> population that selects real historical segments). See `../context.md` for
> the design and `../docs/design-history.md` for how the architecture got here.

## Dependencies

Install with `pip install -r requirements.txt`. The core libraries used:
- `pandas` - For data manipulation and feature engineering.
- `numpy` - For numerical operations.
- `gymnasium` - For creating the custom RL trading environment.
- `stable-baselines3` (and its `torch` dependency) - For the PPO algorithm implementation.
- `joblib` - For saving and loading the feature columns and scaler.
- `tensorboard` - For monitoring training progress.
- `scikit-learn` - For feature scaling (`StandardScaler`).
- `ta` - For technical indicator computation.
- `openpyxl` - For the Excel export in `feature_engineering.ipynb`.

## Project Structure

```text
├── data/                  # Data pipeline files
│   ├── raw/               # Raw downloaded bitcoin data
│   ├── cleaned/           # Cleaned data
│   ├── features/          # Data enriched with technical indicators
│   └── processed/         # Final training/testing parquet files
├── jupyter_project/       # Exploratory notebooks
│   ├── data_collect.ipynb         # Web scraping / API data collection
│   ├── feature_engineering.ipynb  # Adding technical indicators to data
│   ├── trading_env.ipynb          # Imports TradingEnv from src/ (kept in sync, not duplicated)
│   └── train_test.ipynb           # Train/val/test split + feature scaling
├── models/                # Saved trained models and metadata
│   ├── feature_columns.pkl        # Saved list of features
│   ├── feature_scaler.pkl         # StandardScaler fit on train only
│   ├── ppo_btc_seed<N>.zip        # Trained PPO agent for seed N
│   └── val_portfolio_curve_seed<N>.csv  # Held-out validation portfolio curve
├── runs/                  # Coevolution run outputs (gitignored): config.json,
│                          #   generations.csv, best_trader.zip per (mode, seed)
├── src/                   # Executable Python scripts
│   ├── trading_env.py     # Custom Gymnasium env (fees + slippage, segment-parameterized)
│   ├── train_ppo.py       # Legacy single-run PPO script (seeded, held-out eval)
│   ├── regimes.py         # Bull/bear/sideways regime labeling (trailing, no lookahead)
│   ├── segments.py        # Segment pool + descriptors + walk-forward folds
│   ├── baselines.py       # Buy-and-hold and MA-crossover baselines
│   ├── metrics.py         # Shared risk-adjusted metrics (Sharpe, Sortino, Calmar, max DD)
│   ├── run_coevolution.py # Entry point: single / population / coevolution, multi-seed
│   ├── engine_cli.py      # The one shared EngineConfig argument surface
│   ├── run_walkforward.py # Rolling-window ablation, shardable per (fold, mode, seed)
│   ├── build_features.py  # Scripted data pipeline (clean -> features -> scaled test_ext)
│   ├── evaluate.py        # Held-out per-regime evaluation + bootstrap significance
│   ├── compare_arms.py    # Arm-vs-arm significance (the test the hypothesis needs)
│   └── coevo/
│       ├── engine.py             # Generation loop, internal vs external ledgers
│       ├── trader_population.py  # PPO traders + PBT (hyperparam crossover/mutation)
│       ├── env_population.py     # Adversarial segment selectors + fitness sharing
│       └── hall_of_fame.py       # Archives for both populations (anti-cycling)
├── tests/
│   └── smoke_test.py      # End-to-end run on synthetic data (no real data needed)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Coevolutionary engine (`src/coevo/`)

The core contribution. Two populations interact through the market simulator
each generation:

- **Trader population** — PPO agents. Evolution touches *hyperparameters
  only* (PBT: exploit copies weights from a stronger agent, hyperparams come
  from crossover of two top parents plus mutation). PPO's gradient does all
  weight learning.
- **Environment population** — segment-selection strategies: preference
  weights over segment descriptors (volatility percentile, trend
  strength/direction, drawdown depth, regime transitions) sampled through a
  softmax. They can only *select* real historical windows, never invent
  prices. Fitness is the inverted trader score, with competitive fitness
  sharing so adversaries are rewarded for being difficult in different ways.

Guardrails built in from day one: halls of fame on both sides (anti-cycling),
fitness sharing (anti diversity-collapse), and a per-generation **external
yardstick** — the best trader is evaluated every generation on fixed held-out
data against fixed baselines, logged separately from internal fitness
(anti Red Queen).

Run the three-way ablation (this is the paper's core experiment):
```bash
cd src
python run_coevolution.py --mode single       --seeds 42 43 44
python run_coevolution.py --mode population   --seeds 42 43 44
python run_coevolution.py --mode coevolution  --seeds 42 43 44
```
All engine knobs are defined once in `engine_cli.py` and shared with
`run_walkforward.py`. The engine refuses to start in a run directory that
already holds a finished run (pass `--overwrite` to replace it). The full
multi-seed, four-arm run is `../run_v5.sh`; the walk-forward is
`../run_walkforward.sh` (sharded per fold x arm x seed, re-launchable).

Then evaluate frozen models per-regime on the untouched test split:
```bash
python evaluate.py --models ../runs/coevolution_seed*/best_trader.zip \
    --data ../data/processed/test.parquet --out ../runs/coevolution_eval.json
```
and test the hypothesis itself -- every arm against every other arm and both
baselines, with segment / seed / hierarchical paired bootstraps, using either
the saved rank-0 trader or the mean of each seed's final population:
```bash
python compare_arms.py --data ../data/processed/test.parquet \
    --arms coevolution=../runs/v5/coevolution_seed*/best_trader.zip \
           population=../runs/v5/population_seed*/best_trader.zip \
           single=../runs/v5/single_seed*/best_trader.zip \
    --final-traders --out ../runs/v5/compare_test.json
```
To extend the held-out window when Binance publishes new months, bump `END`
in `../../download_klines.py`, run it, then `python build_features.py`
(reproduces the notebooks bit-for-bit and writes only `*_ext` files).

## Prospective validation (stage 5)

Freeze a model (note the freeze date), keep collecting new Binance data with
`data_collect.ipynb` + `feature_engineering.ipynb`, and after 4–8 weeks run
`evaluate.py` pointing `--data` at a parquet containing **only bars after the
freeze date**, scaled with the already-saved `feature_scaler.pkl`. Because
that data did not exist at freeze time, overfitting to it is impossible.
Start this around month 4–5 so the window elapses while the report is being
written — it runs in wall-clock time.

## Smoke test

`tests/smoke_test.py` runs the entire machinery (regimes, segment pool,
env with slippage, baselines, all three engine modes, halls of fame) on
synthetic data in a few minutes, no real data required:
```bash
cd tests && python smoke_test.py
```

## Data Pipeline

The data pipeline progresses through four distinct stages:
1. **Raw (`data/raw/`)**: Initial raw Bitcoin pricing data.
2. **Cleaned (`data/cleaned/`)**: Data with missing values imputed and structured correctly.
3. **Features (`data/features/`)**: The step where technical indicators (like RSI, MACD, Bollinger Bands, EMAs, and ATR) are calculated and appended.
4. **Processed (`data/processed/`)**: The final structured dataset (using `.parquet`) used directly by the training environment.

## The Trading Environment (`src/trading_env.py`)

A custom `gymnasium.Env` environment has been built with the following characteristics:

### 1. Observation Space
The agent observes the environment at each time step. The observation is an array consisting of:
- **19 Technical Indicators**: Includes log returns, EMAs, RSI, MACD, ATR, volatility, volume ratios, and Bollinger Band metrics.
- **3 Account metrics**: The current position type (0 or 1), the ratio of cash in the portfolio, and the ratio of asset value in the portfolio.

### 2. Action Space
The agent uses a discrete action space of size 3:
- **0: Hold** - Do nothing.
- **1: Buy** - Uses all available cash to buy Bitcoin (minus trading fees).
- **2: Sell** - Sells all held Bitcoin for cash (minus trading fees).

### 3. Reward Function
The reward for each step is the **log return** of the total portfolio value between the previous step and the current step:
`reward = log(new_portfolio_value / old_portfolio_value)`
This logically incentivizes the agent to exponentially grow the overall portfolio value over time.

## Training (`src/train_ppo.py`)

Training is executed using the `train_ppo.py` script. The script applies the Proximal Policy Optimization (PPO) algorithm for 500,000 timesteps (configurable) against the `.parquet` train data, then evaluates the trained policy on the **held-out validation split** (`val.parquet`), not the training data itself.

To run the training:
```bash
cd src
python train_ppo.py --seed 42 --timesteps 500000
```

Runs are fully seeded (`--seed`, default 42) — `numpy`, `torch`, and the PPO model itself all take the same seed, and `TradingEnv` uses gymnasium's seeded `self.np_random` rather than global numpy state, so a given seed reproduces the same episode start points and training run. This makes multi-seed sweeps (`for s in 42 43 44 45; do python train_ppo.py --seed $s; done`) safe to run in parallel without clobbering each other's model/curve outputs.

Features and generated models are saved into the `models/` directory, namespaced by seed. During training, live telemetry is recorded and can be monitored via TensorBoard logs configured in `src/tensorboard_logs/`.

## Baselines and metrics (`src/baselines.py`, `src/metrics.py`)

Before trusting any PPO result, compare it against real baselines computed on the same held-out split and cost model:
```bash
cd src
python baselines.py --data ../data/processed/val.parquet
```
This reports buy-and-hold and a 20/50 MA-crossover strategy, using the same `metrics.summarize()` (total return, Sharpe, Sortino, max drawdown, Calmar) that `train_ppo.py` uses for the PPO agent — so all three are directly comparable.
