# Bitcoin Trading Agent using PPO

This project implements a custom algorithmic trading environment using OpenAI's `gymnasium` and trains a Reinforcement Learning agent to trade Bitcoin using Proximal Policy Optimization (PPO) via `stable-baselines3`.

## The Project Approach

The project aims to create a profitable trading strategy by providing an RL agent with historical Bitcoin data combined with various technical indicators. The agent learns when to buy, sell, or hold its position to maximize the portfolio's value over time. 

The project is split into an exploratory phase (Jupyter Notebooks) and a script-based execution phase.

## Dependencies

The core libraries used in this project are:
- `pandas` - For data manipulation and feature engineering.
- `numpy` - For numerical operations.
- `gymnasium` - For creating the custom RL trading environment.
- `stable-baselines3` - For the PPO algorithm implementation.
- `joblib` - For saving and loading the feature columns.
- `tensorboard` - For monitoring training progress.

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
│   ├── trading_env.ipynb          # Designing the Gymnasium environment
│   ├── train_ppo.ipynb            # Notebook for training the PPO agent
│   └── train_test.ipynb           # Model evaluation and testing
├── models/                # Saved trained models and metadata
│   ├── feature_columns.pkl        # Saved list of features
│   ├── ppo_btc.zip                # The trained PPO agent
│   └── train_portfolio_curve.csv  # Portfolio values logged during training evaluation
├── src/                   # Executable Python scripts
│   ├── trading_env.py     # Custom Gymnasium environment implementation
│   └── train_ppo.py       # PPO training script
└── README.md              # This file
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

Training is executed using the `train_ppo.py` script. The script applies the Proximal Policy Optimization (PPO) algorithm for 500,000 timesteps against the `.parquet` train data. 

To run the training:
```bash
cd src
python train_ppo.py
```

Features and generated models are saved off seamlessly into the `models/` directory for later reference and evaluation. During training, live telemetry is recorded and can be monitored via TensorBoard logs configured in `src/tensorboard_logs/`.
