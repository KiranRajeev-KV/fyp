import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        feature_columns,
        initial_balance=10000,
        trading_fee=0.001
    ):

        super().__init__()

        self.df = df.reset_index(drop=True)

        self.feature_columns = feature_columns

        self.initial_balance = initial_balance

        self.trading_fee = trading_fee

        # ----------------------------------
        # Actions
        # ----------------------------------

        self.action_space = spaces.Discrete(3)

        # ----------------------------------
        # Observation Space
        # ----------------------------------

        obs_size = len(feature_columns) + 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = np.random.randint(
            0,
            len(self.df) - 5000
        )

        self.cash = self.initial_balance

        self.btc_held = 0.0

        self.position = 0

        self.portfolio_value = self.initial_balance

        self.trade_count = 0

        observation = self._get_observation()

        return observation, {}

    def _get_observation(self):

        features = self.df.loc[
            self.current_step,
            self.feature_columns
        ].values.astype(np.float32)

        btc_value = (
            self.btc_held *
            self.df.loc[self.current_step, "close"]
        )

        portfolio = max(
            self.cash + btc_value,
            1e-8
        )

        cash_ratio = self.cash / portfolio

        asset_ratio = btc_value / portfolio

        obs = np.concatenate([
            features,
            np.array([
                self.position,
                cash_ratio,
                asset_ratio
            ], dtype=np.float32)
        ])

        return obs

    def step(self, action):

        current_price = self.df.loc[
            self.current_step,
            "close"
        ]

        old_portfolio_value = (
            self.cash +
            self.btc_held * current_price
        )

        # -----------------------------
        # BUY
        # -----------------------------

        if action == 1:

            if self.cash > 0:

                amount_to_spend = self.cash

                fee = (
                    amount_to_spend *
                    self.trading_fee
                )

                amount_to_spend -= fee

                btc_bought = (
                    amount_to_spend /
                    current_price
                )

                self.btc_held += btc_bought

                self.cash = 0

                self.position = 1

                self.trade_count += 1

        # -----------------------------
        # SELL
        # -----------------------------

        elif action == 2:

            if self.btc_held > 0:

                proceeds = (
                    self.btc_held *
                    current_price
                )

                fee = (
                    proceeds *
                    self.trading_fee
                )

                proceeds -= fee

                self.cash += proceeds

                self.btc_held = 0

                self.position = 0

                self.trade_count += 1

        # -----------------------------
        # Next Step
        # -----------------------------

        self.current_step += 1

        done = (
            self.current_step >=
            len(self.df) - 1
        )

        next_price = self.df.loc[
            self.current_step,
            "close"
        ]

        new_portfolio_value = (
            self.cash +
            self.btc_held * next_price
        )

        self.portfolio_value = (
            new_portfolio_value
        )

        # -----------------------------
        # Reward
        # -----------------------------

        reward = np.log(
            new_portfolio_value /
            old_portfolio_value
        )

        observation = (
            self._get_observation()
        )

        info = {
            "portfolio_value":
            new_portfolio_value,

            "cash":
            self.cash,

            "btc_held":
            self.btc_held,

            "trade_count":
            self.trade_count
        }

        return (
            observation,
            reward,
            done,
            False,
            info
        )

    def render(self):

        print(
            f"Step: {self.current_step} | "
            f"Portfolio: "
            f"${self.portfolio_value:.2f} | "
            f"Cash: ${self.cash:.2f} | "
            f"BTC: {self.btc_held:.6f}"
        )
