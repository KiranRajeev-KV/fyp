import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """Single-asset trading environment over real historical bars.

    Costs: a proportional trading fee plus symmetric slippage — buys fill
    above close, sells fill below close — so reward is net of realistic
    execution costs from the start.

    Segments: if `segments` is given (list of (start, end) iloc ranges into
    df), each episode plays one segment; which one is chosen on reset —
    either uniformly at random or forced via reset(options={"segment": i}).
    This is the hook the environment population uses to control what the
    trader must trade on. Without `segments`, episodes start at a random
    offset and run for max_episode_steps (legacy single-agent behavior).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        feature_columns,
        initial_balance=10000,
        trading_fee=0.001,
        slippage=0.0005,
        segments=None,
        max_episode_steps=5000,
        trade_penalty=0.0,
    ):

        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.segments = segments
        self.max_episode_steps = max_episode_steps
        # extra reward penalty applied whenever a buy/sell actually executes —
        # discourages high-churn policies that bleed capital to fees/slippage
        # (the observed dominant failure mode). 0.0 = disabled (original behavior).
        self.trade_penalty = trade_penalty

        self._close = self.df["close"].astype(float).values
        self._features = self.df[feature_columns].values.astype(np.float32)

        self.action_space = spaces.Discrete(3)

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

        if self.segments is not None:
            if options and "segment" in options:
                idx = options["segment"]
            else:
                idx = int(self.np_random.integers(0, len(self.segments)))
            self._episode_start, self._episode_end = self.segments[idx]
        else:
            max_start = max(len(self.df) - self.max_episode_steps, 1)
            self._episode_start = int(self.np_random.integers(0, max_start))
            self._episode_end = min(
                self._episode_start + self.max_episode_steps,
                len(self.df)
            )

        self.current_step = self._episode_start

        self.cash = self.initial_balance
        self.btc_held = 0.0
        self.position = 0
        self.portfolio_value = self.initial_balance
        self.trade_count = 0

        return self._get_observation(), {}

    def _get_observation(self):

        features = self._features[self.current_step]

        btc_value = self.btc_held * self._close[self.current_step]
        portfolio = max(self.cash + btc_value, 1e-8)

        obs = np.concatenate([
            features,
            np.array([
                self.position,
                self.cash / portfolio,
                btc_value / portfolio
            ], dtype=np.float32)
        ])

        return obs

    def step(self, action):

        current_price = self._close[self.current_step]

        old_portfolio_value = (
            self.cash +
            self.btc_held * current_price
        )

        traded = False

        # -----------------------------
        # BUY (fills above close: slippage against the trader)
        # -----------------------------

        if action == 1 and self.cash > 0:

            fill_price = current_price * (1 + self.slippage)

            amount_to_spend = self.cash * (1 - self.trading_fee)

            self.btc_held += amount_to_spend / fill_price
            self.cash = 0
            self.position = 1
            self.trade_count += 1
            traded = True

        # -----------------------------
        # SELL (fills below close)
        # -----------------------------

        elif action == 2 and self.btc_held > 0:

            fill_price = current_price * (1 - self.slippage)

            proceeds = (
                self.btc_held * fill_price * (1 - self.trading_fee)
            )

            self.cash += proceeds
            self.btc_held = 0
            self.position = 0
            self.trade_count += 1
            traded = True

        # -----------------------------
        # Next Step
        # -----------------------------

        self.current_step += 1

        done = self.current_step >= self._episode_end - 1

        next_price = self._close[self.current_step]

        new_portfolio_value = (
            self.cash +
            self.btc_held * next_price
        )

        self.portfolio_value = new_portfolio_value

        reward = np.log(
            new_portfolio_value /
            max(old_portfolio_value, 1e-8)
        )

        # discourage churn: subtract a fixed penalty on any executed trade
        if traded:
            reward -= self.trade_penalty

        info = {
            "portfolio_value": new_portfolio_value,
            "cash": self.cash,
            "btc_held": self.btc_held,
            "trade_count": self.trade_count,
        }

        return self._get_observation(), reward, done, False, info

    def render(self):

        print(
            f"Step: {self.current_step} | "
            f"Portfolio: "
            f"${self.portfolio_value:.2f} | "
            f"Cash: ${self.cash:.2f} | "
            f"BTC: {self.btc_held:.6f}"
        )
