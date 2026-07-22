"""Regime labeling: classify each bar as bull / bear / sideways.

A bar's regime is decided by the trailing net log-return over `window` bars:
above +threshold -> bull, below -threshold -> bear, otherwise sideways.
Labels are trailing-only (no lookahead) so they are safe to use as features
or for splitting evaluation data by regime.
"""

import numpy as np
import pandas as pd

BULL, BEAR, SIDEWAYS = "bull", "bear", "sideways"


def label_regimes(df, window=720, threshold=0.10, price_col="close"):
    """Return a pd.Series of regime labels aligned with df.

    window: trailing bars used to measure the move (720 = ~30 days of 1h bars).
    threshold: net log-return magnitude separating trending from sideways
        (0.10 ~= a 10.5% move over the window).
    The first `window` bars have insufficient history and are labeled sideways.
    """
    log_price = np.log(df[price_col].astype(float))
    trailing_move = log_price - log_price.shift(window)

    labels = pd.Series(SIDEWAYS, index=df.index, dtype=object)
    labels[trailing_move > threshold] = BULL
    labels[trailing_move < -threshold] = BEAR
    return labels


def add_regime_column(df, window=720, threshold=0.10, price_col="close"):
    out = df.copy()
    out["regime"] = label_regimes(out, window, threshold, price_col)
    return out


def regime_counts(df):
    return df["regime"].value_counts().to_dict()
