"""Baseline: buy and hold (exposure 1 on every test bar).

GOAL MODE nodes replace positions(). Contract: `df` holds only bars up to the
fold's test end (features are causal); fit anything on df.iloc[train_start:train_end]
with labels purged; return one exposure in [0, 1] per test bar.
"""

import numpy as np


def positions(df, fold):
    return np.ones(fold["test_end"] - fold["test_start"])
