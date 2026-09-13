"""Segment pool and walk-forward splitting.

The environment population never invents prices — it can only *select* real
historical windows (segments) from a pool built here. Each segment carries a
descriptor vector (volatility percentile, trend strength, drawdown depth,
regime-transition presence) that environment individuals express preferences
over.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metrics import max_drawdown
from regimes import label_regimes

DESCRIPTOR_NAMES = [
    "volatility_pct",       # percentile rank of realized volatility
    "trend_strength",       # |net log return| / realized volatility
    "trend_direction",      # sign of net log return, in [-1, 1]
    "drawdown_depth",       # worst peak-to-trough inside the segment (positive)
    "regime_transitions",   # number of regime label changes inside the segment
]


@dataclass
class Segment:
    start: int              # iloc start into the source df
    end: int                # iloc end (exclusive)
    regime: str             # majority of the per-bar TRAILING labels inside the segment
    descriptor: np.ndarray  # aligned with DESCRIPTOR_NAMES
    # regime by the segment's OWN net log move over [start, end) with the same
    # threshold. The trailing-label majority lags by up to a window, so a
    # segment that rallies +15% can still be "sideways" by majority vote;
    # report per-regime results under both definitions.
    move_regime: str = "sideways"


@dataclass
class SegmentPool:
    df: pd.DataFrame
    segments: list = field(default_factory=list)

    def __len__(self):
        return len(self.segments)

    @property
    def descriptor_matrix(self):
        """(n_segments, n_descriptors) matrix, z-scored per column."""
        raw = np.stack([s.descriptor for s in self.segments])
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std == 0] = 1.0
        return (raw - mean) / std

    def slice(self, segment):
        return self.df.iloc[segment.start:segment.end].reset_index(drop=True)

    def regimes(self, by="regime"):
        return pd.Series([getattr(s, by) for s in self.segments]).value_counts().to_dict()


def move_regime_label(net_log_move, threshold=0.10):
    if net_log_move > threshold:
        return "bull"
    if net_log_move < -threshold:
        return "bear"
    return "sideways"


def build_segment_pool(
    df,
    segment_length=720,
    stride=168,
    regime_window=720,
    regime_threshold=0.10,
):
    """Cut df into overlapping fixed-length segments with descriptors.

    Defaults for 1h bars: 720-bar (~30 day) segments stepped weekly (168 bars).
    """
    df = df.reset_index(drop=True)
    regimes = label_regimes(df, window=regime_window, threshold=regime_threshold)
    close = df["close"].astype(float).values
    log_close = np.log(close)

    # per-bar log returns for volatility
    log_ret = np.diff(log_close, prepend=log_close[0])

    raw_vols = []
    proto = []
    for start in range(0, len(df) - segment_length + 1, stride):
        end = start + segment_length
        seg_log_ret = log_ret[start + 1:end]
        vol = seg_log_ret.std()
        net_move = log_close[end - 1] - log_close[start]
        # total volatility over the segment; guards div-by-zero on flat data
        total_vol = vol * np.sqrt(segment_length) or 1e-12

        seg_regimes = regimes.iloc[start:end]
        transitions = int((seg_regimes != seg_regimes.shift()).sum() - 1)
        majority = seg_regimes.mode().iloc[0]

        dd = abs(max_drawdown(close[start:end]))

        raw_vols.append(vol)
        proto.append((start, end, majority, net_move, total_vol, dd, transitions))

    vol_ranks = pd.Series(raw_vols).rank(pct=True).values

    segments = []
    for i, (start, end, majority, net_move, total_vol, dd, transitions) in enumerate(proto):
        descriptor = np.array([
            vol_ranks[i],
            abs(net_move) / total_vol,
            np.sign(net_move),
            dd,
            transitions,
        ], dtype=np.float64)
        segments.append(Segment(start=start, end=end, regime=majority, descriptor=descriptor,
                                move_regime=move_regime_label(net_move, regime_threshold)))

    return SegmentPool(df=df, segments=segments)


@dataclass
class Fold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_folds(n_rows, train_span, test_span, step=None):
    """Rolling walk-forward folds over a df of n_rows.

    Each fold trains on [train_start, train_end) and tests on the immediately
    following [test_start, test_end). step defaults to test_span so test
    windows tile the data without overlap.
    """
    step = step or test_span
    folds = []
    start = 0
    while start + train_span + test_span <= n_rows:
        folds.append(Fold(
            train_start=start,
            train_end=start + train_span,
            test_start=start + train_span,
            test_end=start + train_span + test_span,
        ))
        start += step
    return folds
