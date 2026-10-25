"""Ordered base features shared by preprocessing, training, and evaluation."""

FEATURE_COLUMNS = [
    "return", "log_return", "high_low_spread", "open_close_spread",
    "ema20_ema50_ratio", "ema50_ema200_ratio", "rsi", "macd", "macd_signal",
    "macd_hist", "atr", "volatility_24h", "volatility_168h", "volume_ratio",
    "volume_change", "bb_width", "bb_position", "price_vs_ema50", "price_vs_ema200",
]
