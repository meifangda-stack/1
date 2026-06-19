"""Baseline strategy that combines RSI levels with swing direction."""

import pandas as pd

from eurusd_elliott_rsi.features.elliott import add_swing_features
from eurusd_elliott_rsi.features.rsi import add_rsi_features


def generate_signals(frame: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    """Generate long/short/flat signals for a baseline EUR/USD strategy."""
    features = add_swing_features(add_rsi_features(frame, period=rsi_period))
    rsi = features[f"rsi_{rsi_period}"]
    signals = features.copy()
    signals["signal"] = 0
    signals.loc[(rsi < 30) & (signals["wave_direction"] >= 0), "signal"] = 1
    signals.loc[(rsi > 70) & (signals["wave_direction"] <= 0), "signal"] = -1
    return signals
