"""Relative Strength Index feature engineering."""

import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder-style RSI from a close-price series."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + relative_strength))


def add_rsi_features(frame: pd.DataFrame, close_column: str = "close", period: int = 14) -> pd.DataFrame:
    """Return a copy of `frame` with RSI and simple threshold flags."""
    enriched = frame.copy()
    enriched[f"rsi_{period}"] = compute_rsi(enriched[close_column], period=period)
    enriched[f"rsi_{period}_oversold"] = enriched[f"rsi_{period}"] < 30
    enriched[f"rsi_{period}_overbought"] = enriched[f"rsi_{period}"] > 70
    return enriched
