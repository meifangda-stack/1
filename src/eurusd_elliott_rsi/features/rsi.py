"""Relative Strength Index (RSI) feature helpers."""

from __future__ import annotations

import pandas as pd


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder's Relative Strength Index.

    Parameters
    ----------
    close:
        Closing prices ordered from oldest to newest.
    period:
        RSI lookback period. Defaults to the standard 14 bars.

    Returns
    -------
    pandas.Series
        RSI values in the range 0-100. The first ``period`` rows are ``NaN``
        because there is not enough completed history to form the indicator.
    """
    if period < 1:
        raise ValueError("period must be a positive integer")

    close = pd.Series(close, copy=False).astype(float)
    delta = close.diff()

    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    # Wilder's smoothing is an EMA with alpha=1/period. ``min_periods`` avoids
    # producing values until the calculation has a full completed window.
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))

    # Handle one-sided markets explicitly: no losses means RSI=100, while no
    # gains and no losses means a neutral 50 after the warm-up period.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)

    return rsi.rename(f"rsi_{period}")


def add_rsi(
    data: pd.DataFrame,
    close_col: str = "close",
    period: int = 14,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``data`` with an RSI feature column added."""
    if close_col not in data.columns:
        raise KeyError(f"missing close column: {close_col!r}")

    result = data.copy()
    name = output_col or f"rsi_{period}"
    result[name] = calculate_rsi(result[close_col], period=period)
    return result
