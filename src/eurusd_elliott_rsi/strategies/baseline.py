"""Baseline daily EUR/USD strategy combining Elliott-wave state and RSI.

The strategy emits desired positions, not same-bar trades. Raw signals are
formed from features available at a daily close, then shifted forward one bar so
backtests can only act on information from the previous completed candle.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from eurusd_elliott_rsi.features.rsi import calculate_rsi


@dataclass(frozen=True)
class BaselineStrategyConfig:
    """Configuration for the baseline daily EUR/USD strategy."""

    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    allow_long: bool = True
    allow_short: bool = True
    flat_when_no_signal: bool = True
    close_col: str = "close"
    high_col: str = "high"
    low_col: str = "low"
    trend_col: str = "elliott_trend"
    wave_state_col: str = "wave_state"
    bullish_trend_values: tuple[str, ...] = ("bull", "bullish", "up", "impulse_up", "wave_3_up", "wave_5_up")
    bearish_trend_values: tuple[str, ...] = ("bear", "bearish", "down", "impulse_down", "wave_3_down", "wave_5_down")
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    atr_period: int = 14
    use_atr_position_sizing: bool = False
    atr_risk_multiple: float = 2.0
    risk_per_trade: float = 0.01
    max_position_size: float = 1.0


def _normalise_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _state_filter(data: pd.DataFrame, config: BaselineStrategyConfig) -> tuple[pd.Series, pd.Series]:
    """Build bullish/bearish filters from Elliott-derived feature columns."""
    bullish = pd.Series(False, index=data.index)
    bearish = pd.Series(False, index=data.index)

    bullish_values = {value.lower() for value in config.bullish_trend_values}
    bearish_values = {value.lower() for value in config.bearish_trend_values}

    for column in (config.trend_col, config.wave_state_col):
        if column not in data.columns:
            continue
        values = _normalise_text(data[column])
        bullish |= values.isin(bullish_values)
        bearish |= values.isin(bearish_values)

        numeric_values = pd.to_numeric(data[column], errors="coerce")
        bullish |= numeric_values > 0
        bearish |= numeric_values < 0

    return bullish, bearish


def calculate_atr(
    data: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Calculate Average True Range using Wilder-style smoothing."""
    for column in (high_col, low_col, close_col):
        if column not in data.columns:
            raise KeyError(f"missing ATR input column: {column!r}")
    if period < 1:
        raise ValueError("period must be a positive integer")

    high = data[high_col].astype(float)
    low = data[low_col].astype(float)
    previous_close = data[close_col].astype(float).shift(1)

    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().rename(f"atr_{period}")


def generate_baseline_signals(
    data: pd.DataFrame,
    config: BaselineStrategyConfig | None = None,
) -> pd.DataFrame:
    """Generate shifted daily EUR/USD baseline strategy signals.

    Required input columns are ``close`` plus at least one Elliott-derived state
    column (by default ``elliott_trend`` or ``wave_state``). ``high`` and ``low``
    are required when ATR sizing is enabled or when ATR output is desired.

    Returns a copy of the input with raw and tradable signal columns:
    ``raw_signal`` is the same-close decision, while ``signal`` and all risk
    management columns are shifted by one bar for backtest safety.
    """
    config = config or BaselineStrategyConfig()
    if config.close_col not in data.columns:
        raise KeyError(f"missing close column: {config.close_col!r}")

    result = data.copy()
    close = result[config.close_col].astype(float)
    rsi_col = f"rsi_{config.rsi_period}"
    result[rsi_col] = calculate_rsi(close, period=config.rsi_period)

    bullish, bearish = _state_filter(result, config)
    if not bullish.any() and not bearish.any():
        raise ValueError(
            "no Elliott-wave trend/state values matched; provide columns named "
            f"{config.trend_col!r} or {config.wave_state_col!r}, or configure accepted values"
        )

    raw_signal = pd.Series(0, index=result.index, dtype="int64")
    if config.allow_long:
        raw_signal = raw_signal.mask(bullish & (result[rsi_col] <= config.rsi_oversold), 1)
    if config.allow_short:
        raw_signal = raw_signal.mask(bearish & (result[rsi_col] >= config.rsi_overbought), -1)
    if not config.flat_when_no_signal:
        raw_signal = raw_signal.replace(0, pd.NA).ffill().fillna(0).astype("int64")

    result["raw_signal"] = raw_signal
    result["signal"] = raw_signal.shift(1).fillna(0).astype("int64")

    result["entry_price"] = close.shift(1)
    result["stop_loss"] = pd.NA
    result["take_profit"] = pd.NA

    long_rows = result["signal"] == 1
    short_rows = result["signal"] == -1
    result.loc[long_rows, "stop_loss"] = result.loc[long_rows, "entry_price"] * (1 - config.stop_loss_pct)
    result.loc[long_rows, "take_profit"] = result.loc[long_rows, "entry_price"] * (1 + config.take_profit_pct)
    result.loc[short_rows, "stop_loss"] = result.loc[short_rows, "entry_price"] * (1 + config.stop_loss_pct)
    result.loc[short_rows, "take_profit"] = result.loc[short_rows, "entry_price"] * (1 - config.take_profit_pct)

    result["position_size"] = result["signal"].abs().astype(float)
    if config.use_atr_position_sizing:
        atr_col = f"atr_{config.atr_period}"
        result[atr_col] = calculate_atr(
            result,
            period=config.atr_period,
            high_col=config.high_col,
            low_col=config.low_col,
            close_col=config.close_col,
        )
        atr_risk = result[atr_col].shift(1) * config.atr_risk_multiple
        size = (config.risk_per_trade / (atr_risk / result["entry_price"])).clip(upper=config.max_position_size)
        result["position_size"] = size.where(result["signal"] != 0, 0.0).fillna(0.0)

    return result


__all__ = ["BaselineStrategyConfig", "calculate_atr", "generate_baseline_signals"]
