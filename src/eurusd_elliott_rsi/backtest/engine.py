"""Small vectorized backtesting engine for signal research."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Container for vectorized strategy performance."""

    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    sharpe: float


def run_vectorized_backtest(
    frame: pd.DataFrame,
    price_column: str = "close",
    signal_column: str = "signal",
    periods_per_year: int = 252 * 24,
) -> BacktestResult:
    """Run a close-to-close vectorized backtest using lagged trading signals."""
    returns = frame[price_column].pct_change().fillna(0)
    strategy_returns = returns * frame[signal_column].shift(1).fillna(0)
    equity_curve = (1 + strategy_returns).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1) if not equity_curve.empty else 0.0
    volatility = strategy_returns.std()
    sharpe = 0.0 if volatility == 0 else float(strategy_returns.mean() / volatility * periods_per_year**0.5)
    return BacktestResult(equity_curve, strategy_returns, total_return, sharpe)
