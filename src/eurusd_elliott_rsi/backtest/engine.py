"""Backtesting utilities for daily EUR/USD trading strategies.

The module is intentionally dependency-light (``pandas`` and ``numpy`` only)
and focuses on transparent, auditable assumptions for daily FX backtests:
signals are traded with a one-bar delay, execution can include spread and
slippage, positions can be long/short/flat, and stop-loss / take-profit exits
are evaluated from daily OHLC data when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SignalLike = pd.Series | Sequence[float] | np.ndarray
ModelFactory = Callable[[], Any]
Trainer = Callable[[Any, pd.DataFrame], Any]
Predictor = Callable[[Any, pd.DataFrame], SignalLike]

TRADING_DAYS_PER_YEAR = 252
_LONG = 1
_FLAT = 0
_SHORT = -1


@dataclass(frozen=True)
class BacktestConfig:
    """Execution and risk assumptions used by :class:`BacktestEngine`.

    All pip-based inputs are converted to decimal EUR/USD price units using
    ``pip_size``. A one-pip spread is therefore ``0.0001`` by default.
    ``position_size`` represents notional EUR exposure per full signal, while
    ``initial_capital`` is denominated in USD.
    """

    initial_capital: float = 10_000.0
    position_size: float = 10_000.0
    spread_pips: float = 0.8
    slippage_pips: float = 0.1
    stop_loss_pips: float | None = None
    take_profit_pips: float | None = None
    pip_size: float = 0.0001
    annualization_factor: int = TRADING_DAYS_PER_YEAR

    @property
    def spread(self) -> float:
        return self.spread_pips * self.pip_size

    @property
    def half_spread(self) -> float:
        return self.spread / 2.0

    @property
    def slippage(self) -> float:
        return self.slippage_pips * self.pip_size

    @property
    def stop_loss(self) -> float | None:
        return None if self.stop_loss_pips is None else self.stop_loss_pips * self.pip_size

    @property
    def take_profit(self) -> float | None:
        return None if self.take_profit_pips is None else self.take_profit_pips * self.pip_size


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single chronological train/test split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class BacktestEngine:
    """Daily EUR/USD backtesting engine.

    Input market data must contain a close column. Open/high/low columns are
    optional; when present, execution occurs at the delayed bar's open and
    intraday stop-loss / take-profit checks use high/low. Without open data the
    engine executes at close, and without high/low stops are evaluated on close.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: BacktestConfig | None = None,
        *,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> None:
        if close_col not in data.columns:
            raise ValueError(f"data must include a '{close_col}' column")
        if len(data) < 2:
            raise ValueError("data must contain at least two rows")

        self.data = data.copy().sort_index()
        self.config = config or BacktestConfig()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col

    def run(self, signals: SignalLike) -> tuple[pd.DataFrame, dict[str, float]]:
        """Run a backtest and return the daily equity curve plus metrics.

        Signals should be ``1`` for long, ``-1`` for short, and ``0`` for flat.
        Any positive value is treated as long and any negative value as short.
        The desired signal is shifted by one bar before trading, preventing
        same-bar look-ahead bias.
        """

        desired = self._coerce_signals(signals)
        delayed = desired.shift(1).fillna(_FLAT).astype(int)
        cfg = self.config
        data = self.data
        close = data[self.close_col].astype(float)
        execution_base = data[self.open_col].astype(float) if self.open_col in data else close
        high = data[self.high_col].astype(float) if self.high_col in data else close
        low = data[self.low_col].astype(float) if self.low_col in data else close

        rows: list[dict[str, float | int | pd.Timestamp | str | None]] = []
        cash = float(cfg.initial_capital)
        equity = float(cfg.initial_capital)
        position = _FLAT
        units = 0.0
        entry_price: float | None = None
        entry_equity: float | None = None
        trade_pnls: list[float] = []
        trade_wins = 0
        exposure_days = 0

        for i, (timestamp, target_position) in enumerate(delayed.items()):
            base_price = float(execution_base.iloc[i])
            exit_reason: str | None = None
            realized_pnl = 0.0

            if position != _FLAT:
                exposure_days += 1
                stop_price, take_price = self._exit_levels(position, entry_price, cfg)
                stop_hit = stop_price is not None and (low.iloc[i] <= stop_price if position == _LONG else high.iloc[i] >= stop_price)
                take_hit = take_price is not None and (high.iloc[i] >= take_price if position == _LONG else low.iloc[i] <= take_price)
                if stop_hit or take_hit:
                    # Conservative daily-bar ambiguity handling: if both levels
                    # are touched on the same bar, assume the stop was hit first.
                    exit_level = stop_price if stop_hit else take_price
                    realized_pnl = self._close_position(position, units, entry_price, exit_level, cfg)
                    cash += realized_pnl
                    equity = cash
                    trade_pnls.append(realized_pnl)
                    trade_wins += int(realized_pnl > 0)
                    position = _FLAT
                    units = 0.0
                    entry_price = None
                    entry_equity = None
                    exit_reason = "stop_loss" if stop_hit else "take_profit"

            if position != target_position:
                if position != _FLAT:
                    realized_pnl = self._close_position(position, units, entry_price, base_price, cfg)
                    cash += realized_pnl
                    trade_pnls.append(realized_pnl)
                    trade_wins += int(realized_pnl > 0)
                    exit_reason = "signal"
                position = int(target_position)
                if position != _FLAT:
                    entry_price = self._entry_price(position, base_price, cfg)
                    units = self._position_units(cash, close.iloc[i])
                    entry_equity = cash
                else:
                    entry_price = None
                    entry_equity = None
                    units = 0.0

            unrealized = 0.0 if position == _FLAT else self._mark_to_market(position, units, entry_price, close.iloc[i], cfg)
            equity = cash + unrealized
            rows.append(
                {
                    "date": timestamp,
                    "signal": int(desired.loc[timestamp]),
                    "position": position,
                    "entry_price": entry_price,
                    "cash": cash,
                    "equity": equity,
                    "daily_return": 0.0,
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized,
                    "exit_reason": exit_reason,
                    "entry_equity": entry_equity,
                }
            )

        result = pd.DataFrame(rows).set_index("date")
        result["daily_return"] = result["equity"].pct_change().fillna(0.0)
        metrics = calculate_metrics(result["daily_return"], result["equity"], trade_pnls, exposure_days, cfg.annualization_factor)
        return result, metrics

    def _coerce_signals(self, signals: SignalLike) -> pd.Series:
        series = signals if isinstance(signals, pd.Series) else pd.Series(signals, index=self.data.index)
        series = series.reindex(self.data.index).fillna(_FLAT)
        return pd.Series(np.sign(series).astype(int), index=self.data.index)

    def _position_units(self, equity: float, price: float) -> float:
        return min(self.config.position_size, max(equity, 0.0)) / float(price)

    @staticmethod
    def _entry_price(position: int, base_price: float, cfg: BacktestConfig) -> float:
        return base_price + cfg.half_spread + cfg.slippage if position == _LONG else base_price - cfg.half_spread - cfg.slippage

    @staticmethod
    def _close_position(position: int, units: float, entry_price: float | None, base_price: float, cfg: BacktestConfig) -> float:
        if entry_price is None:
            return 0.0
        exit_price = base_price - cfg.half_spread - cfg.slippage if position == _LONG else base_price + cfg.half_spread + cfg.slippage
        return position * units * (exit_price - entry_price)

    @staticmethod
    def _mark_to_market(position: int, units: float, entry_price: float | None, close_price: float, cfg: BacktestConfig) -> float:
        if entry_price is None:
            return 0.0
        mark_price = close_price - cfg.half_spread if position == _LONG else close_price + cfg.half_spread
        return position * units * (mark_price - entry_price)

    @staticmethod
    def _exit_levels(position: int, entry_price: float | None, cfg: BacktestConfig) -> tuple[float | None, float | None]:
        if entry_price is None:
            return None, None
        stop = cfg.stop_loss
        take = cfg.take_profit
        if position == _LONG:
            return (None if stop is None else entry_price - stop, None if take is None else entry_price + take)
        return (None if stop is None else entry_price + stop, None if take is None else entry_price - take)


def calculate_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trade_pnls: Iterable[float],
    exposure_days: int,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Calculate common performance statistics for a backtest."""

    returns = returns.astype(float).fillna(0.0)
    equity = equity.astype(float)
    pnls = np.asarray(list(trade_pnls), dtype=float)
    periods = max(len(returns), 1)
    ending = float(equity.iloc[-1]) if len(equity) else np.nan
    beginning = float(equity.iloc[0]) if len(equity) else np.nan
    years = periods / annualization_factor

    annualized_return = (ending / beginning) ** (1.0 / years) - 1.0 if beginning > 0 and years > 0 else np.nan
    volatility = float(returns.std(ddof=0) * np.sqrt(annualization_factor))
    sharpe = float((returns.mean() * annualization_factor) / volatility) if volatility > 0 else np.nan
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(annualization_factor)) if len(downside) else 0.0
    sortino = float((returns.mean() * annualization_factor) / downside_vol) if downside_vol > 0 else np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    return {
        "annualized_return": float(annualized_return),
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": float(len(wins) / len(pnls)) if len(pnls) else np.nan,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan),
        "exposure_time": float(exposure_days / periods),
        "total_return": float(ending / beginning - 1.0) if beginning > 0 else np.nan,
        "ending_equity": ending,
        "trade_count": float(len(pnls)),
    }


def walk_forward_evaluate(
    data: pd.DataFrame,
    model_factories: Mapping[str, ModelFactory],
    trainer: Trainer,
    predictor: Predictor,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    config: BacktestConfig | None = None,
    price_columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Train each architecture on past data and evaluate later holdouts.

    ``model_factories`` maps architecture names to zero-argument constructors.
    For every chronological split, a fresh model is created, trained only on the
    rows in the training window, and asked to predict signals for the strictly
    later test window. The returned frame contains one row per architecture and
    split with the holdout backtest metrics.
    """

    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = test_size if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    price_columns = price_columns or {}
    rows: list[dict[str, Any]] = []
    sorted_data = data.copy().sort_index()

    split_number = 0
    for train_start in range(0, len(sorted_data) - train_size - test_size + 1, step):
        train_end = train_start + train_size
        test_end = train_end + test_size
        train = sorted_data.iloc[train_start:train_end]
        test = sorted_data.iloc[train_end:test_end]
        split_number += 1

        for architecture, factory in model_factories.items():
            model = factory()
            fitted = trainer(model, train)
            if fitted is not None:
                model = fitted
            signals = predictor(model, test)
            engine = BacktestEngine(test, config=config, **price_columns)
            _, metrics = engine.run(signals)
            rows.append(
                {
                    "architecture": architecture,
                    "split": split_number,
                    "train_start": train.index[0],
                    "train_end": train.index[-1],
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "WalkForwardSplit",
    "calculate_metrics",
    "walk_forward_evaluate",
]
