"""Deterministic Elliott-wave approximation features.

This module intentionally implements an *approximation* rather than a formal
Elliott-wave labeller.  It derives confirmed swing pivots with a ZigZag-style
state machine, builds alternating pivot legs, and exposes causal features that
can be used by a model.

No-lookahead rule
-----------------
A pivot at bar ``p`` is not available at bar ``p``.  It is available only at its
``confirmed_index``:

* In ZigZag mode, a swing high is confirmed only after a later bar trades down
  by at least the configured reversal threshold from that high; a swing low is
  confirmed only after a later bar trades up by that threshold.
* ``confirmation_bars`` adds an optional minimum delay after the pivot bar.

Feature rows at bar ``t`` use only pivots with ``confirmed_index <= t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    pd = None

PivotKind = Literal["high", "low"]


@dataclass(frozen=True)
class Pivot:
    """A confirmed swing pivot.

    ``index`` is the bar where the extreme occurred. ``confirmed_index`` is the
    first bar where downstream code may use the pivot without lookahead bias.
    """

    index: int
    confirmed_index: int
    price: float
    kind: PivotKind


def _column(data, name: str) -> list[float]:
    values = data[name]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(v) for v in values]


def _index(data, length: int):
    return getattr(data, "index", range(length))


def average_true_range(
    high: list[float], low: list[float], close: list[float], period: int = 14
) -> list[float]:
    """Return Wilder-style ATR using the same smoothing factor as Wilder ATR."""

    out: list[float] = []
    prev_atr = 0.0
    alpha = 1 / period
    for i, (hi, lo, cl) in enumerate(zip(high, low, close)):
        prev_close = close[i - 1] if i else cl
        true_range = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        prev_atr = true_range if i == 0 else alpha * true_range + (1 - alpha) * prev_atr
        out.append(prev_atr)
    return out


def detect_pivots(
    data,
    *,
    pct_threshold: float = 0.005,
    atr_multiplier: float | None = None,
    atr_period: int = 14,
    confirmation_bars: int = 1,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> list[Pivot]:
    """Detect confirmed alternating ZigZag pivots.

    The reversal threshold at each bar is the maximum of ``pct_threshold`` and,
    when requested, ``atr_multiplier * ATR / price``.  The returned pivots are
    sorted by confirmation time and alternate between highs and lows.
    """

    if pct_threshold <= 0:
        raise ValueError("pct_threshold must be positive")
    if confirmation_bars < 0:
        raise ValueError("confirmation_bars must be non-negative")
    high = _column(data, high_col)
    if not high:
        return []

    low = _column(data, low_col)
    close = _column(data, close_col)
    atr = average_true_range(high, low, close, atr_period) if atr_multiplier else None

    def threshold(i: int, price: float) -> float:
        pct = pct_threshold
        if atr is not None and price > 0:
            pct = max(pct, float(atr[i]) * float(atr_multiplier) / price)
        return pct

    pivots: list[Pivot] = []
    trend: int | None = None  # 1 after low looking for high, -1 after high looking for low
    candidate_high_idx = candidate_low_idx = 0
    candidate_high = float(high[0])
    candidate_low = float(low[0])

    for i in range(1, len(close)):
        hi = float(high[i])
        lo = float(low[i])

        if hi >= candidate_high:
            candidate_high = hi
            candidate_high_idx = i
        if lo <= candidate_low:
            candidate_low = lo
            candidate_low_idx = i

        up_move = (hi - candidate_low) / candidate_low if candidate_low else 0.0
        down_move = (candidate_high - lo) / candidate_high if candidate_high else 0.0

        if trend is None:
            if up_move >= threshold(i, candidate_low) and i - candidate_low_idx >= confirmation_bars:
                pivots.append(Pivot(candidate_low_idx, i, candidate_low, "low"))
                trend = 1
                candidate_high = hi
                candidate_high_idx = i
            elif down_move >= threshold(i, candidate_high) and i - candidate_high_idx >= confirmation_bars:
                pivots.append(Pivot(candidate_high_idx, i, candidate_high, "high"))
                trend = -1
                candidate_low = lo
                candidate_low_idx = i
        elif trend == 1:  # looking for swing high confirmation
            if down_move >= threshold(i, candidate_high) and i - candidate_high_idx >= confirmation_bars:
                pivots.append(Pivot(candidate_high_idx, i, candidate_high, "high"))
                trend = -1
                candidate_low = lo
                candidate_low_idx = i
        else:  # trend == -1, looking for swing low confirmation
            if up_move >= threshold(i, candidate_low) and i - candidate_low_idx >= confirmation_bars:
                pivots.append(Pivot(candidate_low_idx, i, candidate_low, "low"))
                trend = 1
                candidate_high = hi
                candidate_high_idx = i

    return pivots


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def elliott_features(
    data,
    *,
    pct_threshold: float = 0.005,
    atr_multiplier: float | None = None,
    atr_period: int = 14,
    confirmation_bars: int = 1,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
):
    """Return causal Elliott-wave approximation features for each bar.

    Feature definitions:
    * ``elliott_wave_index``: 1..5 for an impulse, then 1..3 for an ABC-style
      correction, estimated from the count of confirmed pivot-to-pivot legs.
    * ``elliott_retracement_ratio``: current pullback versus the previous leg.
    * ``elliott_extension_ratio``: current leg length versus the previous leg.
    * ``elliott_distance_from_last_pivot``: signed percent distance from the
      latest confirmed pivot price.
    * ``elliott_trend_direction``: 1 after a confirmed low, -1 after a confirmed
      high, else 0.
    * ``elliott_pivot_age``: bars since the latest confirmed pivot's true index.
    """

    pivots = detect_pivots(
        data,
        pct_threshold=pct_threshold,
        atr_multiplier=atr_multiplier,
        atr_period=atr_period,
        confirmation_bars=confirmation_bars,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
    )
    close = _column(data, close_col)
    rows: list[dict[str, float]] = []
    available: list[Pivot] = []
    next_pivot = 0

    for i, price in enumerate(close):
        while next_pivot < len(pivots) and pivots[next_pivot].confirmed_index <= i:
            available.append(pivots[next_pivot])
            next_pivot += 1

        if not available:
            rows.append({
                "elliott_wave_index": 0.0,
                "elliott_retracement_ratio": 0.0,
                "elliott_extension_ratio": 0.0,
                "elliott_distance_from_last_pivot": 0.0,
                "elliott_trend_direction": 0.0,
                "elliott_pivot_age": 0.0,
                "elliott_confirmed_pivot_count": 0.0,
            })
            continue

        last = available[-1]
        leg_count = max(len(available) - 1, 0)
        wave_index = float((leg_count % 8) + 1 if leg_count % 8 < 5 else (leg_count % 8) - 4)
        direction = 1.0 if last.kind == "low" else -1.0
        distance = _safe_ratio(float(price) - last.price, last.price) * direction

        prev_leg = abs(available[-1].price - available[-2].price) if len(available) >= 2 else 0.0
        current_leg = abs(float(price) - last.price)
        extension = _safe_ratio(current_leg, prev_leg)

        retracement = 0.0
        if len(available) >= 2:
            prior = available[-2]
            retracement = _safe_ratio(abs(float(price) - last.price), abs(last.price - prior.price))

        rows.append({
            "elliott_wave_index": wave_index,
            "elliott_retracement_ratio": retracement,
            "elliott_extension_ratio": extension,
            "elliott_distance_from_last_pivot": distance,
            "elliott_trend_direction": direction,
            "elliott_pivot_age": float(i - last.index),
            "elliott_confirmed_pivot_count": float(len(available)),
        })

    if pd is not None:
        return pd.DataFrame(rows, index=_index(data, len(rows)))
    return rows
