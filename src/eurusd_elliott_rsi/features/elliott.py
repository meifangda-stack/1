"""Elliott-wave-inspired swing feature helpers."""

import numpy as np
import pandas as pd


def add_swing_features(frame: pd.DataFrame, price_column: str = "close", window: int = 5) -> pd.DataFrame:
    """Add local swing highs/lows and a coarse wave direction label."""
    enriched = frame.copy()
    price = enriched[price_column]
    rolling_high = price.rolling(window=window, center=True).max()
    rolling_low = price.rolling(window=window, center=True).min()
    enriched["swing_high"] = price.eq(rolling_high)
    enriched["swing_low"] = price.eq(rolling_low)
    enriched["wave_direction"] = np.select(
        [enriched["swing_high"], enriched["swing_low"]],
        [-1, 1],
        default=0,
    )
    return enriched
