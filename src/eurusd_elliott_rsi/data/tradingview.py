"""TradingView data adapter placeholders.

TradingView does not provide a simple official public historical-data API for all users, so this
module keeps ingestion behind an adapter boundary. Replace `TradingViewClient.fetch_ohlcv` with a
licensed data feed, broker export, or an approved TradingView integration before production use.
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class TradingViewClient:
    """Configuration for a TradingView-style OHLCV data source."""

    symbol: str = "EURUSD"
    exchange: str = "FX_IDC"
    interval: str = "1h"

    def fetch_ohlcv(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV bars for the configured market.

        The scaffold returns an empty schema so downstream code and tests can be wired before a
        concrete provider is selected.
        """
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        frame = pd.DataFrame(columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame.set_index("timestamp")
