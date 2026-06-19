# EUR/USD Elliott RSI

## TradingView EUR/USD daily data

This project ingests historical daily EUR/USD OHLCV data from a local TradingView CSV export. Using a CSV keeps the data pipeline reproducible and avoids fragile scraping or unofficial API assumptions.

### Export from TradingView

1. Open TradingView and load the **EURUSD** symbol you want to use, for example **FX:EURUSD** or **OANDA:EURUSD**.
2. Set the chart interval to **1D**.
3. Adjust the visible/history range so it includes the full period you want to train or evaluate on.
4. Use TradingView's chart data export feature to download the candles as a CSV file.
5. Save the CSV under `data/raw/`, preferably with a clear name such as `data/raw/eurusd_daily_tradingview.csv`.

The CSV must include these columns, case-insensitively:

- `timestamp` or `date` (TradingView `time` is also accepted)
- `open`
- `high`
- `low`
- `close`
- optional `volume`

### Clean and process the export

Run the TradingView loader from the repository root:

```bash
PYTHONPATH=src python -m eurusd_elliott_rsi.data.tradingview data/raw/eurusd_daily_tradingview.csv
```

If there is exactly one CSV in `data/raw/`, or exactly one CSV whose filename contains both `eur` and `usd`, you can omit the input path:

```bash
PYTHONPATH=src python -m eurusd_elliott_rsi.data.tradingview
```

The loader normalizes timestamps to UTC, sorts candles ascending, removes duplicate timestamps, validates OHLC values, checks for missing weekday daily candles, and writes the cleaned dataset to:

```text
data/processed/eurusd_daily.parquet
```

By default, the loader fails when weekday candles are missing between the first and last date. If you intentionally exported a sparse range and still want to write the Parquet file, pass `--allow-missing`.
