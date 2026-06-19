"""Utilities for loading TradingView EUR/USD daily OHLCV exports.

The loader intentionally works from CSV exports rather than scraping or relying on
TradingView internals. It normalizes common TradingView column names, validates
that weekday daily candles are present, de-duplicates by timestamp, and writes a
clean Parquet file for downstream analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/eurusd_daily.parquet")
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")
PRICE_COLUMNS = ("open", "high", "low", "close")
OPTIONAL_COLUMNS = ("volume",)
OUTPUT_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass(frozen=True)
class LoadResult:
    """Summary returned after a TradingView CSV is cleaned and saved."""

    input_path: Path
    output_path: Path
    rows: int
    start: object
    end: object
    missing_weekday_candles: tuple[object, ...]


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Loading TradingView CSV files requires pandas. Install pandas and "
            "a Parquet engine such as pyarrow before running this loader."
        ) from exc
    return pd


def _normalize_column_name(name: object) -> str:
    """Convert common TradingView CSV headers to stable snake_case names."""

    normalized = str(name).strip().lower().replace(" ", "_")
    aliases = {
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "open_time": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "vol": "volume",
    }
    return aliases.get(normalized, normalized)


def _ensure_columns(columns: Iterable[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        expected = ", ".join(REQUIRED_COLUMNS)
        raise ValueError(
            f"TradingView CSV is missing required column(s): {missing}. "
            f"Expected at least: {expected}."
        )


def _validate_ohlc(df) -> None:
    bad_high = df["high"] < df[["open", "low", "close"]].max(axis=1)
    bad_low = df["low"] > df[["open", "high", "close"]].min(axis=1)
    if bool(bad_high.any()) or bool(bad_low.any()):
        bad_timestamps = df.loc[bad_high | bad_low, "timestamp"].head(10).tolist()
        raise ValueError(
            "Invalid OHLC rows found: high must be >= open/low/close and low "
            f"must be <= open/high/close. First offending timestamps: {bad_timestamps}"
        )


def _missing_weekday_candles(df):
    pd = _import_pandas()
    if df.empty:
        return pd.DatetimeIndex([], tz="UTC")

    normalized = df["timestamp"].dt.normalize()
    observed = pd.DatetimeIndex(normalized.drop_duplicates())
    expected = pd.date_range(observed.min(), observed.max(), freq="B", tz="UTC")
    return expected.difference(observed)


def load_tradingview_eurusd_csv(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    fail_on_missing: bool = True,
) -> LoadResult:
    """Load, validate, and save a TradingView EUR/USD daily CSV export.

    Parameters
    ----------
    input_path:
        Path to a CSV exported from TradingView with daily EUR/USD candles.
    output_path:
        Destination Parquet path. Parent directories are created as needed.
    fail_on_missing:
        When true, raise ``ValueError`` if weekday candles between the first and
        last date are absent. Set false to save while reporting missing dates in
        the returned ``LoadResult``.
    """

    pd = _import_pandas()
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"TradingView CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df = df.rename(columns={column: _normalize_column_name(column) for column in df.columns})
    _ensure_columns(df.columns)

    df = df[list(dict.fromkeys([*REQUIRED_COLUMNS, *[c for c in OPTIONAL_COLUMNS if c in df.columns]]))].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        bad_count = int(df["timestamp"].isna().sum())
        raise ValueError(f"Unable to parse {bad_count} timestamp value(s) as dates.")

    for column in PRICE_COLUMNS + tuple(c for c in OPTIONAL_COLUMNS if c in df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            bad_count = int(df[column].isna().sum())
            raise ValueError(f"Column {column!r} contains {bad_count} non-numeric value(s).")

    if "volume" not in df.columns:
        df["volume"] = pd.NA

    df = df.sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    _validate_ohlc(df)

    missing = _missing_weekday_candles(df)
    if fail_on_missing and len(missing) > 0:
        missing_preview = [date.strftime("%Y-%m-%d") for date in missing[:10]]
        raise ValueError(
            "Missing weekday daily candle(s) detected between the first and last "
            f"timestamp. First missing dates: {missing_preview}. Re-export the "
            "full TradingView range or call with fail_on_missing=False."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df[list(OUTPUT_COLUMNS)]
    df.to_parquet(output_path, index=False)

    return LoadResult(
        input_path=input_path,
        output_path=output_path,
        rows=len(df),
        start=df["timestamp"].iloc[0] if len(df) else None,
        end=df["timestamp"].iloc[-1] if len(df) else None,
        missing_weekday_candles=tuple(missing.to_pydatetime()),
    )


def load_default_raw_csv(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    fail_on_missing: bool = True,
) -> LoadResult:
    """Load the only CSV in ``data/raw`` or one named like an EUR/USD export."""

    raw_dir = Path(raw_dir)
    candidates = sorted(raw_dir.glob("*.csv"))
    preferred = [path for path in candidates if "eur" in path.name.lower() and "usd" in path.name.lower()]
    candidates = preferred or candidates
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple candidate CSV files found; pass one explicitly: {names}")
    return load_tradingview_eurusd_csv(candidates[0], output_path, fail_on_missing=fail_on_missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean a TradingView EUR/USD daily CSV export.")
    parser.add_argument("csv", nargs="?", help="CSV path. If omitted, auto-detects one CSV in data/raw/.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Destination Parquet path.")
    parser.add_argument("--allow-missing", action="store_true", help="Save even if weekday candles are missing.")
    args = parser.parse_args(argv)

    result = (
        load_tradingview_eurusd_csv(args.csv, args.output, fail_on_missing=not args.allow_missing)
        if args.csv
        else load_default_raw_csv(output_path=args.output, fail_on_missing=not args.allow_missing)
    )
    print(f"Saved {result.rows} rows to {result.output_path}")
    print(f"Date range: {result.start} -> {result.end}")
    if result.missing_weekday_candles:
        print(f"Missing weekday candles: {len(result.missing_weekday_candles)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
