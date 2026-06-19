"""Run walk-forward EUR/USD strategy experiments.

The module intentionally keeps dependencies light.  If a configured market data
CSV is unavailable, it creates a deterministic synthetic EUR/USD-like series so
CI can still exercise the full experiment and reporting flow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal envs
    yaml = None


@dataclass(frozen=True)
class Period:
    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration with a small JSON-compatible fallback parser."""
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def generate_synthetic_prices(length: int = 3400, seed: int = 7) -> list[dict[str, float]]:
    """Create deterministic EUR/USD-like OHLCV rows for smoke tests."""
    rng = random.Random(seed)
    close = 1.09
    rows: list[dict[str, float]] = []
    for i in range(length):
        cycle = 0.00025 * math.sin(i / 37.0) + 0.00018 * math.sin(i / 113.0)
        shock = rng.gauss(0.0, 0.00055)
        open_ = close
        close = max(0.8, close * (1.0 + cycle + shock))
        high = max(open_, close) * (1.0 + abs(rng.gauss(0.0, 0.0002)))
        low = min(open_, close) * (1.0 - abs(rng.gauss(0.0, 0.0002)))
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": 1000 + rng.random() * 200})
    return rows


def load_rows(config: dict[str, Any]) -> list[dict[str, float]]:
    data_cfg = config["data"]
    csv_path = REPO_ROOT / data_cfg["input_csv"]
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [{key: float(value) for key, value in row.items() if key != data_cfg.get("datetime_column")} for row in reader]
    if data_cfg.get("synthetic_fallback", False):
        return generate_synthetic_prices()
    raise FileNotFoundError(csv_path)


def rsi(values: list[float], period: int) -> list[float]:
    out = [50.0] * len(values)
    for i in range(period, len(values)):
        gains, losses = 0.0, 0.0
        for j in range(i - period + 1, i + 1):
            change = values[j] - values[j - 1]
            gains += max(change, 0.0)
            losses += max(-change, 0.0)
        rs = gains / losses if losses else 100.0
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def rolling_mean(values: list[float], window: int) -> list[float]:
    return [mean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]


def rolling_stdev(values: list[float], window: int) -> list[float]:
    return [pstdev(values[max(0, i - window + 1) : i + 1]) if i else 0.0 for i in range(len(values))]


def enrich(rows: list[dict[str, float]], config: dict[str, Any]) -> list[dict[str, float]]:
    closes = [row["close"] for row in rows]
    returns = [0.0] + [(closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))]
    rsi_values = rsi(closes, config["features"]["rsi_period"])
    trend = rolling_mean(closes, config["features"]["elliott_window"])
    volatility = rolling_stdev(returns, config["features"]["volatility_window"])
    enriched = []
    for i, row in enumerate(rows):
        item = dict(row)
        item.update(
            return_1=returns[i],
            target_return=returns[i + 1] if i + 1 < len(rows) else 0.0,
            target_direction=1.0 if (returns[i + 1] if i + 1 < len(rows) else 0.0) > 0 else 0.0,
            rsi=rsi_values[i],
            trend_distance=(closes[i] / trend[i] - 1.0) if trend[i] else 0.0,
            volatility=volatility[i],
        )
        enriched.append(item)
    return enriched


def make_periods(row_count: int, config: dict[str, Any]) -> list[Period]:
    wf = config["walk_forward"]
    periods = []
    start = 0
    while len(periods) < wf["max_periods"]:
        train_start = 0 if wf.get("expanding") else start
        train_end = start + wf["train_size"]
        test_start = train_end
        test_end = test_start + wf["test_size"]
        if test_end >= row_count:
            break
        periods.append(Period(len(periods) + 1, train_start, train_end, test_start, test_end))
        start += wf["step_size"]
    return periods


def baseline_signal(row: dict[str, float], params: dict[str, Any]) -> int:
    if row["rsi"] <= params.get("oversold", 35) and row["trend_distance"] > 0:
        return 1
    if row["rsi"] >= params.get("overbought", 65) and row["trend_distance"] < 0:
        return -1
    return 0


def pseudo_model_score(row: dict[str, float], family: str, task: str) -> float:
    weights = {
        "mlp": (0.9, -0.5, 0.2),
        "cnn": (0.6, -0.2, 0.7),
        "gru": (0.5, -0.4, 0.9),
        "transformer": (0.8, -0.3, 0.8),
    }.get(family, (0.7, -0.3, 0.5))
    raw = weights[0] * row["trend_distance"] + weights[1] * ((row["rsi"] - 50.0) / 100.0) + weights[2] * row["return_1"]
    if task == "classification":
        return 1.0 / (1.0 + math.exp(-25.0 * raw))
    return raw / 10.0


def signal_for(row: dict[str, float], experiment: dict[str, Any], config: dict[str, Any], train_rows: list[dict[str, float]]) -> int:
    family, task = experiment["family"], experiment["task"]
    if family == "baseline":
        return baseline_signal(row, experiment.get("params", {}))
    if family == "regime_filter":
        threshold = sorted(item["volatility"] for item in train_rows)[max(0, int(0.35 * len(train_rows)) - 1)]
        return baseline_signal(row, {"oversold": 35, "overbought": 65}) if row["volatility"] <= threshold else 0
    if family == "autoencoder_regime":
        centroid = {key: mean(item[key] for item in train_rows) for key in ("return_1", "rsi", "trend_distance", "volatility")}
        error = sum((row[key] - centroid[key]) ** 2 for key in centroid)
        cutoff = sorted(sum((item[key] - centroid[key]) ** 2 for key in centroid) for item in train_rows)[int(0.65 * len(train_rows))]
        return baseline_signal(row, {"oversold": 40, "overbought": 60}) if error <= cutoff else 0
    score = pseudo_model_score(row, family, task)
    backtest = config["backtest"]
    if task == "classification":
        return 1 if score >= backtest["long_threshold"] else -1 if score <= backtest["short_threshold"] else 0
    return 1 if score >= backtest["regression_long_threshold"] else -1 if score <= backtest["regression_short_threshold"] else 0


def summarize_returns(returns: list[float], signals: list[int], config: dict[str, Any]) -> dict[str, float]:
    costs = config["backtest"]["transaction_cost_bps"] / 10000.0
    strategy_returns = []
    previous_signal = 0
    for market_return, signal in zip(returns, signals):
        turnover_cost = costs if signal != previous_signal else 0.0
        strategy_returns.append(signal * market_return - turnover_cost)
        previous_signal = signal
    total_return = math.prod(1.0 + value for value in strategy_returns) - 1.0
    avg = mean(strategy_returns) if strategy_returns else 0.0
    vol = pstdev(strategy_returns) if len(strategy_returns) > 1 else 0.0
    sharpe = (avg / vol * math.sqrt(config["backtest"]["annualization_factor"])) if vol else 0.0
    wins = sum(1 for value in strategy_returns if value > 0)
    traded = sum(1 for signal in signals if signal != 0)
    return {"total_return": total_return, "sharpe": sharpe, "win_rate": wins / len(strategy_returns), "trade_rate": traded / len(signals)}


def run_experiments(config_path: Path = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rows = enrich(load_rows(config), config)
    periods = make_periods(len(rows), config)
    results = []
    for period in periods:
        train_rows = rows[period.train_start : period.train_end]
        test_rows = rows[period.test_start : period.test_end]
        market_returns = [row["target_return"] for row in test_rows]
        for experiment in config["experiments"]:
            signals = [signal_for(row, experiment, config, train_rows) for row in test_rows]
            metrics = summarize_returns(market_returns, signals, config)
            results.append({"period": period.index, "strategy": experiment["name"], "family": experiment["family"], "task": experiment["task"], **metrics})
    return results


def write_results(results: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["period", "strategy", "family", "task", "total_return", "sharpe", "win_rate", "trade_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: list[dict[str, Any]]) -> None:
    strategies = sorted({row["strategy"] for row in results})
    print("strategy,periods,mean_total_return,mean_sharpe,mean_win_rate,mean_trade_rate")
    for strategy in strategies:
        rows = [row for row in results if row["strategy"] == strategy]
        print(
            f"{strategy},{len(rows)},{mean(row['total_return'] for row in rows):.6f},"
            f"{mean(row['sharpe'] for row in rows):.6f},{mean(row['win_rate'] for row in rows):.6f},"
            f"{mean(row['trade_rate'] for row in rows):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    results = run_experiments(args.config)
    output_path = REPO_ROOT / config["outputs"]["results_csv"]
    write_results(results, output_path)
    print_summary(results)
    print(f"\nWrote {len(results)} walk-forward rows to {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
