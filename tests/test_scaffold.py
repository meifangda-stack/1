import pandas as pd

from eurusd_elliott_rsi.backtest.engine import run_vectorized_backtest
from eurusd_elliott_rsi.strategies.baseline import generate_signals


def test_baseline_signal_and_backtest_smoke():
    frame = pd.DataFrame(
        {
            "close": [1.10, 1.11, 1.09, 1.08, 1.12, 1.13, 1.10, 1.09, 1.14, 1.15,
                      1.13, 1.12, 1.16, 1.17, 1.15, 1.14, 1.18, 1.19, 1.17, 1.16],
        }
    )

    signals = generate_signals(frame, rsi_period=3)
    result = run_vectorized_backtest(signals)

    assert "signal" in signals
    assert len(result.equity_curve) == len(frame)
    assert isinstance(result.total_return, float)
