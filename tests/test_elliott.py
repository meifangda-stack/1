try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from eurusd_elliott_rsi.features.elliott import detect_pivots, elliott_features


def _ohlc(closes):
    data = {"high": closes, "low": closes, "close": closes}
    return pd.DataFrame(data) if pd is not None else data


def _row(features, index, column):
    return features.loc[index, column] if hasattr(features, "loc") else features[index][column]


def test_detect_pivots_alternates_and_records_confirmation_bar():
    data = _ohlc([100, 106, 104, 99, 101, 108, 103, 97, 100])

    pivots = detect_pivots(data, pct_threshold=0.04, confirmation_bars=1)

    assert [(p.index, p.confirmed_index, p.price, p.kind) for p in pivots[:4]] == [
        (0, 1, 100.0, "low"),
        (1, 3, 106.0, "high"),
        (3, 5, 99.0, "low"),
        (5, 6, 108.0, "high"),
    ]
    assert all(a.kind != b.kind for a, b in zip(pivots, pivots[1:]))


def test_features_do_not_expose_unconfirmed_pivots():
    data = _ohlc([100, 106, 104, 99, 101, 108, 103])

    features = elliott_features(data, pct_threshold=0.04, confirmation_bars=1)

    assert _row(features, 2, "elliott_confirmed_pivot_count") == 1.0
    assert _row(features, 2, "elliott_trend_direction") == 1.0
    assert _row(features, 3, "elliott_confirmed_pivot_count") == 2.0
    assert _row(features, 3, "elliott_trend_direction") == -1.0


def test_features_are_stable_when_future_bars_are_appended_before_confirmation():
    prefix = _ohlc([100, 106, 104])
    extended = _ohlc([100, 106, 104, 99, 101])

    prefix_features = elliott_features(prefix, pct_threshold=0.04, confirmation_bars=1)
    extended_features = elliott_features(extended, pct_threshold=0.04, confirmation_bars=1)

    if pd is not None:
        pd.testing.assert_series_equal(prefix_features.iloc[2], extended_features.iloc[2])
    else:
        assert prefix_features[2] == extended_features[2]
