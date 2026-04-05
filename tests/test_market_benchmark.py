"""Tests for pooled buy-hold market benchmark (vs daily rebalanced compound)."""

import numpy as np
import pandas as pd

from ml.backtest.engine import (
    pooled_avg_buyhold_market_curve,
    pooled_avg_buyhold_market_factor,
)


def test_pooled_avg_buyhold_terminal_is_mean_of_symbol_totals() -> None:
    """Two symbols: +10% and -10% total -> average growth factor 1.0."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"] * 2),
            "symbol": ["A", "A", "B", "B"],
            "close": [100.0, 110.0, 200.0, 180.0],
        }
    )
    factor, path = pooled_avg_buyhold_market_curve(df)
    assert np.isclose(factor, 1.0)
    assert len(path) == 2
    assert np.isclose(float(path.iloc[0]), 1.0)
    assert np.isclose(float(path.iloc[1]), 1.0)

    assert pooled_avg_buyhold_market_factor(df) == factor


def test_pooled_avg_buyhold_single_timestamp() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["A", "B"],
            "close": [10.0, 20.0],
        }
    )
    factor, path = pooled_avg_buyhold_market_curve(df)
    assert factor == 1.0
    assert len(path) == 1


def test_basic_backtest_single_symbol_market_is_buyhold() -> None:
    from constants import THRESHOLD
    from ml.backtest.engine import basic_backtest

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "high": [101.0, 111.0],
            "low": [99.0, 109.0],
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="D"),
            "prob_trade_success": [0.0, 0.0],
        }
    )
    out, m = basic_backtest(df, pred_col="prob_trade_success", threshold=THRESHOLD)
    assert np.isclose(float(m["cum_market_return"]), 1.1)
    assert np.isclose(float(out["cum_market_return"].iloc[-1]), 1.1)
