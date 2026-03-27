"""Unit tests for feature math, label generation, and backtesting."""

import numpy as np
import pandas as pd

from ml.backtest.engine import basic_backtest
from constants import THRESHOLD
from ml.features import TARGET_COLUMN
from ml.helpers.generate_trade_labels import generate_trade_labels


def test_sma_calculation_correctness() -> None:
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    df["sma_3"] = df["close"].rolling(3).mean()
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(df["sma_3"].dropna().to_numpy(), expected)


def test_trade_label_hits_take_profit() -> None:
    # Entry 100 on first row; next row high=110 should hit TP before SL.
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 110.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
        }
    )
    out = generate_trade_labels(df)
    assert TARGET_COLUMN in out.columns
    assert int(out[TARGET_COLUMN].iloc[0]) == 1


def test_basic_backtest_signal_metrics_exist() -> None:
    df = pd.DataFrame(
        {
            "close": [100, 102, 101, 103, 105, 106],
            "high": [101, 103, 102, 104, 106, 107],
            "low": [99, 101, 100, 102, 104, 105],
            "prob_trade_success": [0.9, 0.2, 0.8, 0.1, 0.7, 0.3],
        }
    )
    df_result, metrics = basic_backtest(
        df, pred_col="prob_trade_success", threshold=THRESHOLD
    )
    assert "cum_strategy_return" in df_result.columns
    assert "cum_market_return" in df_result.columns
    assert "cum_return" in metrics
    assert "cum_market_return" in metrics
    assert "strategy_trade_count" in metrics
