"""Unit tests for feature math, label generation, and backtesting."""

import numpy as np
import pandas as pd

from ml.backtest.engine import basic_backtest
from ml.experiments.artifacts import _dedupe_pooled_timestamp_for_plot
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
    for key in (
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "expectancy",
        "average_trade_return",
    ):
        assert key in metrics
        assert isinstance(metrics[key], (int, float))


def test_basic_backtest_pooled_avoids_cross_symbol_lookahead() -> None:
    # Interleaved pooled rows:
    # - Symbol A at t=1 should *not* hit TP based on symbol B's future high.
    # - Symbol B at t=1 has a very high `high`, which would incorrectly trigger A's TP
    #   if the backtest lookahead was row-wise instead of symbol-wise.
    df = pd.DataFrame(
        {
            "close": [100.0, 50.0, 100.0, 50.0],
            "high": [101.0, 120.0, 105.0, 110.0],
            "low": [99.0, 10.0, 97.0, 40.0],
            "timestamp": [1, 1, 2, 2],
            "symbol": ["A", "B", "A", "B"],
            "prob_trade_success": [0.9, 0.1, 0.1, 0.1],
        }
    )

    df_result, metrics = basic_backtest(
        df, pred_col="prob_trade_success", threshold=THRESHOLD
    )

    # Row 0: symbol A at t=1, signal should be 1
    assert int(df_result.loc[0, "signal"]) == 1
    # With correct symbol-wise lookahead, A's next row high=105 (<108 TP) and low=97 (>96 SL) => 0 return.
    assert df_result.loc[0, "strategy_return"] == 0.0

    # Only one trade (A at t=1); no TP hits.
    assert metrics["strategy_trade_count"] == 1
    assert metrics["strategy_directional_hits"] == 0


def test_dedupe_pooled_timestamp_for_plot() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [1, 1, 2, 2],
            "symbol": ["A", "B", "A", "B"],
            "cum_strategy_return": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = _dedupe_pooled_timestamp_for_plot(df)
    assert out["timestamp"].tolist() == [1, 2]
    assert len(out) == 2
