import numpy as np
import pandas as pd
import pytest

from ui.backtest_tab import build_trade_pnl_table


def test_build_trade_pnl_table_drops_non_finite_returns():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="D"),
            "close": [10.0, 11.0, 12.0],
            "open": [9.0, 10.0, 11.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "volume": [1000, 2000, 3000],
            "signal": [1, 1, 1],
            "strategy_return": [0.1, np.nan, 0.2],
        }
    )

    tbl = build_trade_pnl_table(df)
    assert not tbl.empty
    assert np.isfinite(tbl["trade_return"]).all()
    assert np.isfinite(tbl["trade_compounded_equity"]).all()


def test_build_trade_pnl_table_allows_nan_prices_but_not_returns():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="D"),
            "close": [np.nan, 11.0],
            "open": [np.nan, 10.0],
            "high": [np.nan, 12.0],
            "low": [np.nan, 11.0],
            "volume": [np.nan, 2000],
            "signal": [1, 1],
            "strategy_return": [0.1, 0.2],
        }
    )

    tbl = build_trade_pnl_table(df)
    assert not tbl.empty
    # We still require finite trade metrics
    assert np.isfinite(tbl["trade_return"]).all()
    assert np.isfinite(tbl["trade_compounded_equity"]).all()


def test_build_trade_pnl_table_uses_exit_trade_and_exit_price_when_present():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="D"),
            "close": [100.0, 101.0, 102.0, 103.0],
            "open": [99.0, 100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [98.0, 99.0, 100.0, 101.0],
            "volume": [1000, 1100, 1200, 1300],
            # Only the entry bar should carry strategy_return.
            "signal": [0, 1, 0, 0],
            "strategy_return": [0.0, 0.1, 0.0, 0.0],
            "entry_trade": [0, 1, 0, 0],
            "exit_trade": [0, 0, 0, 1],
            "entry_price": [np.nan, 100.0, np.nan, np.nan],
            # Raw exit column can disagree with strategy_return; display uses implied exit.
            "exit_price": [np.nan, np.nan, np.nan, 108.0],
        }
    )

    tbl = build_trade_pnl_table(df)
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert row["entry_timestamp"] == df.loc[1, "timestamp"]
    assert row["exit_timestamp"] == df.loc[3, "timestamp"]
    assert row["entry_price"] == 100.0
    assert row["exit_price"] == 100.0 * (1.0 + 0.1)
    assert row["trade_return"] == 0.1
    assert row["trade_compounded_equity"] == 1.1


def test_build_trade_pnl_table_keeps_entry_exit_aligned_when_dropping_nan_returns():
    """Dropping an entry row with NaN strategy_return must drop the paired exit row."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="D"),
            "close": [10.0, 11.0, 12.0, 13.0],
            "signal": [1, 0, 1, 0],
            "strategy_return": [np.nan, 0.0, -0.04, 0.0],
            "entry_trade": [1, 0, 1, 0],
            "exit_trade": [0, 1, 0, 1],
            "entry_price": [10.0, np.nan, 12.0, np.nan],
            "exit_price": [np.nan, 11.0, np.nan, 11.5],
        }
    )
    tbl = build_trade_pnl_table(df)
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert row["trade_return"] == -0.04
    assert row["entry_price"] == 12.0
    assert row["exit_price"] == 12.0 * (1.0 - 0.04)


def test_build_trade_pnl_table_derives_exit_price_when_exit_bar_close_mismatches_return():
    """NaN exit_price on exit row must not fall back to close if it contradicts strategy_return."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="D"),
            "close": [100.0, 100.0, 100.0, 50.0],
            "signal": [0, 1, 0, 0],
            "strategy_return": [0.0, 0.08, 0.0, 0.0],
            "entry_trade": [0, 1, 0, 0],
            "exit_trade": [0, 0, 0, 1],
            "entry_price": [np.nan, 100.0, np.nan, np.nan],
            "exit_price": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    tbl = build_trade_pnl_table(df)
    assert len(tbl) == 1
    assert tbl.iloc[0]["exit_price"] == 100.0 * (1.0 + 0.08)


def test_build_trade_pnl_table_legacy_without_exit_trade_derives_exit_from_return():
    """Artifacts without exit_trade use next-bar close as heuristic; display must match return."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 3,
            "timestamp": pd.date_range("2004-10-13", periods=3, freq="D"),
            "close": [0.8315, 0.8276, 0.90],
            "signal": [1, 0, 0],
            "strategy_return": [0.08, 0.0, 0.0],
        }
    )
    tbl = build_trade_pnl_table(df)
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert row["entry_price"] == 0.8315
    assert row["exit_price"] == 0.8315 * (1.0 + 0.08)
    assert row["trade_return"] == 0.08


def test_build_trade_pnl_table_nan_entry_price_falls_back_to_close():
    """When entry_price is NaN but close is set, synthetic exit uses close as entry."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="D"),
            "close": [100.0, 100.0, 102.0, 103.0],
            "signal": [0, 1, 0, 0],
            "strategy_return": [0.0, 0.1, 0.0, 0.0],
            "entry_trade": [0, 1, 0, 0],
            "exit_trade": [0, 0, 0, 1],
            "entry_price": [np.nan, np.nan, np.nan, np.nan],
            "exit_price": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    tbl = build_trade_pnl_table(df)
    assert len(tbl) == 1
    assert tbl.iloc[0]["entry_price"] == 100.0
    assert tbl.iloc[0]["exit_price"] == 100.0 * (1.0 + 0.1)



def test_build_trade_pnl_table_fifo_exit_before_entry_same_bar():
    """Same bar: exit prior trade then open next; pairing must not swap rows."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 3,
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="D"),
            "close": [100.0, 101.0, 102.0],
            "signal": [1, 1, 0],
            "strategy_return": [0.08, -0.04, 0.0],
            "entry_trade": [1, 1, 0],
            "exit_trade": [0, 1, 1],
            "entry_price": [100.0, 101.0, np.nan],
            "exit_price": [np.nan, np.nan, np.nan],
        }
    )
    tbl = build_trade_pnl_table(df)
    assert len(tbl) == 2
    assert tbl.iloc[0]["trade_return"] == 0.08
    assert tbl.iloc[0]["entry_price"] == 100.0
    assert tbl.iloc[0]["exit_price"] == pytest.approx(100.0 * 1.08)
    assert tbl.iloc[1]["trade_return"] == -0.04
    assert tbl.iloc[1]["entry_price"] == 101.0
    assert tbl.iloc[1]["exit_price"] == pytest.approx(101.0 * (1.0 - 0.04))
