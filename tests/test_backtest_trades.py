import numpy as np
import pandas as pd

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

