"""Synthetic tests for single-position backtest semantics (renewal, overlap, expiry)."""

import numpy as np

from constants import THRESHOLD, TP_PCT
from ml.backtest.engine import _backtest_single_series


def test_two_early_signals_single_round_trip_no_tp() -> None:
    """Two consecutive signals while flat would have been two trades; only one position."""
    n = 25
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    pred = np.zeros(n)
    pred[0] = 0.5
    pred[1] = 0.5

    _, strat, _, hits, count, entry_trade, exit_trade, entry_price, exit_price, completed = _backtest_single_series(
        close=close,
        high=high,
        low=low,
        pred=pred,
        pred_col_threshold=float(THRESHOLD),
    )

    assert count == 1
    assert int(entry_trade.sum()) == 1
    # Entry at t=0, then a renewal signal at t=1 extends the window.
    assert int(exit_trade.sum()) == 1
    assert exit_trade[11] == 1
    assert len(completed) == 1
    assert hits == 0
    assert strat[0] == 0.0
    assert completed[0] == 0.0
    assert entry_price[0] == 100.0
    assert exit_price[11] == 100.0


def test_renewal_extends_window_tp_later_than_first_deadline() -> None:
    """Renewal at t=5 pushes deadline so TP at bar 12 is still inside the window."""
    n = 20
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    high[12] = 100.0 * (1.0 + TP_PCT)

    pred = np.zeros(n)
    pred[0] = 0.5
    pred[5] = 0.5

    _, strat, _, hits, count, entry_trade, exit_trade, entry_price, exit_price, completed = _backtest_single_series(
        close=close,
        high=high,
        low=low,
        pred=pred,
        pred_col_threshold=float(THRESHOLD),
    )

    assert count == 1
    assert int(entry_trade.sum()) == 1
    assert int(exit_trade.sum()) == 1
    assert exit_trade[12] == 1
    assert hits == 1
    assert strat[0] == TP_PCT
    assert completed[0] == TP_PCT
    assert entry_price[0] == 100.0
    assert exit_price[12] == 100.0 * (1.0 + TP_PCT)


def test_without_renewal_trade_expires_before_tp_bar() -> None:
    """Same path as renewal test but no second signal — window ends before bar 12 TP."""
    n = 20
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    high[12] = 100.0 * (1.0 + TP_PCT)

    pred = np.zeros(n)
    pred[0] = 0.5

    _, strat, _, hits, count, _, exit_trade, _, exit_price, completed = _backtest_single_series(
        close=close,
        high=high,
        low=low,
        pred=pred,
        pred_col_threshold=float(THRESHOLD),
    )

    assert count == 1
    assert int(exit_trade.sum()) == 1
    assert exit_trade[10] == 1
    assert hits == 0
    assert strat[0] == 0.0
    assert completed[0] == 0.0
    assert exit_price[10] == 100.0


def test_timeout_realizes_midpoint_return_at_deadline() -> None:
    n = 20
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)

    # Make the timeout exit midpoint meaningfully above entry at the deadline bar.
    # deadline for entry at t=0 is t=10 (MAX_HOLD_DAYS=10).
    # Keep below TP (=108) so we don't trigger a take-profit.
    high[10] = 107.0
    low[10] = 105.0

    pred = np.zeros(n)
    pred[0] = 0.5

    _, strat, _, hits, count, entry_trade, exit_trade, entry_price, exit_price, completed = _backtest_single_series(
        close=close,
        high=high,
        low=low,
        pred=pred,
        pred_col_threshold=float(THRESHOLD),
    )

    assert count == 1
    assert hits == 0
    assert int(entry_trade.sum()) == 1
    assert int(exit_trade.sum()) == 1
    assert exit_trade[10] == 1

    expected_exit = 0.5 * (107.0 + 105.0)
    expected_ret = (expected_exit / 100.0) - 1.0
    assert entry_price[0] == 100.0
    assert exit_price[10] == expected_exit
    assert strat[0] == expected_ret
    assert completed[0] == expected_ret
