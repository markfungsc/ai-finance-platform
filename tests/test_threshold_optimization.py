"""Tests for threshold grid scoring and constraints."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.backtest.threshold_optimization import (
    OBJECTIVE_AVG_CUM_RETURN,
    OBJECTIVE_CALMAR_PROXY,
    OBJECTIVE_MAXIMIN_CUM_RETURN,
    OBJECTIVE_MEDIAN_CUM_RETURN,
    OBJECTIVE_RISK_PENALTY,
    SELECTION_MODE_MULTI_TOP_K,
    SELECTION_MODE_SINGLE,
    aggregate_split_metrics,
    count_total_backtest_trading_days,
    effective_min_total_trades,
    optimize_thresholds,
    select_best_threshold_multi_top_k,
    threshold_selection_score,
)


def _m(
    cum_return: float,
    max_drawdown: float,
    trades: int,
    profit_factor: float = 1.5,
    cum_mkt: float = 1.0,
) -> dict:
    return {
        "cum_return": cum_return,
        "cum_market_return": cum_mkt,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "win_rate": 0.5,
        "expectancy": 0.01,
        "strategy_trade_count": trades,
        "strategy_directional_hits": 0,
        "directional_accuracy": 0.0,
    }


def test_aggregate_split_metrics_median_and_min():
    per = [_m(1.1, -0.1, 5), _m(1.3, -0.2, 5), _m(1.2, -0.15, 5)]
    agg = aggregate_split_metrics(per)
    assert agg["avg_cum_return"] == pytest.approx((1.1 + 1.3 + 1.2) / 3)
    assert agg["median_cum_return"] == pytest.approx(1.2)
    assert agg["min_cum_return"] == pytest.approx(1.1)
    assert agg["avg_directional_accuracy_strategy"] == pytest.approx(0.0)
    assert agg["total_trades"] == 15


def test_aggregate_split_metrics_empty():
    agg = aggregate_split_metrics([])
    assert agg["median_cum_return"] == 1.0
    assert agg["min_cum_return"] == 1.0
    assert agg["total_trades"] == 0


def test_threshold_selection_score_avg_and_median():
    agg = aggregate_split_metrics([_m(1.2, -0.1, 3), _m(1.4, -0.1, 3)])
    assert threshold_selection_score(agg, OBJECTIVE_AVG_CUM_RETURN) == pytest.approx(
        agg["avg_cum_return"]
    )
    assert threshold_selection_score(agg, OBJECTIVE_MEDIAN_CUM_RETURN) == pytest.approx(
        agg["median_cum_return"]
    )


def test_threshold_selection_score_maximin():
    agg = aggregate_split_metrics([_m(1.5, -0.05, 3), _m(1.05, -0.02, 3)])
    assert threshold_selection_score(
        agg, OBJECTIVE_MAXIMIN_CUM_RETURN
    ) == pytest.approx(1.05)


def test_threshold_selection_score_calmar_proxy():
    agg = {
        "avg_cum_return": 1.5,
        "median_cum_return": 1.5,
        "min_cum_return": 1.5,
        "avg_max_drawdown": -0.25,
        "avg_profit_factor": 1.0,
        "total_trades": 10,
    }
    s = threshold_selection_score(agg, OBJECTIVE_CALMAR_PROXY, calmar_eps=1e-6)
    assert s == pytest.approx(0.5 / 0.25)


def test_threshold_selection_score_risk_penalty():
    agg = {
        "avg_cum_return": 1.2,
        "median_cum_return": 1.2,
        "min_cum_return": 1.2,
        "avg_max_drawdown": -0.1,
        "avg_profit_factor": 1.0,
        "total_trades": 10,
    }
    s = threshold_selection_score(agg, OBJECTIVE_RISK_PENALTY, lambda_dd=2.0)
    assert s == pytest.approx(0.2 - 2.0 * 0.1)


def test_threshold_selection_score_bad_objective():
    with pytest.raises(ValueError, match="Unknown objective"):
        threshold_selection_score({}, "not_an_objective")


def test_optimize_thresholds_avg_vs_calmar_picks_differently():
    """High avg return + deep DD loses under calmar vs moderate return + shallow DD."""

    def fake_metrics(df_pred, threshold, df_test_rows, pooled_mode):
        if threshold <= 0.15:
            return _m(2.0, -0.5, 10)
        return _m(1.15, -0.05, 10)

    split_details = [
        {"df_pred_for_backtest": None, "df_test_rows": None},
    ]
    thresholds = np.array([0.1, 0.5])

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        side_effect=fake_metrics,
    ):
        out_avg = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
            objective=OBJECTIVE_AVG_CUM_RETURN,
        )
        out_cal = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
            objective=OBJECTIVE_CALMAR_PROXY,
        )

    assert out_avg["best_threshold"] == pytest.approx(0.1)
    assert out_cal["best_threshold"] == pytest.approx(0.5)
    assert out_avg["objective"] == OBJECTIVE_AVG_CUM_RETURN
    assert out_cal["constraints_relaxed"] is False


def test_optimize_thresholds_drawdown_constraint_excludes_best_avg():
    def fake_metrics(df_pred, threshold, df_test_rows, pooled_mode):
        if threshold <= 0.15:
            return _m(3.0, -0.6, 10, profit_factor=2.0)
        return _m(1.2, -0.05, 10, profit_factor=1.5)

    split_details = [{"df_pred_for_backtest": None, "df_test_rows": None}]
    thresholds = np.array([0.1, 0.5])

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        side_effect=fake_metrics,
    ):
        out = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
            objective=OBJECTIVE_AVG_CUM_RETURN,
            max_mean_abs_drawdown=0.35,
        )

    assert out["best_threshold"] == pytest.approx(0.5)
    assert out["constraints_relaxed"] is False


def test_optimize_thresholds_constraints_relaxed_when_all_violate():
    """If every trade_ok row violates DD cap, fall back to trade_ok pool."""

    def fake_metrics(df_pred, threshold, df_test_rows, pooled_mode):
        return _m(2.0, -0.8, 10)

    split_details = [{"df_pred_for_backtest": None, "df_test_rows": None}]
    thresholds = np.array([0.1])

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        side_effect=fake_metrics,
    ):
        out = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
            objective=OBJECTIVE_AVG_CUM_RETURN,
            max_mean_abs_drawdown=0.1,
        )

    assert out["best_threshold"] == pytest.approx(0.1)
    assert out["constraints_relaxed"] is True


def test_optimize_thresholds_min_profit_factor():
    def fake_metrics(df_pred, threshold, df_test_rows, pooled_mode):
        if threshold <= 0.15:
            return _m(2.0, -0.05, 10, profit_factor=0.5)
        return _m(1.1, -0.05, 10, profit_factor=1.2)

    split_details = [{"df_pred_for_backtest": None, "df_test_rows": None}]
    thresholds = np.array([0.1, 0.5])

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        side_effect=fake_metrics,
    ):
        out = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
            objective=OBJECTIVE_AVG_CUM_RETURN,
            min_avg_profit_factor=1.0,
        )

    assert out["best_threshold"] == pytest.approx(0.5)


def test_optimize_thresholds_grid_has_selection_score():
    split_details = [{"df_pred_for_backtest": None, "df_test_rows": None}]
    thresholds = np.array([0.1])

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        return_value=_m(1.1, -0.1, 10),
    ):
        out = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
        )

    assert len(out["grid"]) == 1
    assert "selection_score" in out["grid"][0]
    assert "selection_score" in out["best_aggregate"]
    assert out["best_aggregate"]["selection_score"] == out["grid"][0]["selection_score"]


def test_count_total_backtest_trading_days_sums_unique_timestamps():
    ts_a = pd.date_range("2020-01-01", periods=3, freq="B")
    ts_b = pd.date_range("2020-02-01", periods=5, freq="B")
    splits = [
        {"df_test_rows": pd.DataFrame({"timestamp": ts_a})},
        {"df_test_rows": pd.DataFrame({"timestamp": ts_b})},
    ]
    assert count_total_backtest_trading_days(splits) == 8


def test_effective_min_total_trades_scales_with_calendar():
    assert effective_min_total_trades(5, 0, trading_days_per_two_months=42) == 5
    assert effective_min_total_trades(1, 84, trading_days_per_two_months=42) == 2
    assert effective_min_total_trades(5, 500, trading_days_per_two_months=42) == 12


def test_trading_day_scaled_min_excludes_sparse_threshold():
    """84 test days / 42 => need >=2 trades; 0-trade row must not win calmar_proxy."""

    rows = pd.DataFrame(
        {"timestamp": pd.date_range("2020-01-01", periods=84, freq="B")}
    )
    split_details = [{"df_pred_for_backtest": None, "df_test_rows": rows}]
    thresholds = np.array([0.1, 0.5])

    def fake_metrics(df_pred, threshold, df_test_rows, pooled_mode):
        if threshold <= 0.15:
            return _m(1.5, -0.001, 0)  # no trades, tiny DD -> strong calmar_proxy
        return _m(1.08, -0.05, 4)

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        side_effect=fake_metrics,
    ):
        out = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
            objective=OBJECTIVE_CALMAR_PROXY,
            trading_days_per_two_months=42,
        )

    assert out["total_backtest_trading_days"] == 84
    assert out["min_total_trades_effective"] == 2
    assert out["best_threshold"] == pytest.approx(0.5)
    assert out["best_aggregate"]["total_trades"] == 4


def test_optimize_thresholds_reports_trade_floor_metadata():
    out = optimize_thresholds([], False)
    assert out["min_total_trades_effective"] == 5
    assert out["total_backtest_trading_days"] == 0
    assert out["selection_mode"] == SELECTION_MODE_SINGLE
    assert out["multi_top_k_meta"] is None


def test_grid_includes_per_split_metrics():
    split_details = [
        {"df_pred_for_backtest": None, "df_test_rows": None, "split": 7},
        {"df_pred_for_backtest": None, "df_test_rows": None, "split": 8},
    ]
    thresholds = np.array([0.2])

    def fake_metrics(df_pred, threshold, df_test_rows, pooled_mode):
        return _m(1.1, -0.1, 3, cum_mkt=1.05)

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        side_effect=fake_metrics,
    ):
        out = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
        )
    row = out["grid"][0]
    assert "per_split" in row
    assert len(row["per_split"]) == 2
    assert [p["split"] for p in row["per_split"]] == [7, 8]
    assert row["per_split"][0]["cum_return"] == pytest.approx(1.1)


def test_select_best_threshold_multi_top_k_smallest_k_wins():
    """Middle row ranks 2nd on both metrics -> max rank 2; wins at K=2 over extremes."""
    pool = [
        {
            "threshold": 0.1,
            "avg_cum_return": 2.0,
            "avg_max_drawdown": -0.5,
            "selection_score": 10.0,
        },
        {
            "threshold": 0.2,
            "avg_cum_return": 1.2,
            "avg_max_drawdown": -0.2,
            "selection_score": 2.0,
        },
        {
            "threshold": 0.3,
            "avg_cum_return": 1.0,
            "avg_max_drawdown": -0.05,
            "selection_score": 1.0,
        },
    ]
    best, meta = select_best_threshold_multi_top_k(
        pool,
        metric_names=["avg_cum_return", "avg_max_drawdown"],
        k_start=2,
        k_max=5,
    )
    assert best["threshold"] == pytest.approx(0.2)
    assert meta["k_effective"] == 2
    assert meta["fallback"] is None


def test_optimize_thresholds_multi_top_k_picks_balanced_threshold():
    """Single-objective calmar can favor one row; multi_top_k favors better mean rank."""

    def fake_metrics(df_pred, threshold, df_test_rows, pooled_mode):
        if threshold <= 0.15:
            # Strong calmar_proxy vs moderate row below.
            return _m(3.0, -0.5, 10, profit_factor=0.7)
        return _m(1.15, -0.05, 10, profit_factor=1.8)

    split_details = [{"df_pred_for_backtest": None, "df_test_rows": None}]
    thresholds = np.array([0.1, 0.5])

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        side_effect=fake_metrics,
    ):
        out_single = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
            objective=OBJECTIVE_CALMAR_PROXY,
            selection_mode=SELECTION_MODE_SINGLE,
        )
        out_multi = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
            objective=OBJECTIVE_CALMAR_PROXY,
            selection_mode=SELECTION_MODE_MULTI_TOP_K,
            multi_top_k_start=2,
            multi_top_k_max=3,
            multi_metrics_spec=(
                "avg_cum_return,avg_max_drawdown,avg_profit_factor"
            ),
        )

    assert out_single["best_threshold"] == pytest.approx(0.1)
    assert out_multi["best_threshold"] == pytest.approx(0.5)
    assert out_multi["selection_mode"] == SELECTION_MODE_MULTI_TOP_K
    assert out_multi["multi_top_k_meta"] is not None


def test_grid_includes_avg_mae_at_threshold_with_y_test():
    split_details = [
        {
            "df_pred_for_backtest": None,
            "df_test_rows": None,
            "probs": np.array([0.3, 0.6, 0.5]),
            "y_test": np.array([0.0, 1.0, 0.0]),
        }
    ]
    thresholds = np.array([0.25, 0.55])

    with patch(
        "ml.backtest.threshold_optimization.metrics_for_split_at_threshold",
        return_value=_m(1.1, -0.1, 10),
    ):
        out = optimize_thresholds(
            split_details,
            False,
            thresholds=thresholds,
            min_total_trades=1,
        )

    assert "avg_mae_at_threshold" in out["grid"][0]
    # t=0.25 -> all preds 1 vs [0,1,0] => MAE 2/3; t=0.55 -> [0,1,0] => MAE 0
    assert out["grid"][0]["avg_mae_at_threshold"] == pytest.approx(2.0 / 3)
    assert out["grid"][1]["avg_mae_at_threshold"] == pytest.approx(0.0)
