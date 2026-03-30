"""Grid search over classification thresholds without retraining."""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import THRESHOLD
from ml.backtest.engine import basic_backtest, pooled_eqw_market_cum_return

# ~0.05 to 0.8 inclusive, step 0.05
DEFAULT_THRESHOLD_GRID = np.arange(0.05, 0.81, 0.05)


def metrics_for_split_at_threshold(
    df_pred: pd.DataFrame,
    threshold: float,
    df_test_rows: pd.DataFrame | None,
    pooled_mode: bool,
) -> dict:
    override = None
    if pooled_mode and df_test_rows is not None:
        override = pooled_eqw_market_cum_return(df_test_rows)
    _, metrics = basic_backtest(
        df_pred,
        pred_col="prob_trade_success",
        threshold=float(threshold),
        market_cum_return_override=override,
    )
    return metrics


def aggregate_split_metrics(per_split: list[dict]) -> dict:
    if not per_split:
        return {
            "avg_cum_return": 1.0,
            "avg_cum_market_return": 1.0,
            "avg_profit_factor": 0.0,
            "avg_max_drawdown": 0.0,
            "avg_win_rate": 0.0,
            "avg_expectancy": 0.0,
            "total_trades": 0,
        }

    def finite_mean(vals: list[float], default: float = 0.0) -> float:
        arr = np.asarray(vals, dtype=float)
        m = np.nanmean(arr)
        return float(m) if np.isfinite(m) else default

    return {
        "avg_cum_return": finite_mean([m["cum_return"] for m in per_split], 1.0),
        "avg_cum_market_return": finite_mean(
            [m["cum_market_return"] for m in per_split], 1.0
        ),
        "avg_profit_factor": finite_mean([m["profit_factor"] for m in per_split]),
        "avg_max_drawdown": finite_mean([m["max_drawdown"] for m in per_split]),
        "avg_win_rate": finite_mean([m["win_rate"] for m in per_split]),
        "avg_expectancy": finite_mean([m["expectancy"] for m in per_split]),
        "total_trades": int(sum(m["strategy_trade_count"] for m in per_split)),
    }


def optimize_thresholds(
    split_details: list[dict],
    pooled_mode: bool,
    thresholds: np.ndarray | None = None,
    *,
    min_total_trades: int = 5,
) -> dict:
    """
    For each threshold, run basic_backtest on stored df_pred per split (no retrain).
    Picks the threshold with highest avg_cum_return among those meeting min_total_trades.
    """
    if not split_details:
        return {
            "best_threshold": float(THRESHOLD),
            "best_aggregate": {},
            "grid": [],
        }

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_GRID

    grid: list[dict] = []
    for t in thresholds:
        per_split: list[dict] = []
        for d in split_details:
            df_pred = d["df_pred_for_backtest"]
            df_test_rows = d.get("df_test_rows")
            m = metrics_for_split_at_threshold(
                df_pred, float(t), df_test_rows, pooled_mode
            )
            per_split.append(m)
        agg = aggregate_split_metrics(per_split)
        agg["threshold"] = float(t)
        grid.append(agg)

    eligible = [r for r in grid if r["total_trades"] >= min_total_trades]
    if eligible:
        best = max(eligible, key=lambda r: r["avg_cum_return"])
    else:
        best = max(grid, key=lambda r: r["avg_cum_return"])

    return {
        "best_threshold": best["threshold"],
        "best_aggregate": best,
        "grid": grid,
    }
