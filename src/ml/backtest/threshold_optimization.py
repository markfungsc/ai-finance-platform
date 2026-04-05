"""Grid search over classification thresholds without retraining."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from constants import THRESHOLD, THRESHOLD_TRADING_DAYS_PER_TWO_MONTHS
from ml.backtest.engine import basic_backtest, pooled_avg_buyhold_market_factor

# ~0.05 to 0.8 inclusive, step 0.05
DEFAULT_THRESHOLD_GRID = np.arange(0.05, 0.81, 0.05)

# Denominator floor for calmar_proxy when drawdown magnitude is ~0
DEFAULT_CALMAR_EPS = 1e-6

OBJECTIVE_AVG_CUM_RETURN = "avg_cum_return"
OBJECTIVE_CALMAR_PROXY = "calmar_proxy"
OBJECTIVE_RISK_PENALTY = "risk_penalty"
OBJECTIVE_MEDIAN_CUM_RETURN = "median_cum_return"
OBJECTIVE_MAXIMIN_CUM_RETURN = "maximin_cum_return"

VALID_OBJECTIVES = frozenset(
    {
        OBJECTIVE_AVG_CUM_RETURN,
        OBJECTIVE_CALMAR_PROXY,
        OBJECTIVE_RISK_PENALTY,
        OBJECTIVE_MEDIAN_CUM_RETURN,
        OBJECTIVE_MAXIMIN_CUM_RETURN,
    }
)


def count_total_backtest_trading_days(split_details: list[dict]) -> int:
    """
    Sum of unique trading timestamps (or row counts) per walk-forward test window.
    Pooled data: one row per symbol per day; nunique(timestamp) counts calendar days.
    """
    total = 0
    for d in split_details:
        rows = d.get("df_test_rows")
        if rows is None or getattr(rows, "empty", True):
            continue
        if "timestamp" in rows.columns:
            total += int(rows["timestamp"].nunique())
        else:
            total += int(len(rows))
    return total


def effective_min_total_trades(
    min_total_trades: int,
    total_trading_days: int,
    *,
    trading_days_per_two_months: int = THRESHOLD_TRADING_DAYS_PER_TWO_MONTHS,
) -> int:
    """
    Require at least one completed trade per ``trading_days_per_two_months`` trading
    days on average (ceiling), in addition to the fixed ``min_total_trades`` floor.
    If no trading days could be counted, only ``min_total_trades`` applies.
    """
    if total_trading_days <= 0:
        return int(min_total_trades)
    span = max(int(trading_days_per_two_months), 1)
    scaled = max(1, math.ceil(total_trading_days / span))
    return max(int(min_total_trades), scaled)


def metrics_for_split_at_threshold(
    df_pred: pd.DataFrame,
    threshold: float,
    df_test_rows: pd.DataFrame | None,
    pooled_mode: bool,
) -> dict:
    override = None
    if pooled_mode and df_test_rows is not None:
        override = pooled_avg_buyhold_market_factor(df_test_rows)
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
            "median_cum_return": 1.0,
            "min_cum_return": 1.0,
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

    def finite_median(vals: list[float], default: float = 1.0) -> float:
        arr = np.asarray(vals, dtype=float)
        m = np.nanmedian(arr)
        return float(m) if np.isfinite(m) else default

    def finite_min(vals: list[float], default: float = 1.0) -> float:
        arr = np.asarray(vals, dtype=float)
        m = np.nanmin(arr)
        return float(m) if np.isfinite(m) else default

    cum_returns = [m["cum_return"] for m in per_split]
    return {
        "avg_cum_return": finite_mean(cum_returns, 1.0),
        "median_cum_return": finite_median(cum_returns, 1.0),
        "min_cum_return": finite_min(cum_returns, 1.0),
        "avg_cum_market_return": finite_mean(
            [m["cum_market_return"] for m in per_split], 1.0
        ),
        "avg_profit_factor": finite_mean([m["profit_factor"] for m in per_split]),
        "avg_max_drawdown": finite_mean([m["max_drawdown"] for m in per_split]),
        "avg_win_rate": finite_mean([m["win_rate"] for m in per_split]),
        "avg_expectancy": finite_mean([m["expectancy"] for m in per_split]),
        "total_trades": int(sum(m["strategy_trade_count"] for m in per_split)),
    }


def threshold_selection_score(
    agg: dict,
    objective: str,
    *,
    lambda_dd: float = 1.0,
    calmar_eps: float = DEFAULT_CALMAR_EPS,
) -> float:
    """
    Higher is better. Used only for ordering threshold grid candidates.
    """
    if objective not in VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective {objective!r}; expected one of {sorted(VALID_OBJECTIVES)}"
        )

    avg_r = float(agg.get("avg_cum_return", 1.0))
    med_r = float(agg.get("median_cum_return", 1.0))
    min_r = float(agg.get("min_cum_return", 1.0))
    avg_dd = float(agg.get("avg_max_drawdown", 0.0))

    if objective == OBJECTIVE_AVG_CUM_RETURN:
        return avg_r
    if objective == OBJECTIVE_MEDIAN_CUM_RETURN:
        return med_r
    if objective == OBJECTIVE_MAXIMIN_CUM_RETURN:
        return min_r

    abs_dd = abs(avg_dd)
    excess = avg_r - 1.0

    if objective == OBJECTIVE_RISK_PENALTY:
        return excess - float(lambda_dd) * abs_dd

    # calmar_proxy
    denom = max(float(calmar_eps), abs_dd)
    return excess / denom


def _passes_profit_factor(agg: dict, min_avg_profit_factor: float | None) -> bool:
    if min_avg_profit_factor is None:
        return True
    pf = agg.get("avg_profit_factor", 0.0)
    if pf == float("inf"):
        return True
    return float(pf) >= float(min_avg_profit_factor)


def _passes_drawdown_cap(agg: dict, max_mean_abs_drawdown: float | None) -> bool:
    if max_mean_abs_drawdown is None:
        return True
    return abs(float(agg.get("avg_max_drawdown", 0.0))) <= float(max_mean_abs_drawdown)


def _passes_hard_constraints(
    agg: dict,
    *,
    min_avg_profit_factor: float | None,
    max_mean_abs_drawdown: float | None,
) -> bool:
    return _passes_profit_factor(agg, min_avg_profit_factor) and _passes_drawdown_cap(
        agg, max_mean_abs_drawdown
    )


def optimize_thresholds(
    split_details: list[dict],
    pooled_mode: bool,
    thresholds: np.ndarray | None = None,
    *,
    min_total_trades: int = 5,
    objective: str = OBJECTIVE_CALMAR_PROXY,
    lambda_dd: float = 1.0,
    calmar_eps: float = DEFAULT_CALMAR_EPS,
    min_avg_profit_factor: float | None = None,
    max_mean_abs_drawdown: float | None = None,
    trading_days_per_two_months: int = THRESHOLD_TRADING_DAYS_PER_TWO_MONTHS,
) -> dict:
    """
    For each threshold, run basic_backtest on stored df_pred per split (no retrain).
    Picks the best candidate by ``threshold_selection_score`` among those meeting
    ``min_total_trades`` (plus a scale from total test trading days) and optional
    hard constraints; if none qualify, falls back to the best score over the full grid.
    """
    if not split_details:
        return {
            "best_threshold": float(THRESHOLD),
            "best_aggregate": {},
            "grid": [],
            "objective": objective,
            "lambda_dd": lambda_dd,
            "calmar_eps": calmar_eps,
            "min_avg_profit_factor": min_avg_profit_factor,
            "max_mean_abs_drawdown": max_mean_abs_drawdown,
            "constraints_relaxed": False,
            "total_backtest_trading_days": 0,
            "min_total_trades_base": int(min_total_trades),
            "min_total_trades_effective": int(min_total_trades),
            "trading_days_per_two_months": int(trading_days_per_two_months),
        }

    if objective not in VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective {objective!r}; expected one of {sorted(VALID_OBJECTIVES)}"
        )

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_GRID

    total_backtest_trading_days = count_total_backtest_trading_days(split_details)
    min_total_trades_effective = effective_min_total_trades(
        min_total_trades,
        total_backtest_trading_days,
        trading_days_per_two_months=trading_days_per_two_months,
    )

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
        agg["per_split"] = [
            {
                "split": int(d.get("split", j)),
                "cum_return": float(m["cum_return"]),
                "cum_market_return": float(m["cum_market_return"]),
            }
            for j, (d, m) in enumerate(zip(split_details, per_split))
        ]
        agg["threshold"] = float(t)
        agg["selection_score"] = threshold_selection_score(
            agg,
            objective,
            lambda_dd=lambda_dd,
            calmar_eps=calmar_eps,
        )
        grid.append(agg)

    def score_row(r: dict) -> float:
        return float(r["selection_score"])

    trade_ok = [r for r in grid if r["total_trades"] >= min_total_trades_effective]
    constrained = [
        r
        for r in trade_ok
        if _passes_hard_constraints(
            r,
            min_avg_profit_factor=min_avg_profit_factor,
            max_mean_abs_drawdown=max_mean_abs_drawdown,
        )
    ]

    constraints_relaxed = False
    if constrained:
        pool = constrained
    elif trade_ok:
        pool = trade_ok
        constraints_relaxed = True
    else:
        pool = grid
        constraints_relaxed = True

    best = max(pool, key=score_row)

    return {
        "best_threshold": best["threshold"],
        "best_aggregate": dict(best),
        "grid": grid,
        "objective": objective,
        "lambda_dd": lambda_dd,
        "calmar_eps": calmar_eps,
        "min_avg_profit_factor": min_avg_profit_factor,
        "max_mean_abs_drawdown": max_mean_abs_drawdown,
        "constraints_relaxed": constraints_relaxed,
        "total_backtest_trading_days": int(total_backtest_trading_days),
        "min_total_trades_base": int(min_total_trades),
        "min_total_trades_effective": int(min_total_trades_effective),
        "trading_days_per_two_months": int(trading_days_per_two_months),
    }
