"""Grid search over classification thresholds without retraining."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from constants import (
    THRESHOLD,
    THRESHOLD_MULTI_METRICS,
    THRESHOLD_TRADING_DAYS_PER_TWO_MONTHS,
)
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

SELECTION_MODE_SINGLE = "single_objective"
SELECTION_MODE_MULTI_TOP_K = "multi_top_k"


def parse_multi_metric_names(spec: str) -> list[str]:
    return [x.strip() for x in (spec or "").split(",") if x.strip()]


def _metric_higher_is_better(name: str) -> bool:
    """Higher numeric value is better (rank 1 = largest), except MAE."""
    return name != "avg_mae_at_threshold"


def average_mae_at_threshold(
    split_details: list[dict],
    threshold: float,
) -> float:
    """
    Mean MAE across splits using binary predictions (prob > threshold) vs ``y_test``.
    Missing ``probs``/``y_test`` on a split yields NaN for that split; aggregate uses nanmean.
    """
    per: list[float] = []
    t = float(threshold)
    for d in split_details:
        probs = d.get("probs")
        y_te = d.get("y_test")
        if probs is None or y_te is None:
            per.append(float("nan"))
            continue
        p = np.asarray(probs, dtype=float).reshape(-1)
        y = np.asarray(y_te, dtype=float).reshape(-1)
        if len(p) != len(y):
            per.append(float("nan"))
            continue
        pred = (p > t).astype(np.float64)
        per.append(float(mean_absolute_error(y, pred)))
    if not per or all(not np.isfinite(x) for x in per):
        return float("nan")
    return float(np.nanmean(np.asarray(per, dtype=float)))


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
            "avg_directional_accuracy_strategy": 0.0,
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
        "avg_directional_accuracy_strategy": finite_mean(
            [m.get("directional_accuracy", 0.0) for m in per_split],
            0.0,
        ),
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


def _ranks_for_metric(values: list[float], *, higher_is_better: bool) -> np.ndarray:
    """Per-row ranks within ``values``; 1 = best. Ties use pandas 'min' rank."""
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    if len(s) == 0:
        return np.array([], dtype=float)
    if s.isna().all():
        return np.ones(len(s), dtype=float)
    if higher_is_better:
        s = s.fillna(s.min() - 1.0)
    else:
        s = s.fillna(s.max() + 1.0)
    ascending = not higher_is_better
    r = s.rank(ascending=ascending, method="min")
    return r.to_numpy(dtype=float)


def select_best_threshold_multi_top_k(
    pool: list[dict],
    *,
    metric_names: list[str],
    k_start: int,
    k_max: int,
) -> tuple[dict, dict]:
    """
    Each row is a threshold grid aggregate (must include keys in ``metric_names``).

    Try the smallest K in [``k_start``, ``k_max``] such that some row has
    rank <= K on **every** metric (ranks computed within ``pool``). Among those,
    pick the row with minimum mean rank; tie-break: higher ``selection_score``.

    If no K works, pick the row with minimum mean rank across metrics (fallback).
    """
    if not pool:
        raise ValueError("pool must be non-empty")
    if not metric_names:
        best = max(pool, key=lambda r: float(r.get("selection_score", 0.0)))
        return best, {
            "mode": "multi_top_k",
            "reason": "empty_metric_list",
            "k_effective": None,
            "fallback": "selection_score",
        }

    names = [n for n in metric_names if n in pool[0]]
    names = [
        n
        for n in names
        if any(np.isfinite(float(r.get(n, float("nan")))) for r in pool)
    ]
    if not names:
        best = max(pool, key=lambda r: float(r.get("selection_score", 0.0)))
        return best, {
            "mode": "multi_top_k",
            "reason": "no_finite_metrics",
            "k_effective": None,
            "fallback": "selection_score",
        }

    n = len(pool)
    rank_matrix = np.zeros((n, len(names)))
    for j, name in enumerate(names):
        vals = [float(r.get(name, float("nan"))) for r in pool]
        rank_matrix[:, j] = _ranks_for_metric(
            vals, higher_is_better=_metric_higher_is_better(name)
        )

    mean_ranks = np.mean(rank_matrix, axis=1)

    k_lo = max(1, int(k_start))
    k_hi = max(k_lo, int(k_max))

    for k in range(k_lo, k_hi + 1):
        eligible = [i for i in range(n) if float(np.max(rank_matrix[i])) <= float(k)]
        if not eligible:
            continue
        best_i = min(
            eligible,
            key=lambda i: (
                float(mean_ranks[i]),
                -float(pool[i].get("selection_score", 0.0)),
            ),
        )
        meta = {
            "mode": "multi_top_k",
            "k_effective": k,
            "metric_names": names,
            "rank_matrix": rank_matrix.tolist(),
            "mean_ranks": mean_ranks.tolist(),
            "fallback": None,
        }
        return pool[best_i], meta

    best_i = int(np.argmin(mean_ranks))
    meta = {
        "mode": "multi_top_k",
        "k_effective": None,
        "metric_names": names,
        "rank_matrix": rank_matrix.tolist(),
        "mean_ranks": mean_ranks.tolist(),
        "fallback": "mean_rank",
    }
    return pool[best_i], meta


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
    selection_mode: str = SELECTION_MODE_SINGLE,
    multi_top_k_start: int = 3,
    multi_top_k_max: int = 16,
    multi_metrics_spec: str | None = None,
) -> dict:
    """
    For each threshold, run basic_backtest on stored df_pred per split (no retrain).
    Picks the best candidate by ``threshold_selection_score`` among those meeting
    ``min_total_trades`` (plus a scale from total test trading days) and optional
    hard constraints; if none qualify, falls back to the best score over the full grid.

    With ``selection_mode=multi_top_k``, picks via :func:`select_best_threshold_multi_top_k`
    on the same eligibility pool (after trade and constraint filters).
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
            "selection_mode": selection_mode,
            "multi_top_k_meta": None,
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
        agg["avg_mae_at_threshold"] = average_mae_at_threshold(split_details, float(t))
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

    mode = (selection_mode or SELECTION_MODE_SINGLE).strip().lower()
    multi_meta: dict | None = None
    m_spec = (
        multi_metrics_spec
        if multi_metrics_spec is not None
        else THRESHOLD_MULTI_METRICS
    )
    if mode == SELECTION_MODE_MULTI_TOP_K and len(pool) > 0:
        metric_names = parse_multi_metric_names(m_spec)
        best, multi_meta = select_best_threshold_multi_top_k(
            pool,
            metric_names=metric_names,
            k_start=multi_top_k_start,
            k_max=multi_top_k_max,
        )
    else:
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
        "selection_mode": mode,
        "multi_top_k_meta": multi_meta,
    }
