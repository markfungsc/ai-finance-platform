import numpy as np
import pandas as pd

from constants import (
    EXPERIMENT_STRATEGY_SLUG,
    THRESHOLD,
    THRESHOLD_CALMAR_EPS,
    THRESHOLD_MAX_MEAN_ABS_DRAWDOWN,
    THRESHOLD_MIN_AVG_PROFIT_FACTOR,
    THRESHOLD_MULTI_METRICS,
    THRESHOLD_MULTI_TOP_K_MAX,
    THRESHOLD_MULTI_TOP_K_START,
    THRESHOLD_OBJECTIVE,
    THRESHOLD_SELECTION_LAMBDA_DD,
    THRESHOLD_SELECTION_MODE,
    THRESHOLD_TRADING_DAYS_PER_TWO_MONTHS,
)
from log_config import get_logger
from ml.backtest.engine import basic_backtest, pooled_avg_buyhold_market_curve
from ml.backtest.threshold_optimization import optimize_thresholds
from ml.backtest.walk_forward import walk_forward_split
from ml.evaluate import evaluate_model
from ml.models.save_loads import save_feature_columns, save_model

logger = get_logger(__name__)


def _backtest_split_at_threshold(
    df_pred_for_backtest: pd.DataFrame,
    df_test_rows: pd.DataFrame,
    threshold: float,
    pooled_mode: bool,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run basic_backtest and pooled market overlay; per-symbol metrics at same threshold."""
    df_backtest, metrics_strategy = basic_backtest(
        df_pred_for_backtest,
        pred_col="prob_trade_success",
        threshold=threshold,
    )
    if pooled_mode:
        buyhold_factor, buyhold_path = pooled_avg_buyhold_market_curve(df_test_rows)
        metrics_strategy["cum_market_return"] = buyhold_factor
        metrics_strategy["cum_market_return_pooled_eqw"] = buyhold_factor
        metrics_strategy["market_return_label"] = "pooled_avg_buyhold"
        if "timestamp" in df_test_rows.columns and not df_backtest.empty:
            if len(buyhold_path):
                row_timestamps = pd.Index(df_backtest["timestamp"])
                mapped = row_timestamps.map(buyhold_path)
                cum_m = (
                    pd.Series(mapped, index=df_backtest.index)
                    .ffill()
                    .bfill()
                    .fillna(1.0)
                )
                path_ret = buyhold_path.pct_change().fillna(0.0)
                mapped_r = row_timestamps.map(path_ret).fillna(0.0)
                df_backtest.loc[:, "cum_market_return_pooled_eqw"] = cum_m
                df_backtest.loc[:, "cum_market_return"] = cum_m
                df_backtest.loc[:, "market_return"] = mapped_r.to_numpy(dtype=float)
    per_symbol_metrics: dict = {}
    if "symbol" in df_pred_for_backtest.columns:
        for sym, grp in df_pred_for_backtest.groupby("symbol"):
            _df_sym, _m_sym = basic_backtest(
                grp, pred_col="prob_trade_success", threshold=threshold
            )
            per_symbol_metrics[str(sym)] = _m_sym
    return df_backtest, metrics_strategy, per_symbol_metrics


def run_backtest(
    symbol: str,
    X,
    y,
    model_fn,
    model_name,
    df_merged,  # <-- need actual close prices for backtesting
    train_size=2000,
    test_size=250,
    step_size=250,
    threshold: float = THRESHOLD,
    threshold_objective: str | None = None,
    threshold_lambda_dd: float | None = None,
    threshold_calmar_eps: float | None = None,
    threshold_min_avg_profit_factor: float | None = None,
    threshold_max_mean_abs_drawdown: float | None = None,
    threshold_trading_days_per_two_months: int | None = None,
):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df_merged = df_merged.reset_index(drop=True)

    pooled_mode = ("symbol" in df_merged.columns) and (
        df_merged["symbol"].nunique() > 1
    )
    run_scope = "pooled" if pooled_mode else symbol
    logger.info("Running backtest for %s with %d rows", run_scope, len(X))

    results = []
    if pooled_mode:
        # Time-safe split for pooled symbols: train/test partitions by unique timestamps.
        ts = np.array(sorted(df_merged["timestamp"].unique()))
        split_ranges: list[tuple[int, int, int]] = []
        start = 0
        while True:
            train_end = start + train_size
            test_end = train_end + test_size
            if test_end > len(ts):
                break
            split_ranges.append((start, train_end, test_end))
            start += step_size
    else:
        splits = walk_forward_split(
            X,
            y,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
        )

    final_model = None
    split_details = []

    if pooled_mode:
        split_iter = enumerate(split_ranges)
    else:
        split_iter = enumerate(splits)

    for i, split_data in split_iter:
        if pooled_mode:
            start, train_end, test_end = split_data
            train_ts = ts[start:train_end]
            test_ts = ts[train_end:test_end]

            train_mask = df_merged["timestamp"].isin(train_ts)
            test_mask = df_merged["timestamp"].isin(test_ts)

            X_train = X.loc[train_mask].copy()
            y_train = y.loc[train_mask].copy()
            X_test = X.loc[test_mask].copy()
            y_test = y.loc[test_mask].copy()

            df_test_rows = df_merged.loc[test_mask].copy()

            if X_train.empty or X_test.empty:
                continue
        else:
            X_train, X_test, y_train, y_test = split_data
            df_test_rows = df_merged.loc[X_test.index].copy()

        # exclude symbol column for model training
        feature_cols = [c for c in X.columns if c != "symbol"]
        # before fit/predict
        X_train_model = X_train[feature_cols]
        X_test_model = X_test[feature_cols]

        # Train
        model = model_fn(X_train_model, y_train)
        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_model)[:, 1]
        else:
            # Regressor fallback: map score into [0, 1] for thresholding/backtest.
            preds = model.predict(X_test_model)
            probs = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)

        logger.info(
            "Probability stats min=%s max=%s mean=%s",
            probs.min(),
            probs.max(),
            probs.mean(),
        )
        # Evaluate prediction metrics
        pred_classes = (probs > 0.5).astype(int)
        metrics = evaluate_model(pred_classes, y_test, verbose=False)
        metrics["split"] = i
        results.append(metrics)
        final_model = model

        logger.info("split %s | %s", i, metrics)

        # --- Run basic backtest using predicted returns ---
        df_test = X_test.copy()
        df_test["close"] = df_test_rows["close"].to_numpy()
        df_test["high"] = df_test_rows["high"].to_numpy()
        df_test["low"] = df_test_rows["low"].to_numpy()
        if "timestamp" in df_test_rows.columns:
            df_test["timestamp"] = df_test_rows["timestamp"].to_numpy()
        if "symbol" in df_test_rows.columns:
            df_test["symbol"] = df_test_rows["symbol"].to_numpy()
        df_test["prob_trade_success"] = probs

        df_pred_for_backtest = df_test.copy()
        df_backtest, metrics_strategy, per_symbol_metrics = (
            _backtest_split_at_threshold(
                df_pred_for_backtest,
                df_test_rows,
                threshold,
                pooled_mode,
            )
        )
        split_details.append(
            {
                "split": i,
                "X_train_head": X_train.head().copy(),
                "y_train_head": y_train.head().copy(),
                "X_test_head": X_test.head().copy(),
                "y_test_head": y_test.head().copy(),
                "y_test": y_test.copy(),
                "probs": probs.copy(),
                "df_pred_for_backtest": df_pred_for_backtest,
                "df_test_rows": df_test_rows.copy(),
                "df_backtest": df_backtest.copy(),
                "metrics_strategy": metrics_strategy,
                "per_symbol_metrics": per_symbol_metrics,
                "market_return_label": metrics_strategy.get(
                    "market_return_label", "single_symbol_close"
                ),
            }
        )

    threshold_optimization = optimize_thresholds(
        split_details,
        pooled_mode,
        objective=threshold_objective or THRESHOLD_OBJECTIVE,
        lambda_dd=(
            float(threshold_lambda_dd)
            if threshold_lambda_dd is not None
            else float(THRESHOLD_SELECTION_LAMBDA_DD)
        ),
        calmar_eps=(
            float(threshold_calmar_eps)
            if threshold_calmar_eps is not None
            else float(THRESHOLD_CALMAR_EPS)
        ),
        min_avg_profit_factor=(
            threshold_min_avg_profit_factor
            if threshold_min_avg_profit_factor is not None
            else THRESHOLD_MIN_AVG_PROFIT_FACTOR
        ),
        max_mean_abs_drawdown=(
            threshold_max_mean_abs_drawdown
            if threshold_max_mean_abs_drawdown is not None
            else THRESHOLD_MAX_MEAN_ABS_DRAWDOWN
        ),
        trading_days_per_two_months=(
            int(threshold_trading_days_per_two_months)
            if threshold_trading_days_per_two_months is not None
            else int(THRESHOLD_TRADING_DAYS_PER_TWO_MONTHS)
        ),
        selection_mode=THRESHOLD_SELECTION_MODE,
        multi_top_k_start=THRESHOLD_MULTI_TOP_K_START,
        multi_top_k_max=THRESHOLD_MULTI_TOP_K_MAX,
        multi_metrics_spec=THRESHOLD_MULTI_METRICS,
    )
    best_thr = float(threshold_optimization["best_threshold"])
    for d in split_details:
        df_b, m_s, per_sym = _backtest_split_at_threshold(
            d["df_pred_for_backtest"],
            d["df_test_rows"],
            best_thr,
            pooled_mode,
        )
        d["df_backtest"] = df_b.copy()
        d["metrics_strategy"] = m_s
        d["per_symbol_metrics"] = per_sym
        d["market_return_label"] = m_s.get("market_return_label", "single_symbol_close")
        d["backtest_threshold"] = best_thr

    all_backtest_metrics = [d["metrics_strategy"] for d in split_details]

    for d in split_details:
        m = d["metrics_strategy"]
        logger.info(
            "split %s strategy (threshold=%.4f) | cum_return: %.4f, cum_market_return: %.4f, "
            "directional_accuracy: %.2f%% (trades=%s)",
            d["split"],
            d["backtest_threshold"],
            m["cum_return"],
            m["cum_market_return"],
            m["directional_accuracy"],
            m["strategy_trade_count"],
        )

    # Aggregate prediction metrics
    avg_mae = np.mean([r["mae"] for r in results])
    avg_dir = np.mean([r["directional_accuracy"] for r in results])

    # Aggregate backtest metrics (aligned with optimized threshold / saved artifacts)
    def _finite_mean(values, default=0.0):
        m = np.nanmean(values)
        return float(m) if np.isfinite(m) else default

    avg_cum_return = _finite_mean(
        [m["cum_return"] for m in all_backtest_metrics], default=1.0
    )
    avg_cum_market_return = _finite_mean(
        [m["cum_market_return"] for m in all_backtest_metrics], default=1.0
    )
    total_hits = sum(m["strategy_directional_hits"] for m in all_backtest_metrics)
    total_trades = sum(m["strategy_trade_count"] for m in all_backtest_metrics)
    strategy_dir_pooled_pct = (
        (100.0 * total_hits / total_trades) if total_trades else 0.0
    )

    if all_backtest_metrics:
        avg_win_rate = _finite_mean([m["win_rate"] for m in all_backtest_metrics])
        avg_profit_factor = _finite_mean(
            [m["profit_factor"] for m in all_backtest_metrics]
        )
        avg_max_drawdown = _finite_mean(
            [m["max_drawdown"] for m in all_backtest_metrics]
        )
    else:
        avg_win_rate = 0.0
        avg_profit_factor = 0.0
        avg_max_drawdown = 0.0

    summary = {
        "avg_mae": avg_mae,
        "avg_directional_accuracy": avg_dir,
        "avg_strategy_cum_return": avg_cum_return,
        "avg_market_return": avg_cum_market_return,
        "avg_market_return_pooled_eqw": avg_cum_market_return if pooled_mode else None,
        "avg_strategy_directional_accuracy": strategy_dir_pooled_pct,
        "strategy_total_trades": total_trades,
        "strategy_total_hits": total_hits,
        "avg_win_rate": avg_win_rate,
        "avg_profit_factor": avg_profit_factor,
        "avg_max_drawdown": avg_max_drawdown,
        "threshold_optimization": threshold_optimization,
        "split_details": split_details,
    }

    logger.info("Model trained for %s", run_scope)

    # Save final model
    model_path = f"models/{EXPERIMENT_STRATEGY_SLUG}/{model_name}_{run_scope}.pkl"
    save_model(final_model, model_path)
    # Save feature columns
    feature_cols_path = f"models/{EXPERIMENT_STRATEGY_SLUG}/{model_name}_{run_scope}_feature_columns.pkl"
    save_feature_columns(feature_cols, feature_cols_path)
    logger.info("Model saved to %s", model_path)

    return results, summary, final_model
