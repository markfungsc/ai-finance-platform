import numpy as np
import pandas as pd

from constants import EXPERIMENT_STRATEGY_SLUG, THRESHOLD
from log_config import get_logger
from ml.backtest.engine import basic_backtest, pooled_eqw_market_cum_return
from ml.backtest.threshold_optimization import optimize_thresholds
from ml.backtest.walk_forward import walk_forward_split
from ml.evaluate import evaluate_model
from ml.models.save_loads import save_feature_columns, save_model

logger = get_logger(__name__)


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
    all_backtest_metrics = []
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
        df_backtest, metrics_strategy = basic_backtest(
            df_test, pred_col="prob_trade_success", threshold=threshold
        )
        if pooled_mode:
            pooled_eqw_cum_market_return = pooled_eqw_market_cum_return(df_test_rows)
            metrics_strategy["cum_market_return"] = pooled_eqw_cum_market_return
            metrics_strategy["cum_market_return_pooled_eqw"] = (
                pooled_eqw_cum_market_return
            )
            metrics_strategy["market_return_label"] = "pooled_equal_weight"
            if "timestamp" in df_test_rows.columns and not df_backtest.empty:
                px = df_test_rows.loc[:, ["timestamp", "symbol", "close"]].copy()
                px = px.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
                # Single-step `.loc` assignment to avoid pandas CoW/chained-assignment warnings.
                px.loc[:, "symbol_return"] = px.groupby("symbol", sort=False)[
                    "close"
                ].pct_change()
                eqw_path = (
                    px.dropna(subset=["symbol_return"])
                    .groupby("timestamp", sort=True)["symbol_return"]
                    .mean()
                )
                if not eqw_path.empty:
                    eqw_path = (1.0 + eqw_path).cumprod()
                    row_timestamps = pd.Index(df_test_rows["timestamp"])
                    mapped = row_timestamps.map(eqw_path)
                    df_backtest.loc[:, "cum_market_return_pooled_eqw"] = (
                        pd.Series(mapped, index=df_backtest.index).ffill().fillna(1.0)
                    )
                    df_backtest.loc[:, "cum_market_return"] = df_backtest[
                        "cum_market_return_pooled_eqw"
                    ]
        per_symbol_metrics = {}
        if "symbol" in df_test.columns:
            for sym, grp in df_test.groupby("symbol"):
                _df_sym, _m_sym = basic_backtest(
                    grp, pred_col="prob_trade_success", threshold=threshold
                )
                per_symbol_metrics[str(sym)] = _m_sym

        all_backtest_metrics.append(metrics_strategy)
        split_details.append(
            {
                "split": i,
                "X_train_head": X_train.head().copy(),
                "y_train_head": y_train.head().copy(),
                "X_test_head": X_test.head().copy(),
                "y_test_head": y_test.head().copy(),
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

        logger.info(
            "split %s strategy | cum_return: %.4f, cum_market_return: %.4f, "
            "directional_accuracy: %.2f%% (trades=%s)",
            i,
            metrics_strategy["cum_return"],
            metrics_strategy["cum_market_return"],
            metrics_strategy["directional_accuracy"],
            metrics_strategy["strategy_trade_count"],
        )

    # Aggregate prediction metrics
    avg_mae = np.mean([r["mae"] for r in results])
    avg_dir = np.mean([r["directional_accuracy"] for r in results])

    # Aggregate backtest metrics
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

    threshold_optimization = optimize_thresholds(split_details, pooled_mode)

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
