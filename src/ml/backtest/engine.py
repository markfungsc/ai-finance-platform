import numpy as np
import pandas as pd

from constants import MAX_HOLD_DAYS, SL_PCT, THRESHOLD, TP_PCT


def pooled_eqw_market_cum_return(df_test_rows: pd.DataFrame) -> float:
    """Equal-weight pooled market cumulative return (same as walk-forward pooled mode)."""
    if df_test_rows.empty:
        return 1.0
    required = {"timestamp", "symbol", "close"}
    if not required.issubset(df_test_rows.columns):
        return 1.0

    px = df_test_rows.loc[:, ["timestamp", "symbol", "close"]].copy()
    px = px.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    # Use a single-step `.loc` assignment to avoid pandas CoW/chained-assignment
    # warnings in newer versions.
    px.loc[:, "symbol_return"] = px.groupby("symbol", sort=False)["close"].pct_change()
    eqw_ret = (
        px.dropna(subset=["symbol_return"])
        .groupby("timestamp", sort=True)["symbol_return"]
        .mean()
    )
    if eqw_ret.empty:
        return 1.0
    return float((1.0 + eqw_ret).cumprod().iloc[-1])


def _extended_trade_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Metrics from signal rows (strategy_return, cum_strategy_return)."""
    sig = df["signal"] == 1
    trades = df.loc[sig, "strategy_return"]
    n = int(len(trades))
    if n == 0:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "average_trade_return": 0.0,
        }

    wins = trades[trades > 0]
    losses = trades[trades < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    if gross_loss > 1e-12:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # In pooled mode, `cum_strategy_return` can repeat per timestamp (one per symbol row).
    # Compute drawdown from the unique timestamp curve when possible.
    if "timestamp" in df.columns:
        eq_df = (
            df.loc[:, ["timestamp", "cum_strategy_return"]]
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
        )
        eq = eq_df["cum_strategy_return"].to_numpy(dtype=float)
    else:
        eq = df["cum_strategy_return"].to_numpy(dtype=float)

    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 1e-12, (eq - peak) / peak, 0.0)
    max_drawdown = float(np.min(dd)) if len(dd) else 0.0

    exp = float(trades.mean())
    win_rate = float((trades > 0).sum() / n)

    pf = float(profit_factor) if np.isfinite(profit_factor) else 999.0

    return {
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_drawdown": max_drawdown,
        "expectancy": exp,
        "average_trade_return": exp,
    }


def _backtest_single_series(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    pred: np.ndarray,
    pred_col_threshold: float,
):
    """
    Backtest assuming `pred/high/low/close` are aligned to a single chronological time series.
    Returns arrays aligned with the input order.
    """
    n = int(len(pred))
    signal = (pred > pred_col_threshold).astype(int)
    strategy_returns = np.zeros(n, dtype=float)

    trade_hits = 0
    trade_count = 0

    for i in range(n):
        if signal[i] == 0:
            continue

        trade_count += 1
        entry_price = float(close[i])
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)

        trade_result = 0
        for j in range(1, MAX_HOLD_DAYS + 1):
            if i + j >= n:
                break

            hi = float(high[i + j])
            lo = float(low[i + j])

            if hi >= tp_price:
                strategy_returns[i] = TP_PCT
                trade_result = 1
                break

            if lo <= sl_price:
                strategy_returns[i] = -SL_PCT
                trade_result = 0
                break

        if trade_result == 1:
            trade_hits += 1

    cum_strategy = np.cumprod(1.0 + strategy_returns)
    return signal, strategy_returns, cum_strategy, trade_hits, trade_count


def basic_backtest(
    df: pd.DataFrame,
    pred_col: str = "prob_trade_success",
    threshold: float = THRESHOLD,
    market_cum_return_override: float | None = None,
):
    # Force an owned frame to avoid pandas CoW/chained-assignment warnings.
    df = df.reset_index(drop=True).copy(deep=True)

    required = {"close", "high", "low", pred_col}
    if not required.issubset(df.columns):
        missing = sorted(list(required - set(df.columns)))
        raise ValueError(f"basic_backtest missing columns: {missing}")

    pooled_mode = "symbol" in df.columns and df["symbol"].nunique() > 1
    ts_col = "timestamp" if "timestamp" in df.columns else None

    df.loc[:, "signal"] = 0
    df.loc[:, "strategy_return"] = 0.0
    df.loc[:, "cum_strategy_return"] = 1.0

    trade_hits = 0
    trade_count = 0
    cum_return: float = 1.0

    if pooled_mode:
        # Compute exits (TP/SL) strictly within each symbol to avoid cross-symbol lookahead.
        symbols = [s for s in df["symbol"].dropna().unique()]

        for sym in symbols:
            sub = df.loc[df["symbol"] == sym]
            sub_sorted = sub.sort_values(ts_col) if ts_col else sub

            pred = sub_sorted[pred_col].to_numpy(dtype=float)
            close = sub_sorted["close"].to_numpy(dtype=float)
            high = sub_sorted["high"].to_numpy(dtype=float)
            low = sub_sorted["low"].to_numpy(dtype=float)

            (
                signal_arr,
                strat_arr,
                cum_arr,
                hits,
                trades,
            ) = _backtest_single_series(
                close=close,
                high=high,
                low=low,
                pred=pred,
                pred_col_threshold=float(threshold),
            )

            df.loc[sub_sorted.index, "signal"] = signal_arr
            df.loc[sub_sorted.index, "strategy_return"] = strat_arr

            trade_hits += hits
            trade_count += trades

        if ts_col:
            # Equal-weight pooled portfolio return per timestamp (like your market eqw_ret):
            #   portfolio_return[t] = mean(strategy_return across symbols at t)
            # Then compound the portfolio return to get the pooled equity curve.
            port_ret = df.groupby(ts_col, sort=True)["strategy_return"].mean()
            pooled_curve = (1.0 + port_ret).cumprod()
            eq_map = pooled_curve.to_dict()
            df.loc[:, "cum_strategy_return"] = df[ts_col].map(eq_map).astype(float)

            cum_return = float(pooled_curve.iloc[-1]) if not pooled_curve.empty else 1.0
        else:
            # Fallback if no timestamp is present: keep behavior sane by compounding row-wise.
            df.loc[:, "cum_strategy_return"] = (
                1.0 + df["strategy_return"].astype(float)
            ).cumprod()
            cum_return = float(df["cum_strategy_return"].iloc[-1])
    else:
        # Single symbol case: still sort by timestamp if available.
        sub_sorted = df.sort_values(ts_col) if ts_col else df
        pred = sub_sorted[pred_col].to_numpy(dtype=float)
        close = sub_sorted["close"].to_numpy(dtype=float)
        high = sub_sorted["high"].to_numpy(dtype=float)
        low = sub_sorted["low"].to_numpy(dtype=float)

        (
            signal_arr,
            strat_arr,
            cum_arr,
            hits,
            trades,
        ) = _backtest_single_series(
            close=close,
            high=high,
            low=low,
            pred=pred,
            pred_col_threshold=float(threshold),
        )

        df.loc[sub_sorted.index, "signal"] = signal_arr
        df.loc[sub_sorted.index, "strategy_return"] = strat_arr
        df.loc[sub_sorted.index, "cum_strategy_return"] = cum_arr
        trade_hits += hits
        trade_count += trades
        cum_return = float(cum_arr[-1]) if len(cum_arr) else 1.0

    # simple market benchmark
    df.loc[:, "market_return"] = df["close"].pct_change().fillna(0)
    df.loc[:, "cum_market_return"] = (1 + df["market_return"]).cumprod()

    directional_accuracy = (trade_hits / trade_count) * 100 if trade_count > 0 else 0.0

    ext = _extended_trade_metrics(df)

    metrics = {
        "cum_return": float(cum_return),
        "cum_market_return": float(
            market_cum_return_override
            if market_cum_return_override is not None
            else df["cum_market_return"].iloc[-1]
        ),
        "directional_accuracy": float(directional_accuracy),
        "strategy_directional_hits": trade_hits,
        "strategy_trade_count": trade_count,
        **ext,
    }

    return df, metrics
