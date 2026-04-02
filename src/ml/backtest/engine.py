import numpy as np
import pandas as pd

from constants import MAX_HOLD_DAYS, SL_PCT, THRESHOLD, TP_PCT


def _stable_sort_by_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Sort by time; break ties with original row order so OHLC aligns with preds."""
    out = df.copy()
    out["_row_order"] = np.arange(len(out))
    return out.sort_values([ts_col, "_row_order"], kind="mergesort").drop(
        columns="_row_order"
    )


def _mask_trade_prices(df: pd.DataFrame) -> None:
    """Clear entry/exit prices where the corresponding trade flag is off (CSV clarity)."""
    et = df["entry_trade"].fillna(0).astype(int).to_numpy()
    xt = df["exit_trade"].fillna(0).astype(int).to_numpy()
    ep = df["entry_price"].to_numpy(dtype=float)
    xp = df["exit_price"].to_numpy(dtype=float)
    ep = np.where(et == 1, ep, np.nan)
    xp = np.where(xt == 1, xp, np.nan)
    df.loc[:, "entry_price"] = ep
    df.loc[:, "exit_price"] = xp


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


def _extended_trade_metrics(
    df: pd.DataFrame,
    completed_trade_returns: np.ndarray | None = None,
) -> dict[str, float]:
    """Metrics from completed round-trip returns (preferred) or legacy signal rows."""
    if completed_trade_returns is not None:
        trades = pd.Series(completed_trade_returns, dtype=float)
    else:
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

    At most one open position at a time. New signals while already in a trade **renew**
    the TP/SL window: ``deadline = t + MAX_HOLD_DAYS`` (same entry price until exit).

    Realized PnL is written to ``strategy_returns[entry_i]`` when the trade closes.
    ``signal`` remains the raw model intent (pred > threshold) for plotting.
    ``entry_trade`` is 1 on bars where a new position opens from flat (not renewals).
    """
    n = int(len(pred))
    signal = (pred > pred_col_threshold).astype(int)
    strategy_returns = np.zeros(n, dtype=float)
    entry_trade = np.zeros(n, dtype=np.int8)
    # Mark the exact bar where the position is closed.
    exit_trade = np.zeros(n, dtype=np.int8)
    entry_price = np.full(n, np.nan, dtype=float)
    exit_price = np.full(n, np.nan, dtype=float)

    trade_hits = 0
    trade_count = 0
    completed_returns: list[float] = []

    open_pos = False
    entry_i = -1
    entry_px = 0.0
    deadline = -1
    last_idx = n - 1

    def _window_deadline(t: int) -> int:
        return min(t + MAX_HOLD_DAYS, last_idx)

    def _timeout_exit_price(exit_t: int) -> float:
        """
        Practical timeout/expiry exit: liquidate at the deadline bar.
        Use midpoint (high+low)/2 when finite, otherwise fall back to close, then entry.
        """
        hi = float(high[exit_t])
        lo = float(low[exit_t])
        if np.isfinite(hi) and np.isfinite(lo):
            return 0.5 * (hi + lo)
        cl = float(close[exit_t])
        if np.isfinite(cl):
            return cl
        return float(entry_px)

    def _close_trade(ret: float, hit_tp: bool, *, exit_t: int, exit_px: float) -> None:
        nonlocal open_pos, trade_count, trade_hits
        strategy_returns[entry_i] = ret
        exit_trade[exit_t] = 1
        exit_price[exit_t] = float(exit_px)
        completed_returns.append(float(ret))
        trade_count += 1
        if hit_tp:
            trade_hits += 1
        open_pos = False

    for t in range(n):
        if open_pos:
            if t > deadline:
                # We close when `t > deadline`, but the actual last in-window bar is `deadline`.
                # Mark the exit on `deadline` for accurate UI history.
                exit_t = int(deadline)
                px = _timeout_exit_price(exit_t)
                ret = (px / entry_px) - 1.0 if entry_px > 1e-12 else 0.0
                _close_trade(
                    float(ret),
                    hit_tp=False,
                    exit_t=exit_t,
                    exit_px=float(px),
                )
            elif t > entry_i:
                tp_price = entry_px * (1 + TP_PCT)
                sl_price = entry_px * (1 - SL_PCT)
                hi = float(high[t])
                lo = float(low[t])
                if hi >= tp_price:
                    _close_trade(
                        TP_PCT,
                        hit_tp=True,
                        exit_t=int(t),
                        exit_px=float(tp_price),
                    )
                elif lo <= sl_price:
                    _close_trade(
                        -SL_PCT,
                        hit_tp=False,
                        exit_t=int(t),
                        exit_px=float(sl_price),
                    )
            if open_pos and t > entry_i and signal[t] == 1:
                deadline = _window_deadline(t)

        if not open_pos and signal[t] == 1:
            open_pos = True
            entry_i = t
            entry_px = float(close[t])
            deadline = _window_deadline(t)
            entry_trade[t] = 1
            entry_price[t] = entry_px

    if open_pos:
        # End-of-series close: treat as timeout at the last `deadline` bar.
        exit_t = int(deadline)
        px = _timeout_exit_price(exit_t)
        ret = (px / entry_px) - 1.0 if entry_px > 1e-12 else 0.0
        _close_trade(
            float(ret),
            hit_tp=False,
            exit_t=exit_t,
            exit_px=float(px),
        )

    cum_strategy = np.cumprod(1.0 + strategy_returns)
    completed = np.asarray(completed_returns, dtype=float)
    return (
        signal,
        strategy_returns,
        cum_strategy,
        trade_hits,
        trade_count,
        entry_trade,
        exit_trade,
        entry_price,
        exit_price,
        completed,
    )


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
    df.loc[:, "entry_trade"] = 0
    df.loc[:, "exit_trade"] = 0
    df.loc[:, "entry_price"] = np.nan
    df.loc[:, "exit_price"] = np.nan

    trade_hits = 0
    trade_count = 0
    cum_return: float = 1.0
    completed_all: list[float] = []

    if pooled_mode:
        # Compute exits (TP/SL) strictly within each symbol to avoid cross-symbol lookahead.
        symbols = [s for s in df["symbol"].dropna().unique()]

        for sym in symbols:
            sub = df.loc[df["symbol"] == sym]
            sub_sorted = _stable_sort_by_timestamp(sub, ts_col) if ts_col else sub

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
                entry_trade_arr,
                exit_trade_arr,
                entry_price_arr,
                exit_price_arr,
                completed,
            ) = _backtest_single_series(
                close=close,
                high=high,
                low=low,
                pred=pred,
                pred_col_threshold=float(threshold),
            )

            df.loc[sub_sorted.index, "signal"] = signal_arr
            df.loc[sub_sorted.index, "strategy_return"] = strat_arr
            df.loc[sub_sorted.index, "entry_trade"] = entry_trade_arr
            df.loc[sub_sorted.index, "exit_trade"] = exit_trade_arr
            df.loc[sub_sorted.index, "entry_price"] = entry_price_arr
            df.loc[sub_sorted.index, "exit_price"] = exit_price_arr

            trade_hits += hits
            trade_count += trades
            completed_all.extend(completed.tolist())

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
        sub_sorted = _stable_sort_by_timestamp(df, ts_col) if ts_col else df
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
            entry_trade_arr,
            exit_trade_arr,
            entry_price_arr,
            exit_price_arr,
            completed,
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
        df.loc[sub_sorted.index, "entry_trade"] = entry_trade_arr
        df.loc[sub_sorted.index, "exit_trade"] = exit_trade_arr
        df.loc[sub_sorted.index, "entry_price"] = entry_price_arr
        df.loc[sub_sorted.index, "exit_price"] = exit_price_arr
        trade_hits += hits
        trade_count += trades
        cum_return = float(cum_arr[-1]) if len(cum_arr) else 1.0
        completed_all = completed.tolist()

    # simple market benchmark
    df.loc[:, "market_return"] = df["close"].pct_change().fillna(0)
    df.loc[:, "cum_market_return"] = (1 + df["market_return"]).cumprod()

    _mask_trade_prices(df)

    directional_accuracy = (trade_hits / trade_count) * 100 if trade_count > 0 else 0.0

    ext = _extended_trade_metrics(
        df,
        np.asarray(completed_all, dtype=float),
    )

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
