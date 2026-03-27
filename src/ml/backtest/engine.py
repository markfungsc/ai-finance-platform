import pandas as pd

from constants import MAX_HOLD_DAYS, SL_PCT, THRESHOLD, TP_PCT


def basic_backtest(
    df: pd.DataFrame,
    pred_col: str = "prob_trade_success",
    threshold: float = THRESHOLD,
    market_cum_return_override: float | None = None,
):
    # Force an owned frame to avoid pandas CoW/chained-assignment warnings.
    df = df.reset_index(drop=True).copy(deep=True)
    df.loc[:, "signal"] = (df[pred_col] > threshold).astype(int)

    strategy_returns = []
    trade_hits = 0
    trade_count = 0

    for i in range(len(df)):
        if df.iloc[i]["signal"] == 0:
            strategy_returns.append(0.0)
            continue

        entry_price = df.iloc[i]["close"]
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)

        trade_return = 0.0
        trade_result = 0

        for j in range(1, MAX_HOLD_DAYS + 1):
            if i + j >= len(df):
                break

            high = df.iloc[i + j]["high"]
            low = df.iloc[i + j]["low"]

            if high >= tp_price:
                trade_return = TP_PCT
                trade_result = 1
                break

            if low <= sl_price:
                trade_return = -SL_PCT
                trade_result = 0
                break

        strategy_returns.append(trade_return)

        trade_count += 1
        if trade_result == 1:
            trade_hits += 1

    df.loc[:, "strategy_return"] = strategy_returns

    # cumulative strategy return
    df.loc[:, "cum_strategy_return"] = (1 + df["strategy_return"]).cumprod()

    # simple market benchmark
    df.loc[:, "market_return"] = df["close"].pct_change().fillna(0)
    df.loc[:, "cum_market_return"] = (1 + df["market_return"]).cumprod()

    directional_accuracy = (trade_hits / trade_count) * 100 if trade_count > 0 else 0.0

    metrics = {
        "cum_return": float(df["cum_strategy_return"].iloc[-1]),
        "cum_market_return": float(
            market_cum_return_override
            if market_cum_return_override is not None
            else df["cum_market_return"].iloc[-1]
        ),
        "directional_accuracy": float(directional_accuracy),
        "strategy_directional_hits": trade_hits,
        "strategy_trade_count": trade_count,
    }

    return df, metrics
