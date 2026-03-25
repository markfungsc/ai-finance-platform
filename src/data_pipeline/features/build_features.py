import pandas as pd

from database.queries import (
    STOCK_FEATURES_VALUE_COLUMNS,
    delete_incomplete_stock_feature_rows,
    fetch_clean_data,
    get_latest_feature_timestamp,
    upsert_features,
    upsert_features_z,
)

# Rows must be complete on these columns before upsert (matches stock_features INSERT).
_STOCK_FEATURES_COLUMNS = [
    "symbol",
    "timestamp",
    "close",
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_100",
    "ema_10",
    "ema_20",
    "ema_50",
    "ema_100",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_50",
    "volatility_100",
    "lag_1",
    "lag_2",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by="timestamp")

    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)

    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_100"] = df["close"].rolling(100).mean()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_100"] = df["close"].ewm(span=100).mean()

    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["volatility_50"] = df["return_1d"].rolling(50).std()
    df["volatility_100"] = df["return_1d"].rolling(100).std()

    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)

    return df


def rowwise_cross_sectional_zscore(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Per row (one symbol, one timestamp): z = (feature − row_mean) / row_std across ``feature_cols``.

    Mean and standard deviation are taken **across the feature columns** for that row only,
    not across other symbols or other dates.
    """
    out = df.copy()
    X = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    row_mean = X.mean(axis=1)
    row_std = X.std(axis=1, ddof=0)
    row_std = row_std.replace(0, pd.NA).fillna(1.0)
    for col in feature_cols:
        out[f"{col}_z"] = (X[col] - row_mean) / row_std
    return out


def _z_column_names() -> list[str]:
    return [f"{c}_z" for c in STOCK_FEATURES_VALUE_COLUMNS]


def run_feature_pipeline(symbol: str, backfill: bool = False) -> None:
    print(f"Building features for {symbol}")

    if backfill:
        print("[Manual FULL BACKFILL]")
        df = fetch_clean_data(symbol, None)
    else:
        latest_ts = get_latest_feature_timestamp(symbol)
        if latest_ts:
            print(f"[INCREMENTAL] from {latest_ts}")
            df = fetch_clean_data(symbol, latest_ts)
        else:
            print("[FULL BACKFILL]")
            df = fetch_clean_data(symbol, None)

    df = compute_features(df)
    value_cols = list(STOCK_FEATURES_VALUE_COLUMNS)
    df = df.dropna(subset=value_cols)
    if df.empty:
        print(f"No complete rows for {symbol} after compute; skipping upsert")
        return

    df = rowwise_cross_sectional_zscore(df, value_cols)
    z_cols = _z_column_names()
    df = df.dropna()
    if df.empty:
        print(f"No complete rows for {symbol} after z-score; skipping upsert")
        return

    removed = delete_incomplete_stock_feature_rows(symbol)
    if removed:
        print(f"Removed {removed} incomplete stock_features row(s)")

    base_records = df[_STOCK_FEATURES_COLUMNS].to_dict(orient="records")
    z_records = df[["symbol", "timestamp", *z_cols]].to_dict(orient="records")
    upsert_features(base_records)
    upsert_features_z(z_records)

    print(f"Finished features for {symbol}")


if __name__ == "__main__":
    # symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    symbols = ["AAPL"]
    for sym in symbols:
        run_feature_pipeline(sym, backfill=True)
