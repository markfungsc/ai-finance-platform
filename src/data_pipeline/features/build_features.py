import pandas as pd

from database.queries import (
    fetch_clean_data,
    get_latest_feature_timestamp,
    upsert_features,
)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by="timestamp")

    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)

    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["ema_10"] = df["close"].ewm(span=10).mean()

    df["volatility_5"] = df["return_1d"].rolling(5).std()

    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)

    return df


def run_feature_pipeline(symbol: str):
    print(f"Building features for {symbol}")

    latest_ts = get_latest_feature_timestamp(symbol)
    if latest_ts:
        print(f"[INCREMENTAL] from {latest_ts}")
        df = fetch_clean_data(symbol, latest_ts)
    else:
        print("[FULL BACKFILL]")
        df = fetch_clean_data(symbol)

    df = compute_features(df)
    df = df.dropna()

    records = df.to_dict(orient="records")
    upsert_features(records)

    print(f"Finished features for {symbol}")


if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    for symbol in symbols:
        run_feature_pipeline(symbol)
