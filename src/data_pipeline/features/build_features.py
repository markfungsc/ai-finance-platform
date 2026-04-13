import os

import pandas as pd
import ta

from database.queries import (
    STOCK_FEATURES_VALUE_COLUMNS,
    delete_incomplete_stock_feature_rows,
    delete_incomplete_stock_feature_zscore_rows,
    fetch_clean_data,
    get_features_count,
    get_features_zscore_count,
    get_latest_feature_timestamp,
    upsert_features,
    upsert_features_z,
)
from universe.resolve import resolve_ingestion_symbols

# Rows must be complete on these columns before upsert (matches stock_features INSERT).
_STOCK_FEATURES_COLUMNS = ["symbol", "timestamp", *STOCK_FEATURES_VALUE_COLUMNS]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort then materialize; use `.loc[:, col] =` for new columns (pandas 3 CoW).
    df = df.sort_values(by="timestamp").reset_index(drop=True).copy()

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[:, "return_1d"] = df["close"].pct_change(1, fill_method=None)
    df.loc[:, "return_5d"] = df["close"].pct_change(5, fill_method=None)
    df.loc[:, "return_10d"] = df["close"].pct_change(10, fill_method=None)
    df.loc[:, "return_20d"] = df["close"].pct_change(20, fill_method=None)

    df.loc[:, "sma_5"] = df["close"].rolling(5).mean()
    df.loc[:, "sma_10"] = df["close"].rolling(10).mean()
    df.loc[:, "sma_20"] = df["close"].rolling(20).mean()
    df.loc[:, "sma_50"] = df["close"].rolling(50).mean()
    df.loc[:, "sma_100"] = df["close"].rolling(100).mean()
    df.loc[:, "sma_200"] = df["close"].rolling(200).mean()
    df.loc[:, "ema_10"] = df["close"].ewm(span=10).mean()
    df.loc[:, "ema_20"] = df["close"].ewm(span=20).mean()
    df.loc[:, "ema_50"] = df["close"].ewm(span=50).mean()
    df.loc[:, "ema_100"] = df["close"].ewm(span=100).mean()
    df.loc[:, "ema_200"] = df["close"].ewm(span=200).mean()

    df.loc[:, "ema_trend_bull"] = df["ema_20"] - df["ema_50"]
    df.loc[:, "ema_slope_20"] = df["ema_20"].pct_change(20, fill_method=None)

    df.loc[:, "volatility_5"] = df["return_1d"].rolling(5).std()
    df.loc[:, "volatility_10"] = df["return_1d"].rolling(10).std()
    df.loc[:, "volatility_20"] = df["return_1d"].rolling(20).std()
    df.loc[:, "volatility_50"] = df["return_1d"].rolling(50).std()
    df.loc[:, "volatility_100"] = df["return_1d"].rolling(100).std()

    df.loc[:, "lag_1"] = df["close"].shift(1)
    df.loc[:, "lag_2"] = df["close"].shift(2)

    # Trend
    trend_mask = (df["ema_20"] > df["sma_50"]) & (df["ema_50"] > df["ema_100"])
    df.loc[:, "ema_trend_bull"] = trend_mask.astype(float)
    df.loc[:, "ema_slope_20"] = (df["ema_20"] - df["ema_20"].shift(5)) / df[
        "ema_20"
    ].shift(5)

    # Momentum
    df.loc[:, "rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df.loc[:, "rsi_21"] = ta.momentum.rsi(df["close"], window=21)
    df.loc[:, "macd"] = ta.trend.macd(df["close"], window_slow=26, window_fast=12)
    df.loc[:, "macd_signal"] = ta.trend.macd_signal(
        df["close"], window_slow=26, window_fast=12, window_sign=9
    )
    df.loc[:, "roc_5"] = ta.momentum.roc(df["close"], window=5)
    df.loc[:, "roc_10"] = ta.momentum.roc(df["close"], window=10)
    df.loc[:, "stochastic_k"] = ta.momentum.stoch(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df.loc[:, "stochastic_d"] = ta.momentum.stoch_signal(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )

    # Risk
    df.loc[:, "atr_14"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )
    df.loc[:, "volatility_ratio"] = df["volatility_20"] / df["volatility_50"]

    # Price / Structure
    df.loc[:, "close_vs_high_10"] = (df["close"] - df["high"].rolling(10).max()) / df[
        "high"
    ].rolling(10).max()
    df.loc[:, "close_vs_low_10"] = (df["close"] - df["low"].rolling(10).min()) / df[
        "low"
    ].rolling(10).min()

    # Bollinger Bands (20, 2.0)
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df.loc[:, "bb_mavg_20"] = bb.bollinger_mavg()
    df.loc[:, "bb_hband_20"] = bb.bollinger_hband()
    df.loc[:, "bb_lband_20"] = bb.bollinger_lband()
    df.loc[:, "bb_width_20"] = bb.bollinger_wband()
    df.loc[:, "bb_pband_20"] = bb.bollinger_pband()
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
    # Avoid replace+fillna downcasting deprecation; std 0 or NaN -> 1 for safe division.
    row_std = row_std.where((row_std != 0) & row_std.notna(), 1.0)
    for col in feature_cols:
        out.loc[:, f"{col}_z"] = (X[col] - row_mean) / row_std
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
    print(z_cols)
    df = df.dropna()
    if df.empty:
        print(f"No complete rows for {symbol} after z-score; skipping upsert")
        return

    removed = delete_incomplete_stock_feature_rows(symbol)
    if removed:
        print(f"Removed {removed} incomplete stock_features row(s)")

    removed_z = delete_incomplete_stock_feature_zscore_rows(symbol)
    if removed_z:
        print(f"Removed {removed} incomplete stock_features_zscore row(s)")

    base_records = df[_STOCK_FEATURES_COLUMNS].to_dict(orient="records")
    z_records = df[["symbol", "timestamp", *z_cols]].to_dict(orient="records")
    upsert_features(base_records)
    upsert_features_z(z_records)

    features_count = get_features_count()
    features_zscore_count = get_features_zscore_count()

    if features_count != features_zscore_count:
        raise ValueError("Features count and features zscore count do not match")

    print(f"Finished features for {symbol}")
    print(f"Total Features count: {features_count}")
    print(f"Total Features zscore count: {features_zscore_count}")


if __name__ == "__main__":
    backfill = os.environ.get("FEATURES_BACKFILL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    for sym in resolve_ingestion_symbols():
        run_feature_pipeline(sym, backfill=backfill)
