import pandas as pd
from sqlalchemy import text

from database.connection import engine
from feature_schema import BASE_FEATURE_COLUMNS, z_columns

BATCH_SIZE = 5000

# Non-key columns on stock_features; NULL in any of these means a partial / pre-migration row.
STOCK_FEATURES_VALUE_COLUMNS = ("close", *BASE_FEATURE_COLUMNS)
STOCK_FEATURES_ZSCORE_VALUE_COLUMNS = z_columns(STOCK_FEATURES_VALUE_COLUMNS)


def delete_incomplete_stock_feature_rows(symbol: str) -> int:
    """Delete rows for ``symbol`` in ``[ts_min, ts_max]`` with NULL in any feature column.

    Clears stale rows from before schema extensions so the following upsert can insert
    a full feature vector. Scoped to the batch window so incremental runs do not remove
    history outside the recomputed range.
    """
    null_or = " OR ".join(f"{c} IS NULL" for c in STOCK_FEATURES_VALUE_COLUMNS)
    query = text(f"""
        DELETE FROM stock_features
        WHERE symbol = :symbol
          AND ({null_or})
    """)

    with engine.begin() as conn:
        result = conn.execute(query, {"symbol": symbol})
        return result.rowcount or 0


def delete_incomplete_stock_feature_zscore_rows(symbol: str) -> int:
    """Delete rows for ``symbol`` in ``[ts_min, ts_max]`` with NULL in any feature column.

    Clears stale rows from before schema extensions so the following upsert can insert
    a full feature vector. Scoped to the batch window so incremental runs do not remove
    history outside the recomputed range.
    """
    null_or = " OR ".join(f"{c} IS NULL" for c in STOCK_FEATURES_ZSCORE_VALUE_COLUMNS)
    query = text(f"""
        DELETE FROM stock_features_zscore
        WHERE symbol = :symbol
          AND ({null_or})
    """)
    with engine.begin() as conn:
        result = conn.execute(query, {"symbol": symbol})
        return result.rowcount or 0


def get_latest_timestamp(session, symbol: str):
    result = session.execute(
        text("""
            SELECT MAX(timestamp) 
            FROM raw_stock_prices
            WHERE symbol = :symbol
        """),
        {"symbol": symbol},
    ).scalar()
    return result


def get_all_symbols_from_raw_stock_prices(session):
    """Return a list of all unique symbols in raw_stock_prices"""
    result = session.execute(
        text("""
        SELECT DISTINCT symbol FROM raw_stock_prices
    """)
    )
    return [row[0] for row in result.fetchall()]


def fetch_clean_data(symbol: str, start=None):
    if start:
        # Enough history for 100-bar rolling features after incremental cutoff.
        query = text("""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM clean_stock_prices
            WHERE symbol = :symbol
              AND timestamp >= :start - INTERVAL '400 days'
            ORDER BY timestamp
        """)
        params = {"symbol": symbol, "start": start}
    else:
        query = text("""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM clean_stock_prices
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        params = {"symbol": symbol}

    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def get_latest_feature_timestamp(symbol: str):
    query = text("""
        SELECT MAX(timestamp)
        FROM stock_features
        WHERE symbol = :symbol
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"symbol": symbol}).scalar()

    return result


def upsert_features(records: list[dict]):
    query = text("""
        INSERT INTO stock_features (
            symbol, timestamp, open, high, low, close, volume,
            return_1d, return_5d, return_10d, return_20d,
            sma_5, sma_10, sma_20, sma_50, sma_100,
            ema_10, ema_20, ema_50, ema_100,
            volatility_5, volatility_10, volatility_20, volatility_50, volatility_100,
            lag_1, lag_2,
            sma_200, ema_200,
            ema_trend_bull, ema_slope_20,
            rsi_14, rsi_21,
            macd, macd_signal,
            roc_5, roc_10,
            stochastic_k, stochastic_d,
            atr_14, volatility_ratio,
            close_vs_high_10, close_vs_low_10,
            bb_mavg_20, bb_hband_20, bb_lband_20, bb_width_20, bb_pband_20
        )
        VALUES (
            :symbol, :timestamp, :open, :high, :low, :close, :volume,
            :return_1d, :return_5d, :return_10d, :return_20d,
            :sma_5, :sma_10, :sma_20, :sma_50, :sma_100,
            :ema_10, :ema_20, :ema_50, :ema_100,
            :volatility_5, :volatility_10, :volatility_20, :volatility_50, :volatility_100,
            :lag_1, :lag_2,
            :sma_200, :ema_200,
            :ema_trend_bull, :ema_slope_20,
            :rsi_14, :rsi_21,
            :macd, :macd_signal,
            :roc_5, :roc_10,
            :stochastic_k, :stochastic_d,
            :atr_14, :volatility_ratio,
            :close_vs_high_10, :close_vs_low_10,
            :bb_mavg_20, :bb_hband_20, :bb_lband_20, :bb_width_20, :bb_pband_20
        )
        ON CONFLICT (symbol, timestamp) DO NOTHING
    """)

    with engine.begin() as conn:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            conn.execute(query, batch)


def upsert_features_z(records: list[dict]):
    query = text("""
        INSERT INTO stock_features_zscore (
            symbol, timestamp, open_z, high_z, low_z, close_z, volume_z,
            return_1d_z, return_5d_z, return_10d_z, return_20d_z,
            sma_5_z, sma_10_z, sma_20_z, sma_50_z, sma_100_z,
            ema_10_z, ema_20_z, ema_50_z, ema_100_z,
            volatility_5_z, volatility_10_z, volatility_20_z, volatility_50_z, volatility_100_z,
            lag_1_z, lag_2_z,
            sma_200_z, ema_200_z,
            ema_trend_bull_z, ema_slope_20_z,
            rsi_14_z, rsi_21_z,
            macd_z, macd_signal_z,
            roc_5_z, roc_10_z,
            stochastic_k_z, stochastic_d_z,
            atr_14_z, volatility_ratio_z,
            close_vs_high_10_z, close_vs_low_10_z,
            bb_mavg_20_z, bb_hband_20_z, bb_lband_20_z, bb_width_20_z, bb_pband_20_z
        )
        VALUES (
            :symbol, :timestamp, :open_z, :high_z, :low_z, :close_z, :volume_z,
            :return_1d_z, :return_5d_z, :return_10d_z, :return_20d_z,
            :sma_5_z, :sma_10_z, :sma_20_z, :sma_50_z, :sma_100_z,
            :ema_10_z, :ema_20_z, :ema_50_z, :ema_100_z,
            :volatility_5_z, :volatility_10_z, :volatility_20_z, :volatility_50_z, :volatility_100_z,
            :lag_1_z, :lag_2_z,
            :sma_200_z, :ema_200_z,
            :ema_trend_bull_z, :ema_slope_20_z,
            :rsi_14_z, :rsi_21_z,
            :macd_z, :macd_signal_z,
            :roc_5_z, :roc_10_z,
            :stochastic_k_z, :stochastic_d_z,
            :atr_14_z, :volatility_ratio_z,
            :close_vs_high_10_z, :close_vs_low_10_z,
            :bb_mavg_20_z, :bb_hband_20_z, :bb_lband_20_z, :bb_width_20_z, :bb_pband_20_z
        )
        ON CONFLICT (symbol, timestamp) DO NOTHING
    """)

    with engine.begin() as conn:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            conn.execute(query, batch)


def fetch_features(symbol: str):
    _raw_cols = {"open", "high", "low", "close", "volume"}
    _f_cols = [c for c in STOCK_FEATURES_VALUE_COLUMNS if c not in _raw_cols]
    _f_sql = ", ".join(f"f.{c}" for c in _f_cols)
    query = text(f"""
        SELECT
            p.symbol,
            p.timestamp,
            p.open,
            p.high,
            p.low,
            p.close,
            p.volume,
            {_f_sql}
        FROM clean_stock_prices p
        JOIN stock_features f ON p.symbol = f.symbol AND p.timestamp = f.timestamp
        WHERE p.symbol = :symbol
        ORDER BY p.timestamp
    """)

    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"symbol": symbol})


def fetch_features_z(symbol: str):
    query = text("""
        SELECT *
        FROM stock_features_zscore
        WHERE symbol = :symbol
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"symbol": symbol})


def get_features_count():
    query = text("""
        SELECT COUNT(*) FROM stock_features
    """)
    with engine.connect() as conn:
        return conn.execute(query).scalar()


def get_features_zscore_count():
    query = text("""
        SELECT COUNT(*) FROM stock_features_zscore
    """)
    with engine.connect() as conn:
        return conn.execute(query).scalar()
