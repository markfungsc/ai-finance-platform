import pandas as pd
from sqlalchemy import text

from database.connection import engine

BATCH_SIZE = 5000

# Non-key columns on stock_features; NULL in any of these means a partial / pre-migration row.
STOCK_FEATURES_VALUE_COLUMNS = (
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
)


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
            SELECT symbol, timestamp, close
            FROM clean_stock_prices
            WHERE symbol = :symbol
              AND timestamp >= :start - INTERVAL '400 days'
            ORDER BY timestamp
        """)
        params = {"symbol": symbol, "start": start}
    else:
        query = text("""
            SELECT symbol, timestamp, close
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
            symbol, timestamp, close,
            return_1d, return_5d, return_10d, return_20d,
            sma_5, sma_10, sma_20, sma_50, sma_100,
            ema_10, ema_20, ema_50, ema_100,
            volatility_5, volatility_10, volatility_20, volatility_50, volatility_100,
            lag_1, lag_2
        )
        VALUES (
            :symbol, :timestamp, :close,
            :return_1d, :return_5d, :return_10d, :return_20d,
            :sma_5, :sma_10, :sma_20, :sma_50, :sma_100,
            :ema_10, :ema_20, :ema_50, :ema_100,
            :volatility_5, :volatility_10, :volatility_20, :volatility_50, :volatility_100,
            :lag_1, :lag_2
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
            symbol, timestamp, close_z,
            return_1d_z, return_5d_z, return_10d_z, return_20d_z,
            sma_5_z, sma_10_z, sma_20_z, sma_50_z, sma_100_z,
            ema_10_z, ema_20_z, ema_50_z, ema_100_z,
            volatility_5_z, volatility_10_z, volatility_20_z, volatility_50_z, volatility_100_z,
            lag_1_z, lag_2_z
        )
        VALUES (
            :symbol, :timestamp, :close_z,
            :return_1d_z, :return_5d_z, :return_10d_z, :return_20d_z,
            :sma_5_z, :sma_10_z, :sma_20_z, :sma_50_z, :sma_100_z,
            :ema_10_z, :ema_20_z, :ema_50_z, :ema_100_z,
            :volatility_5_z, :volatility_10_z, :volatility_20_z, :volatility_50_z, :volatility_100_z,
            :lag_1_z, :lag_2_z
        )
        ON CONFLICT (symbol, timestamp) DO NOTHING
    """)

    with engine.begin() as conn:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            conn.execute(query, batch)


def fetch_features(symbol: str):
    query = text("""
        SELECT *
        FROM stock_features
        WHERE symbol = :symbol
        ORDER BY timestamp
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
