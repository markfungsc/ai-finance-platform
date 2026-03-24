import pandas as pd
from sqlalchemy import text

from database.connection import engine

BATCH_SIZE = 5000


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
        query = text("""
            SELECT symbol, timestamp, close
            FROM clean_stock_prices
            WHERE symbol = :symbol
              AND timestamp >= :start - INTERVAL '20 days'
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
            return_1d, return_5d,
            sma_5, sma_10, ema_10,
            volatility_5,
            lag_1, lag_2
        )
        VALUES (
            :symbol, :timestamp, :close,
            :return_1d, :return_5d,
            :sma_5, :sma_10, :ema_10,
            :volatility_5,
            :lag_1, :lag_2
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
