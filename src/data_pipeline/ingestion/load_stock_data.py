import time
from datetime import timedelta
from itertools import islice

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from data_pipeline.ingestion.fetch_stock_price import fetch_stock_price
from database.connection import engine
from database.queries import get_latest_timestamp


def fetch_records(symbol: str, session, backfill: bool = False):
    """Fetch dataframe and yield record dicts"""
    end_date = None
    if backfill:
        start_date = None
    else:
        last_timestamp = get_latest_timestamp(session, symbol)
        if last_timestamp:
            last_ts = pd.Timestamp(last_timestamp)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            else:
                last_ts = last_ts.tz_convert("UTC")
            start_dt = last_ts.date() + timedelta(days=1)
            today_utc = pd.Timestamp.now(tz="UTC").date()
            if start_dt > today_utc:
                return
            start_date = start_dt.isoformat()
            # yfinance end is exclusive; use tomorrow to include today's session.
            end_date = (today_utc + timedelta(days=1)).isoformat()
        else:
            start_date = None
    df = fetch_stock_price(symbol, start_date=start_date, end_date=end_date)

    for record in df.to_dict(orient="records"):
        yield record


def record_stream(symbols: list[str], session, backfill: bool = False):
    """Create a generator of records across all symbols"""
    time.sleep(0.2)
    for symbol in symbols:
        yield from fetch_records(symbol, session, backfill)


def batch_iterator(iterator, batch_size: int):
    """Yield batches from iterator"""
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def upsert_batch(records: list[dict]):
    """Insert batch into database"""
    insert_query = text("""
        INSERT INTO raw_stock_prices (symbol, timestamp, open, high, low, close, volume)
        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)

        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """)
    with engine.begin() as conn:
        conn.execute(insert_query, records)


def load_stock_data_batch(
    symbols: list[str], session, batch_size: int = 1000, backfill: bool = False
):
    stream = record_stream(symbols, session, backfill)
    for batch in batch_iterator(stream, batch_size):
        upsert_batch(batch)


Session = sessionmaker(bind=engine)

if __name__ == "__main__":
    # symbols = SUBSCRIPTIONS
    symbols = ["QQQ", "SPY", "^VIX"]
    batch_size = 1000

    with Session() as session:
        load_stock_data_batch(symbols, session, batch_size, True)

    print("Stock data loaded")
