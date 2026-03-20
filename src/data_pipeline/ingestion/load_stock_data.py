import time
from datetime import timedelta
from itertools import islice

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from data_pipeline.ingestion.fetch_stock_price import fetch_stock_price
from database.connection import engine
from database.queries import get_latest_timestamp


def fetch_records(symbol: str, session):
    """Fetch dataframe and yield record dicts"""
    last_timestamp = get_latest_timestamp(session, symbol)
    if last_timestamp:
        start_date = (last_timestamp + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = None
    df = fetch_stock_price(symbol, start_date=start_date)

    for record in df.to_dict(orient="records"):
        yield record


def record_stream(symbols: list[str], session):
    """Create a generator of records across all symbols"""
    time.sleep(0.2)
    for symbol in symbols:
        yield from fetch_records(symbol, session)


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

        ON CONFLICT (symbol, timestamp) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(insert_query, records)


def load_stock_data_batch(symbols: list[str], session, batch_size: int = 1000):
    stream = record_stream(symbols, session)
    for batch in batch_iterator(stream, batch_size):
        upsert_batch(batch)


Session = sessionmaker(bind=engine)

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    batch_size = 1000

    with Session() as session:
        load_stock_data_batch(symbols, session, batch_size)

    print("Stock data loaded")
