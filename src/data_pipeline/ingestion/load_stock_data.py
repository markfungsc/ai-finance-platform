from sqlalchemy import text
from database.connection import engine

from data_pipeline.ingestion.fetch_stock_price import fetch_stock_price

from itertools import islice
import time

def fetch_records(symbol: str, period: str):
    """Fetch dataframe and yield record dicts"""
    df = fetch_stock_price(symbol, period)

    for record in df.to_dict(orient="records"):
        yield record

def record_stream(symbols: list[str], period: str):
    """Create a generator of records across all symbols"""
    time.sleep(0.2)
    for symbol in symbols:
        yield from fetch_records(symbol, period)

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
        INSERT INTO stock_prices (symbol, timestamp, open, high, low, close, volume)
        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)

        ON CONFLICT (symbol, timestamp) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(insert_query, records)

def load_stock_data_batch(symbols: list[str], period: str = "1y", batch_size: int = 1000):
    stream = record_stream(symbols, period)
    for batch in batch_iterator(stream, batch_size):
        upsert_batch(batch)

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    period = "1y"
    batch_size = 1000
    load_stock_data_batch(symbols, period, batch_size)
    print("Stock data loaded")