from sqlalchemy.orm import sessionmaker

from constants import SUBSCRIPTIONS
from data_pipeline.ingestion.load_stock_data import (
    batch_iterator,
    record_stream,
    upsert_batch,
)
from database.connection import engine

Session = sessionmaker(bind=engine)


def run_ingestion(symbols: list[str], batch_size: int = 500):
    print("[START] ingestion")
    with Session() as session:
        stream = record_stream(symbols, session)

        for batch in batch_iterator(stream, batch_size):
            print(f"[BATCH] size={len(batch)}")
            upsert_batch(batch)

    print("[DONE] ingestion")


if __name__ == "__main__":
    symbols = SUBSCRIPTIONS
    run_ingestion(symbols)
