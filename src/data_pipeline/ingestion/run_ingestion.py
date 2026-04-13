from sqlalchemy.orm import sessionmaker

from data_pipeline.ingestion.load_stock_data import (
    batch_iterator,
    record_stream,
    upsert_batch,
)
from database.connection import engine
from universe.resolve import resolve_ingestion_symbols

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
    run_ingestion(resolve_ingestion_symbols())
