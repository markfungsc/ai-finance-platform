from sqlalchemy.orm import sessionmaker

from data_pipeline.ingestion.load_stock_data import (
    batch_iterator,
    record_stream,
    upsert_batch,
)
from database.connection import engine
from universe.resolve import resolve_ingestion_symbols

Session = sessionmaker(bind=engine)


def run_ingestion(
    symbols: list[str],
    batch_size: int = 500,
    *,
    backfill: bool = False,
    BACKFILL: bool | None = None,
):
    """Ingest Yahoo prices into ``raw_stock_prices``.

    Use ``backfill=True`` or ``BACKFILL=True`` (same meaning). Keyword-only after ``batch_size``.

    - **Backfill:** full history per symbol (``period=max`` in ``fetch_stock_price``).
    - **Incremental:** from the day after ``MAX(timestamp)`` in ``raw_stock_prices`` per symbol.
    """
    use_backfill = BACKFILL if BACKFILL is not None else backfill
    mode = "BACKFILL" if use_backfill else "incremental"
    print(f"[START] ingestion ({mode})")
    with Session() as session:
        stream = record_stream(symbols, session, backfill=use_backfill)

        for batch in batch_iterator(stream, batch_size):
            print(f"[BATCH] size={len(batch)}")
            upsert_batch(batch)

    print("[DONE] ingestion")


if __name__ == "__main__":
    run_ingestion(resolve_ingestion_symbols())
