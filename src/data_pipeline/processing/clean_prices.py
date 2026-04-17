from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from database.connection import engine
from database.queries import get_all_symbols_from_raw_stock_prices

Session = sessionmaker(bind=engine)

# Indices / proxies where volume is zero or missing; OHLC still valid.
_SYMBOLS_VOLUME_NOT_REQUIRED = frozenset({"^VIX"})


def _volume_required(symbol: str) -> bool:
    return symbol.upper() not in _SYMBOLS_VOLUME_NOT_REQUIRED


def clean_prices(session, symbol: str):
    """
    Clean raw stock data for a given symbol:
    - Deduplicate
    - Remove missing or invalid price/volume rows
    - Ensure clean_stock_prices table is ready for feature engineering
    """
    # Count before cleaning
    count_before = session.execute(
        text("""
        SELECT COUNT(*) FROM clean_stock_prices WHERE symbol = :symbol
    """),
        {"symbol": symbol},
    ).scalar()

    # Step 1: Insert deduplicated data from raw_stock_prices
    session.execute(
        text("""
        INSERT INTO clean_stock_prices (symbol, timestamp, open, high, low, close, volume)
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM raw_stock_prices
        WHERE symbol = :symbol
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """),
        {"symbol": symbol},
    )

    # Step 2: Delete rows with missing values
    if _volume_required(symbol):
        session.execute(
            text("""
            DELETE FROM clean_stock_prices
            WHERE symbol = :symbol
              AND (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL)
        """),
            {"symbol": symbol},
        )
    else:
        session.execute(
            text("""
            DELETE FROM clean_stock_prices
            WHERE symbol = :symbol
              AND (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL)
        """),
            {"symbol": symbol},
        )

    # Step 3: Delete rows with invalid price or volume
    if _volume_required(symbol):
        session.execute(
            text("""
            DELETE FROM clean_stock_prices
            WHERE symbol = :symbol
              AND (open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 OR volume <= 0)
        """),
            {"symbol": symbol},
        )
    else:
        session.execute(
            text("""
            DELETE FROM clean_stock_prices
            WHERE symbol = :symbol
              AND (open <= 0 OR high <= 0 OR low <= 0 OR close <= 0)
        """),
            {"symbol": symbol},
        )

    session.commit()

    # Count after cleaning
    count_after = session.execute(
        text("""
        SELECT COUNT(*) FROM clean_stock_prices WHERE symbol = :symbol
    """),
        {"symbol": symbol},
    ).scalar()

    print(f"[CLEAN] {symbol} | before: {count_before} rows, after: {count_after} rows")


def run_clean_prices(symbols: list[str]) -> None:
    """Run :func:`clean_prices` for each symbol (same lifecycle as :func:`run_ingestion`)."""
    print("[START] clean_prices")
    with Session() as session:
        for symbol in symbols:
            clean_prices(session, symbol)
    print("[DONE] clean_prices")


if __name__ == "__main__":
    with Session() as session:
        syms = get_all_symbols_from_raw_stock_prices(session)
    run_clean_prices(syms)
