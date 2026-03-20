from sqlalchemy import text


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
