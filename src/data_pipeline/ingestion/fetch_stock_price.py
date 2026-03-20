import pandas as pd
import yfinance as yf


def fetch_stock_price(
    symbol: str, start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """
    Fetch the stock price history from Yahoo Finance
    """
    stock = yf.Ticker(symbol)

    if start_date:
        df = stock.history(start=start_date, end=end_date)
    else:
        df = stock.history(period="max")

    df.reset_index(inplace=True)
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["symbol"] = symbol
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

    return df


if __name__ == "__main__":
    data = fetch_stock_price("AAPL")
    print(data.head())
    print(data.tail())
