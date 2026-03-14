import yfinance as yf
import pandas as pd

def fetch_stock_price(symbol: str, period: str = "1d") -> pd.DataFrame:
    """
    Fetch the stock price history from Yahoo Finance
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    df= df.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    df["symbol"] = symbol
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

    return df

if __name__ == "__main__":
    data = fetch_stock_price("AAPL")
    print(data.head())
    print(data.tail())