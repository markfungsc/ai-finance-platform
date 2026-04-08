"""News sentiment features (optional FinBERT + yfinance headlines)."""

from ml.sentiment.attach import SENTIMENT_Z_COL, attach_sentiment_features
from ml.sentiment.paths import default_sentiment_cache_path

__all__ = [
    "SENTIMENT_Z_COL",
    "attach_sentiment_features",
    "default_sentiment_cache_path",
]
