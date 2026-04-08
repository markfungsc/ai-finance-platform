from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_sentiment_cache_path() -> Path:
    return repo_root() / "data" / "sentiment" / "daily_sentiment.parquet"
