"""Unit tests for ml.sentiment.attach.attach_sentiment_features."""

from pathlib import Path

import pandas as pd
import pytest

from ml.sentiment.attach import (
    SENTIMENT_ALL_COLS,
    SENTIMENT_Z_COL,
    attach_sentiment_features,
)


def _assert_all_sentiment_cols(out: pd.DataFrame) -> None:
    for c in SENTIMENT_ALL_COLS:
        assert c in out.columns


def test_missing_cache_fills_neutral(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        }
    )
    out = attach_sentiment_features(
        df, cache_path=Path("/nonexistent/sentiment.parquet")
    )
    _assert_all_sentiment_cols(out)
    for c in SENTIMENT_ALL_COLS:
        assert (out[c] == 0.0).all()


def test_cache_merge_by_day(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    cache = tmp_path / "sent.parquet"
    day = pd.Timestamp("2020-01-02", tz="UTC")
    pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "timestamp": [day],
            SENTIMENT_Z_COL: [1.5],
            "sentiment_24h": [0.7],
            "news_volume": [5],
        }
    ).to_parquet(cache)

    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": [
                pd.Timestamp("2020-01-02 15:30:00", tz="UTC"),
                pd.Timestamp("2020-01-03", tz="UTC"),
            ],
        }
    )
    out = attach_sentiment_features(df, cache_path=cache)
    _assert_all_sentiment_cols(out)
    assert out[SENTIMENT_Z_COL].iloc[0] == pytest.approx(1.5)
    assert out["sentiment_24h"].iloc[0] == pytest.approx(0.7)
    assert out["news_volume"].iloc[0] == pytest.approx(5.0)
    assert out[SENTIMENT_Z_COL].iloc[1] == pytest.approx(0.0)


def test_missing_timestamp_symbol_adds_column(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    df = pd.DataFrame({"x": [1]})
    out = attach_sentiment_features(df)
    _assert_all_sentiment_cols(out)
    assert out[SENTIMENT_Z_COL].iloc[0] == pytest.approx(0.0)


def test_gold_table_from_db_takes_priority_over_parquet(tmp_path, monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    cache = tmp_path / "sent.parquet"
    pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "timestamp": [pd.Timestamp("2020-01-02", tz="UTC")],
            SENTIMENT_Z_COL: [99.0],
        }
    ).to_parquet(cache)

    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "timestamp": [pd.Timestamp("2020-01-02 12:00:00", tz="UTC")],
        }
    )
    db_frame = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "as_of_date": [__import__("datetime").date(2020, 1, 2)],
            SENTIMENT_Z_COL: [1.5],
            "sentiment_1h": [0.2],
            "sentiment_24h": [0.9],
            "sentiment_3d": [0.3],
            "news_volume": [8],
            "sentiment_volatility": [0.4],
        }
    )
    with patch(
        "database.news_queries.fetch_daily_symbol_sentiment_df", return_value=db_frame
    ):
        out = attach_sentiment_features(df, cache_path=cache)
    _assert_all_sentiment_cols(out)
    assert out[SENTIMENT_Z_COL].iloc[0] == pytest.approx(1.5)
    assert out["news_volume"].iloc[0] == pytest.approx(8.0)
