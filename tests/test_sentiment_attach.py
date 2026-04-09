"""Unit tests for ml.sentiment.attach.attach_sentiment_features."""

from pathlib import Path

import pandas as pd
import pytest

from ml.sentiment.attach import (
    SENTIMENT_ALL_COLS,
    SENTIMENT_DAILY_DUAL_COLS,
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
            "published_at": [pd.Timestamp("2020-01-02 09:00:00", tz="UTC")],
            "finbert_scalar": [1.5],
        }
    )
    with patch(
        "database.news_queries.fetch_clean_news_for_asof", return_value=db_frame
    ):
        out = attach_sentiment_features(df, cache_path=cache)
    _assert_all_sentiment_cols(out)
    assert out[SENTIMENT_Z_COL].iloc[0] == pytest.approx(1.5)
    assert out["news_volume"].iloc[0] == pytest.approx(1.0)


def test_asof_attach_excludes_post_bar_news(monkeypatch):
    # Bar at 10:00 should not include article published at 12:00 same day.
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": [
                pd.Timestamp("2020-01-02 10:00:00", tz="UTC"),
                pd.Timestamp("2020-01-02 13:00:00", tz="UTC"),
            ],
        }
    )
    articles = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "SPY"],
            "published_at": [
                pd.Timestamp("2020-01-02 09:00:00", tz="UTC"),
                pd.Timestamp("2020-01-02 12:00:00", tz="UTC"),
                pd.Timestamp("2020-01-02 08:00:00", tz="UTC"),
            ],
            "finbert_scalar": [1.0, -1.0, 0.5],
        }
    )

    from unittest.mock import patch

    with patch(
        "database.news_queries.fetch_clean_news_for_asof", return_value=articles
    ):
        out = attach_sentiment_features(bars)

    _assert_all_sentiment_cols(out)
    # At 10:00 only 09:00 article is visible.
    assert out["news_volume"].iloc[0] == pytest.approx(1.0)
    assert out["sentiment_24h"].iloc[0] == pytest.approx(1.0)
    # At 13:00 both are visible.
    assert out["news_volume"].iloc[1] == pytest.approx(2.0)
    assert out["sentiment_24h"].iloc[1] == pytest.approx(0.0)
    # Daily dual-stream columns are present and populated.
    for c in SENTIMENT_DAILY_DUAL_COLS:
        assert c in out.columns
    assert out["sym_sentiment_d1"].iloc[0] == pytest.approx(1.0)
    assert out["spy_sentiment_d1"].iloc[0] == pytest.approx(0.5)


def test_asof_attach_uses_spy_when_symbol_is_missing(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    bars = pd.DataFrame(
        {
            "symbol": ["MSFT"],
            "timestamp": [pd.Timestamp("2020-01-02 10:00:00", tz="UTC")],
        }
    )
    articles = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "published_at": [pd.Timestamp("2020-01-02 09:00:00", tz="UTC")],
            "finbert_scalar": [0.6],
        }
    )
    from unittest.mock import patch

    with patch(
        "database.news_queries.fetch_clean_news_for_asof", return_value=articles
    ):
        out = attach_sentiment_features(bars)
    _assert_all_sentiment_cols(out)
    assert out["sym_sentiment_d1"].iloc[0] == pytest.approx(0.0)
    assert out["spy_sentiment_d1"].iloc[0] == pytest.approx(0.6)
    assert out["spy_news_volume_d1"].iloc[0] == pytest.approx(1.0)
