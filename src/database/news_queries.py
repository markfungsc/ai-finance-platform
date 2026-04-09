"""Postgres access for news medallion + gold daily sentiment."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import text

from database.connection import engine


def fetch_clean_news_for_rollup() -> pd.DataFrame:
    """Return rows needed for daily sentiment rollups."""
    q = text("""
        SELECT symbol, published_at, finbert_scalar
        FROM clean_news_articles
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn)


def fetch_feature_days_for_symbol(symbol: str) -> pd.DataFrame:
    """Return distinct UTC feature dates for one symbol."""
    q = text("""
        SELECT DISTINCT (timestamp AT TIME ZONE 'UTC')::date AS as_of_date
        FROM stock_features_zscore
        WHERE symbol = :symbol
        ORDER BY as_of_date
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn, params={"symbol": symbol.upper()})


def fetch_clean_news_text_for_embedding(symbol: str) -> pd.DataFrame:
    """Return rows needed to embed clean news into vector stores."""
    q = text("""
        SELECT id, title, summary
        FROM clean_news_articles
        WHERE symbol = :symbol
        ORDER BY id
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn, params={"symbol": symbol.upper()})


def fetch_clean_news_for_asof(symbols: list[str], ts_min, ts_max) -> pd.DataFrame:
    """Fetch raw clean-news rows suitable for timestamp as-of joins."""
    if not symbols or ts_min is None or ts_max is None:
        return pd.DataFrame()
    q = text(
        """
        SELECT symbol, published_at, finbert_scalar
        FROM clean_news_articles
        WHERE symbol = ANY(:symbols)
          AND published_at <= :ts_max
          AND published_at >= :ts_min - INTERVAL '90 days'
        ORDER BY symbol, published_at
        """
    )
    params = {
        "symbols": [s.upper() for s in symbols],
        "ts_min": ts_min,
        "ts_max": ts_max,
    }
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn, params=params)


def upsert_raw_news(
    source: str,
    external_id: str,
    content_sha256: str,
    raw_payload: dict[str, Any],
) -> int:
    """Insert or update bronze row; return raw id."""
    q = text("""
        INSERT INTO raw_news_articles (source, external_id, content_sha256, raw_payload)
        VALUES (:source, :external_id, :sha, CAST(:payload AS jsonb))
        ON CONFLICT (source, external_id) DO UPDATE SET
            content_sha256 = EXCLUDED.content_sha256,
            raw_payload = EXCLUDED.raw_payload
        RETURNING id
    """)
    payload = json.dumps(raw_payload)
    with engine.begin() as conn:
        row = conn.execute(
            q,
            {
                "source": source,
                "external_id": external_id,
                "sha": content_sha256,
                "payload": payload,
            },
        ).one()
        return int(row[0])


def insert_clean_article(
    raw_news_id: int | None,
    symbol: str,
    url: str | None,
    title: str,
    summary: str,
    published_at,
    content_sha256: str,
    finbert_scalar: float | None = None,
) -> int:
    """Insert silver row if new; return clean article id (existing or new)."""
    ins = text("""
        INSERT INTO clean_news_articles (
            raw_news_id, symbol, url, title, summary, published_at, content_sha256, finbert_scalar
        )
        VALUES (
            :raw_id, :symbol, :url, :title, :summary, :published_at, :sha, :finbert
        )
        ON CONFLICT (content_sha256) DO NOTHING
        RETURNING id
    """)
    sel = text("SELECT id FROM clean_news_articles WHERE content_sha256 = :sha")
    with engine.begin() as conn:
        r = conn.execute(
            ins,
            {
                "raw_id": raw_news_id,
                "symbol": symbol,
                "url": url,
                "title": title[:8000] if title else "",
                "summary": summary[:16000] if summary else "",
                "published_at": published_at,
                "sha": content_sha256,
                "finbert": finbert_scalar,
            },
        ).first()
        if r is not None:
            return int(r[0])
        row = conn.execute(sel, {"sha": content_sha256}).one()
        return int(row[0])


def fetch_daily_symbol_sentiment_df() -> pd.DataFrame:
    """All gold rows for merging onto (symbol, UTC calendar day)."""
    q = text("""
        SELECT
            symbol,
            as_of_date,
            news_sentiment_mean_z,
            sentiment_1h,
            sentiment_24h,
            sentiment_3d,
            news_volume,
            sentiment_volatility
        FROM daily_symbol_sentiment
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn)


def upsert_daily_symbol_sentiment_rows(rows: list[dict]) -> None:
    """rows: z=news_sentiment_mean_z; horizon cols are rolling z; news_volume is rolling z of log1p(raw 24h count)."""
    if not rows:
        return
    q = text("""
        INSERT INTO daily_symbol_sentiment (
            symbol,
            as_of_date,
            news_sentiment_mean_z,
            sentiment_1h,
            sentiment_24h,
            sentiment_3d,
            news_volume,
            sentiment_volatility,
            article_count
        )
        VALUES (
            :symbol,
            :as_of_date,
            :z,
            :sentiment_1h,
            :sentiment_24h,
            :sentiment_3d,
            :news_volume,
            :sentiment_volatility,
            :article_count
        )
        ON CONFLICT (symbol, as_of_date) DO UPDATE SET
            news_sentiment_mean_z = EXCLUDED.news_sentiment_mean_z,
            sentiment_1h = EXCLUDED.sentiment_1h,
            sentiment_24h = EXCLUDED.sentiment_24h,
            sentiment_3d = EXCLUDED.sentiment_3d,
            news_volume = EXCLUDED.news_volume,
            sentiment_volatility = EXCLUDED.sentiment_volatility,
            article_count = EXCLUDED.article_count,
            updated_at = NOW()
    """)
    with engine.begin() as conn:
        for r in rows:
            conn.execute(q, r)


def delete_daily_symbol_sentiment_for_symbol(symbol: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM daily_symbol_sentiment WHERE symbol = :s"),
            {"s": symbol},
        )
