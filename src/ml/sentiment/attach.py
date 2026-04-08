"""Join precomputed daily sentiment features onto feature frames."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from log_config import get_logger
from ml.sentiment.paths import default_sentiment_cache_path

logger = get_logger(__name__)

SENTIMENT_Z_COL = "news_sentiment_mean_z"
SENTIMENT_EXTRA_COLS = [
    "sentiment_1h",
    "sentiment_24h",
    "sentiment_3d",
    "news_volume",
    "sentiment_volatility",
]
SENTIMENT_ALL_COLS = [SENTIMENT_Z_COL, *SENTIMENT_EXTRA_COLS]


def _neutralize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SENTIMENT_ALL_COLS:
        out[col] = 0.0
    return out


def _load_db_sentiment() -> pd.DataFrame | None:
    if not os.getenv("DATABASE_URL"):
        return None
    try:
        from database.news_queries import fetch_daily_symbol_sentiment_df

        df = fetch_daily_symbol_sentiment_df()
    except Exception as e:
        logger.warning("Could not load daily_symbol_sentiment from DB: %s", e)
        return None
    if df.empty:
        return None
    return df


def _merge_sentiment_frame(
    out: pd.DataFrame, src: pd.DataFrame, day_col: str
) -> pd.DataFrame:
    keep_cols = [
        "symbol",
        day_col,
        *[c for c in SENTIMENT_ALL_COLS if c in src.columns],
    ]
    raw = (
        src[keep_cols]
        .copy()
        .assign(
            symbol=lambda d: d["symbol"].astype(str),
            _sent_merge_day=lambda d: pd.to_datetime(
                d[day_col], utc=True
            ).dt.normalize(),
        )
    )
    sub = raw.drop_duplicates(subset=["symbol", "_sent_merge_day"], keep="last")

    merged = out.merge(
        sub.drop(columns=[day_col], errors="ignore"),
        on=["symbol", "_sent_merge_day"],
        how="left",
    )
    updates = {}
    for col in SENTIMENT_ALL_COLS:
        base = (
            merged[col] if col in merged.columns else pd.Series(0.0, index=merged.index)
        )
        updates[col] = pd.to_numeric(base, errors="coerce").fillna(0.0)
    merged = merged.assign(**updates)
    return merged.drop(columns=["_sent_merge_day"])


def attach_sentiment_features(
    df: pd.DataFrame,
    *,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Left-merge sentiment features on (symbol, UTC calendar day).

    Postgres gold rows (from ``rollup_daily``) store rolling z-scores per symbol
    (60-day window, 10-day min) for horizon means and volatility; ``news_volume``
    is rolling z of log1p(trailing 24h article count). Legacy Parquet may only
    have ``news_sentiment_mean_z``; other columns default to 0.

    Priority: (1) Postgres when ``DATABASE_URL`` is set and rows exist;
    (2) Parquet cache; (3) neutral 0.0.
    """
    if "timestamp" not in df.columns or "symbol" not in df.columns:
        logger.warning("attach_sentiment_features: missing timestamp/symbol; skipping")
        return _neutralize(df)

    merge_day = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    out = df.assign(_sent_merge_day=merge_day)

    db_df = _load_db_sentiment()
    if db_df is not None and "as_of_date" in db_df.columns:
        try:
            return _merge_sentiment_frame(out, db_df, "as_of_date")
        except Exception as e:
            logger.warning("DB sentiment merge failed, trying Parquet: %s", e)

    p = cache_path or default_sentiment_cache_path()
    if not p.is_file():
        logger.info("No sentiment cache at %s — using neutral features", p)
        return _neutralize(out.drop(columns=["_sent_merge_day"]))

    try:
        cache = pd.read_parquet(p)
    except Exception as e:
        logger.warning("Failed to read sentiment cache %s: %s", p, e)
        return _neutralize(out.drop(columns=["_sent_merge_day"]))

    if "symbol" not in cache.columns or "timestamp" not in cache.columns:
        logger.warning("Sentiment cache missing symbol/timestamp columns")
        return _neutralize(out.drop(columns=["_sent_merge_day"]))

    return _merge_sentiment_frame(out, cache, "timestamp")
