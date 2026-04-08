"""Build Parquet cache: yfinance headlines → FinBERT scores → rolling z per symbol."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from constants import TRAIN_SYMBOLS
from database.queries import fetch_features_z
from log_config import get_logger
from ml.sentiment.finbert_scorer import score_texts
from ml.sentiment.news_yfinance import fetch_news_texts_for_bar_day
from ml.sentiment.paths import default_sentiment_cache_path

logger = get_logger(__name__)


def _rolling_z(g: pd.Series, window: int, min_periods: int) -> pd.Series:
    m = g.rolling(window, min_periods=min_periods).mean()
    s = g.rolling(window, min_periods=min_periods).std()
    out = (g - m) / s.replace(0.0, np.nan)
    return out.fillna(0.0)


def build_rows_for_symbol(
    symbol: str,
    *,
    max_bars: int | None,
    score_news: bool,
) -> pd.DataFrame:
    df = fetch_features_z(symbol)
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "news_sentiment_raw",
                "news_sentiment_mean_z",
            ]
        )
    ts = df[["timestamp"]].drop_duplicates().sort_values("timestamp")
    if max_bars is not None:
        ts = ts.tail(int(max_bars))

    rows: list[dict] = []
    for _, r in ts.iterrows():
        bar_ts = r["timestamp"]
        day = pd.Timestamp(bar_ts)
        if day.tzinfo is None:
            day = day.tz_localize("UTC")
        else:
            day = day.tz_convert("UTC")
        day_n = day.normalize()

        if not score_news:
            raw = np.nan
        else:
            texts = fetch_news_texts_for_bar_day(symbol, day_n)
            if not texts:
                raw = np.nan
            else:
                scores = score_texts(texts)
                raw = float(np.nanmean(scores)) if scores else np.nan

        rows.append(
            {
                "symbol": str(symbol),
                "timestamp": day_n,
                "news_sentiment_raw": raw,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(news_sentiment_mean_z=0.0)

    out["news_sentiment_raw"] = pd.to_numeric(
        out["news_sentiment_raw"], errors="coerce"
    )
    out["news_sentiment_raw_filled"] = out["news_sentiment_raw"].fillna(0.0)

    g = out.groupby("symbol", group_keys=False)["news_sentiment_raw_filled"]
    out["news_sentiment_mean_z"] = g.transform(
        lambda s: _rolling_z(s, window=60, min_periods=10)
    )
    out = out.drop(columns=["news_sentiment_raw_filled"])
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="Build daily sentiment Parquet cache")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: data/sentiment/daily_sentiment.parquet)",
    )
    ap.add_argument(
        "--symbols",
        nargs="*",
        default=list(TRAIN_SYMBOLS),
        help="Symbols (default: TRAIN_SYMBOLS)",
    )
    ap.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="Last N bars per symbol (for faster dev runs)",
    )
    ap.add_argument(
        "--no-score",
        action="store_true",
        help="Skip FinBERT / news fetch; write neutral zeros (schema check)",
    )
    args = ap.parse_args(argv)

    out_path = args.output or default_sentiment_cache_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for sym in args.symbols:
        logger.info("Sentiment cache: processing %s", sym)
        part = build_rows_for_symbol(
            sym,
            max_bars=args.max_bars,
            score_news=not args.no_score,
        )
        if not part.empty:
            parts.append(part)

    if not parts:
        df = pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "news_sentiment_raw",
                "news_sentiment_mean_z",
            ]
        )
    else:
        df = pd.concat(parts, ignore_index=True)

    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return out_path


if __name__ == "__main__":
    main()
