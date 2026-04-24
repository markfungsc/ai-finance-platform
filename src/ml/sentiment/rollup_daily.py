"""Recompute gold ``daily_symbol_sentiment`` from silver ``clean_news_articles``."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from constants import TRAIN_SYMBOLS
from database.news_queries import (
    fetch_clean_news_for_rollup,
    fetch_feature_days_for_symbol,
    fetch_latest_daily_rollup_date_for_symbol,
    upsert_daily_symbol_sentiment_rows,
)
from log_config import get_logger
from universe.resolve import resolve_ingestion_universe

logger = get_logger(__name__)

ROLL_WINDOW = 60
ROLL_MIN_PERIODS = 10


def _rolling_z(
    series: pd.Series, window: int = ROLL_WINDOW, min_periods: int = ROLL_MIN_PERIODS
) -> pd.Series:
    m = series.rolling(window, min_periods=min_periods).mean()
    s = series.rolling(window, min_periods=min_periods).std()
    out = (series - m) / s.replace(0.0, np.nan)
    return out.fillna(0.0)


def _window_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).mean())


def _window_std(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    s = float(pd.to_numeric(series, errors="coerce").fillna(0.0).std())
    if np.isnan(s):
        return 0.0
    return s


def load_clean_news() -> pd.DataFrame:
    df = fetch_clean_news_for_rollup()
    if df.empty:
        return df
    # Own a mutable frame (read_sql / mocks may return a view; CoW-safe for pandas 3).
    df = df.copy()
    df.loc[:, "published_at"] = pd.to_datetime(df["published_at"], utc=True)
    fin = pd.to_numeric(df["finbert_scalar"], errors="coerce").fillna(0.0)
    df.loc[:, "finbert_scalar"] = fin.to_numpy(dtype=float, copy=False)
    df.loc[:, "as_of_date"] = df["published_at"].dt.normalize().dt.date
    return df.sort_values(["symbol", "published_at"]).reset_index(drop=True)


def recompute_daily_rollups(
    *,
    full_recompute: bool = False,
    lookback_days: int = 90,
    progress_every: int = 25,
    upsert_chunk_size: int = 2000,
) -> int:
    df = load_clean_news()
    if df.empty:
        logger.info(
            "No clean news rows found; generating neutral rows from feature timeline"
        )
        df = pd.DataFrame(
            columns=["symbol", "published_at", "finbert_scalar", "as_of_date"]
        )

    try:
        _mode, resolved_symbols = resolve_ingestion_universe()
        symbols = [s.strip().upper() for s in resolved_symbols if s and str(s).strip()]
        logger.info(
            "Using symbols from INGESTION_UNIVERSE resolver for rollup: count=%d",
            len(symbols),
        )
    except Exception:
        logger.exception(
            "Failed to resolve INGESTION_UNIVERSE for rollup; falling back to TRAIN_SYMBOLS"
        )
        symbols = list(TRAIN_SYMBOLS)

    raw_rows: list[dict] = []
    t0 = time.perf_counter()
    for idx, sym in enumerate(symbols, start=1):
        day_df = fetch_feature_days_for_symbol(sym)
        if day_df.empty:
            if idx % max(1, progress_every) == 0:
                elapsed = time.perf_counter() - t0
                rate = idx / max(elapsed, 1e-6)
                logger.info(
                    "Rollup progress: symbols=%d/%d rows=%d rate=%.2f sym/s",
                    idx,
                    len(symbols),
                    len(raw_rows),
                    rate,
                )
            continue
        day_df = day_df.sort_values("as_of_date").reset_index(drop=True)
        if not full_recompute:
            last_done = fetch_latest_daily_rollup_date_for_symbol(sym)
            if last_done is not None:
                cutoff = (
                    pd.Timestamp(last_done)
                    - pd.Timedelta(days=max(int(lookback_days), ROLL_WINDOW))
                ).date()
                day_df = day_df[day_df["as_of_date"] >= cutoff].reset_index(drop=True)
                if day_df.empty:
                    if idx % max(1, progress_every) == 0:
                        elapsed = time.perf_counter() - t0
                        rate = idx / max(elapsed, 1e-6)
                        logger.info(
                            "Rollup progress: symbols=%d/%d rows=%d rate=%.2f sym/s",
                            idx,
                            len(symbols),
                            len(raw_rows),
                            rate,
                        )
                    continue
        sym_articles = df[df["symbol"] == sym]

        if sym_articles.empty:
            daily_mean_raw = pd.Series(0.0, index=day_df.index)
        else:
            daily_means = (
                sym_articles.groupby("as_of_date", as_index=False)["finbert_scalar"]
                .mean()
                .rename(columns={"finbert_scalar": "daily_mean_raw"})
            )
            day_df = day_df.merge(daily_means, on="as_of_date", how="left")
            daily_mean_raw = pd.to_numeric(
                day_df["daily_mean_raw"], errors="coerce"
            ).fillna(0.0)

        z_mean = _rolling_z(
            daily_mean_raw, window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
        )

        for i, row in enumerate(day_df.itertuples(index=False)):
            day_end = pd.Timestamp(row.as_of_date, tz="UTC") + pd.Timedelta(days=1)
            w1h = sym_articles[
                (sym_articles["published_at"] > day_end - pd.Timedelta(hours=1))
                & (sym_articles["published_at"] <= day_end)
            ]["finbert_scalar"]
            w24h = sym_articles[
                (sym_articles["published_at"] > day_end - pd.Timedelta(hours=24))
                & (sym_articles["published_at"] <= day_end)
            ]["finbert_scalar"]
            w3d = sym_articles[
                (sym_articles["published_at"] > day_end - pd.Timedelta(days=3))
                & (sym_articles["published_at"] <= day_end)
            ]["finbert_scalar"]
            day_mask = sym_articles["as_of_date"] == row.as_of_date
            article_count = int(day_mask.sum())

            raw_rows.append(
                {
                    "symbol": sym,
                    "as_of_date": row.as_of_date,
                    "news_sentiment_mean_z": float(z_mean.iloc[i]),
                    "sentiment_1h_raw": _window_mean(w1h),
                    "sentiment_24h_raw": _window_mean(w24h),
                    "sentiment_3d_raw": _window_mean(w3d),
                    "news_volume_raw": int(len(w24h)),
                    "sentiment_volatility_raw": _window_std(w3d),
                    "article_count": article_count,
                }
            )
        if idx % max(1, progress_every) == 0:
            elapsed = time.perf_counter() - t0
            rate = idx / max(elapsed, 1e-6)
            logger.info(
                "Rollup progress: symbols=%d/%d rows=%d rate=%.2f sym/s",
                idx,
                len(symbols),
                len(raw_rows),
                rate,
            )

    rdf = pd.DataFrame(raw_rows)
    if rdf.empty:
        return 0

    out_rows: list[dict] = []
    for sym, g in rdf.groupby("symbol"):
        g = g.sort_values("as_of_date").reset_index(drop=True)
        g = g.assign(
            sentiment_1h=_rolling_z(
                g["sentiment_1h_raw"], window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
            ),
            sentiment_24h=_rolling_z(
                g["sentiment_24h_raw"], window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
            ),
            sentiment_3d=_rolling_z(
                g["sentiment_3d_raw"], window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
            ),
            news_volume=_rolling_z(
                np.log1p(g["news_volume_raw"].astype(float)),
                window=ROLL_WINDOW,
                min_periods=ROLL_MIN_PERIODS,
            ),
            sentiment_volatility=_rolling_z(
                g["sentiment_volatility_raw"],
                window=ROLL_WINDOW,
                min_periods=ROLL_MIN_PERIODS,
            ),
        )
        for _, rr in g.iterrows():
            out_rows.append(
                {
                    "symbol": rr["symbol"],
                    "as_of_date": rr["as_of_date"],
                    "z": float(rr["news_sentiment_mean_z"]),
                    "sentiment_1h": float(rr["sentiment_1h"]),
                    "sentiment_24h": float(rr["sentiment_24h"]),
                    "sentiment_3d": float(rr["sentiment_3d"]),
                    "news_volume": float(rr["news_volume"]),
                    "sentiment_volatility": float(rr["sentiment_volatility"]),
                    "article_count": int(rr["article_count"]),
                }
            )

    upsert_daily_symbol_sentiment_rows(out_rows, chunk_size=upsert_chunk_size)
    logger.info("Upserted %d daily_symbol_sentiment rows", len(out_rows))
    return len(out_rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Recompute daily_symbol_sentiment gold table"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Full recompute of all available feature days",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Incremental mode lookback window (minimum rolling window is enforced)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress every N symbols",
    )
    ap.add_argument(
        "--upsert-chunk-size",
        type=int,
        default=2000,
        help="Batch size for sentiment upsert writes",
    )
    args = ap.parse_args(argv)
    recompute_daily_rollups(
        full_recompute=bool(args.full),
        lookback_days=int(args.lookback_days),
        progress_every=int(args.progress_every),
        upsert_chunk_size=int(args.upsert_chunk_size),
    )


if __name__ == "__main__":
    main()
