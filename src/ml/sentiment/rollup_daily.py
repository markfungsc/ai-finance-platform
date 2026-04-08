"""Recompute gold ``daily_symbol_sentiment`` from silver ``clean_news_articles``."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from constants import TRAIN_SYMBOLS
from database.news_queries import (
    fetch_clean_news_for_rollup,
    fetch_feature_days_for_symbol,
    upsert_daily_symbol_sentiment_rows,
)
from log_config import get_logger

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
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["finbert_scalar"] = pd.to_numeric(df["finbert_scalar"], errors="coerce").fillna(
        0.0
    )
    df["as_of_date"] = df["published_at"].dt.normalize().dt.date
    return df.sort_values(["symbol", "published_at"]).reset_index(drop=True)


def recompute_daily_rollups() -> int:
    df = load_clean_news()
    if df.empty:
        logger.info(
            "No clean news rows found; generating neutral rows from feature timeline"
        )
        df = pd.DataFrame(
            columns=["symbol", "published_at", "finbert_scalar", "as_of_date"]
        )

    raw_rows: list[dict] = []
    for sym in TRAIN_SYMBOLS:
        day_df = fetch_feature_days_for_symbol(sym)
        if day_df.empty:
            continue
        day_df = day_df.sort_values("as_of_date").reset_index(drop=True)
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

    upsert_daily_symbol_sentiment_rows(out_rows)
    logger.info("Upserted %d daily_symbol_sentiment rows", len(out_rows))
    return len(out_rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Recompute daily_symbol_sentiment gold table"
    )
    ap.parse_args(argv)
    recompute_daily_rollups()


if __name__ == "__main__":
    main()
