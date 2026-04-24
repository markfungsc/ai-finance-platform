from datetime import UTC

import pandas as pd

from ml.sentiment import rollup_daily


def test_rollup_writes_neutral_for_days_without_news(monkeypatch):
    clean = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "published_at": [pd.Timestamp("2015-01-03 18:00:00", tz=UTC)],
            "finbert_scalar": [0.7],
        }
    )
    feature_days = pd.DataFrame(
        {
            "as_of_date": [
                pd.Timestamp("2014-12-31").date(),
                pd.Timestamp("2015-01-03").date(),
            ]
        }
    )
    captured = {}

    monkeypatch.setattr(
        rollup_daily, "resolve_ingestion_universe", lambda: ("sp500", ["AAPL"])
    )
    monkeypatch.setattr(rollup_daily, "fetch_clean_news_for_rollup", lambda: clean)
    monkeypatch.setattr(
        rollup_daily, "fetch_feature_days_for_symbol", lambda _s: feature_days
    )
    monkeypatch.setattr(
        rollup_daily, "fetch_latest_daily_rollup_date_for_symbol", lambda _s: None
    )
    monkeypatch.setattr(
        rollup_daily,
        "upsert_daily_symbol_sentiment_rows",
        lambda rows, **_kwargs: captured.setdefault("rows", rows),
    )

    n = rollup_daily.recompute_daily_rollups()
    assert n == 2
    rows = {r["as_of_date"]: r for r in captured["rows"]}
    old = rows[pd.Timestamp("2014-12-31").date()]
    assert old["article_count"] == 0
    assert old["news_volume"] == 0.0


def test_rollup_incremental_filters_old_feature_days(monkeypatch):
    clean = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "published_at": [pd.Timestamp("2026-04-20 18:00:00", tz=UTC)],
            "finbert_scalar": [0.3],
        }
    )
    feature_days = pd.DataFrame(
        {
            "as_of_date": [
                pd.Timestamp("2025-01-01").date(),
                pd.Timestamp("2026-04-20").date(),
            ]
        }
    )
    captured = {}

    monkeypatch.setattr(
        rollup_daily, "resolve_ingestion_universe", lambda: ("sp500", ["AAPL"])
    )
    monkeypatch.setattr(rollup_daily, "fetch_clean_news_for_rollup", lambda: clean)
    monkeypatch.setattr(
        rollup_daily, "fetch_feature_days_for_symbol", lambda _s: feature_days
    )
    monkeypatch.setattr(
        rollup_daily,
        "fetch_latest_daily_rollup_date_for_symbol",
        lambda _s: pd.Timestamp("2026-04-19").date(),
    )
    monkeypatch.setattr(
        rollup_daily,
        "upsert_daily_symbol_sentiment_rows",
        lambda rows, **_kwargs: captured.setdefault("rows", rows),
    )

    n = rollup_daily.recompute_daily_rollups(lookback_days=30)
    assert n == 1
    assert captured["rows"][0]["as_of_date"] == pd.Timestamp("2026-04-20").date()
