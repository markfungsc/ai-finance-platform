import asyncio
import time
from datetime import UTC, date, datetime
from unittest.mock import patch

import httpx
import pytest

from data_pipeline.news import ingest


def test_heartbeat_emits_at_least_one_tick(monkeypatch):
    lines = []

    class _L:
        def info(self, msg, *args):
            lines.append(msg % args if args else msg)

    monkeypatch.setattr(ingest, "logger", _L())
    state = {
        "t0": time.time(),
        "done": False,
        "symbols_done": 0,
        "symbols_total": 2,
        "fetched_total": 0,
        "inserted_total": 0,
        "retries_total": 0,
    }

    async def _run():
        t = asyncio.create_task(ingest._heartbeat_loop(state, 1))
        await asyncio.sleep(1.2)
        state["done"] = True
        await t

    asyncio.run(_run())
    assert any("HEARTBEAT" in ln for ln in lines)


def test_fetch_items_async_kaggle_requires_path():
    async def _run():
        async with httpx.AsyncClient() as client:
            await ingest._fetch_items_async(
                "AAPL",
                provider="kaggle",
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 2),
                client=client,
                sem=asyncio.Semaphore(1),
                request_timeout=5.0,
                retry_max=1,
                kaggle_dataset_paths=None,
                kaggle_dataset_keys=["generic_financial_news"],
            )

    with pytest.raises(ValueError):
        asyncio.run(_run())


def test_fetch_items_async_kaggle_happy_path(tmp_path):
    fp = tmp_path / "hist.csv"
    fp.write_text(
        "\n".join(
            [
                "symbol,published_at,title,summary,body,url",
                "AAPL,2020-01-02T10:30:00Z,Beat estimates,Strong quarter,Revenue growth,https://ex/a1",
            ]
        ),
        encoding="utf-8",
    )

    async def _run():
        async with httpx.AsyncClient() as client:
            return await ingest._fetch_items_async(
                "AAPL",
                provider="kaggle",
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 3),
                client=client,
                sem=asyncio.Semaphore(1),
                request_timeout=5.0,
                retry_max=1,
                kaggle_dataset_paths=[str(fp)],
                kaggle_dataset_keys=["generic_financial_news"],
            )

    items, metrics = asyncio.run(_run())
    assert len(items) == 1
    assert metrics["kept"] == 1


def test_fetch_items_async_kaggle_dual_path(tmp_path):
    sp = tmp_path / "sp500.csv"
    yg = tmp_path / "yogesh.csv"
    sp.write_text(
        "\n".join(
            [
                "Title,Date",
                "Apple positive update,2020-01-02T10:30:00Z",
            ]
        ),
        encoding="utf-8",
    )
    yg.write_text(
        "\n".join(
            [
                "ticker,published_at,headline,summary,text,url",
                "AAPL,2020-01-03T11:00:00Z,Apple negative warning,Guidance cut,Weak demand,https://ex/a2",
            ]
        ),
        encoding="utf-8",
    )

    async def _run():
        async with httpx.AsyncClient() as client:
            return await ingest._fetch_items_async(
                "AAPL",
                provider="kaggle",
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 5),
                client=client,
                sem=asyncio.Semaphore(1),
                request_timeout=5.0,
                retry_max=1,
                kaggle_dataset_paths=[str(sp), str(yg)],
                kaggle_dataset_keys=[
                    "sp500_headlines_2008_2024",
                    "yogeshchary_financial_news",
                ],
            )

    items, metrics = asyncio.run(_run())
    assert len(items) == 2
    assert metrics["kept"] == 2


def test_refresh_symbol_news_gap_yfinance_invokes_ingest():
    with (
        patch(
            "database.news_queries.fetch_max_published_at_clean_news",
            return_value=None,
        ),
        patch.object(ingest, "ingest_symbol_yfinance", return_value=(4, 3, [101, 102])) as yfin,
    ):
        meta = ingest.refresh_symbol_news_gap(
            "nvda", news_lookback_days=5, provider="yfinance", score_finbert=False
        )
        yfin.assert_called_once_with("NVDA", score_finbert=False)
    assert meta["provider"] == "yfinance"
    assert meta["raw_upserts"] == 4
    assert meta["clean_inserts"] == 3
    assert meta["fetched"] == 4
    assert meta["error"] is None
    assert meta["end_date"] == date.today().isoformat()
    assert meta.get("finbert_scored") is False


def test_refresh_symbol_news_gap_start_date_from_latest_row():
    latest = datetime(2026, 4, 20, 15, 0, 0, tzinfo=UTC)
    with (
        patch(
            "database.news_queries.fetch_max_published_at_clean_news",
            return_value=latest,
        ),
        patch.object(ingest, "ingest_symbol_yfinance", return_value=(1, 1, [55])),
    ):
        meta = ingest.refresh_symbol_news_gap(
            "NVDA", news_lookback_days=30, provider="yfinance", score_finbert=False
        )
    assert meta["start_date"] == "2026-04-20"


def test_refresh_symbol_news_gap_defaults_finbert_on(monkeypatch):
    monkeypatch.delenv("TRADE_ANALYSIS_NEWS_SCORE_FINBERT", raising=False)
    with (
        patch(
            "database.news_queries.fetch_max_published_at_clean_news",
            return_value=None,
        ),
        patch.object(ingest, "ingest_symbol_yfinance", return_value=(1, 1, [9])) as yfin,
    ):
        meta = ingest.refresh_symbol_news_gap("X", news_lookback_days=3, provider="yfinance")
        yfin.assert_called_once_with("X", score_finbert=True)
    assert meta.get("finbert_scored") is True


def test_refresh_symbol_news_gap_env_can_disable_finbert(monkeypatch):
    monkeypatch.setenv("TRADE_ANALYSIS_NEWS_SCORE_FINBERT", "false")
    with (
        patch(
            "database.news_queries.fetch_max_published_at_clean_news",
            return_value=None,
        ),
        patch.object(ingest, "ingest_symbol_yfinance", return_value=(1, 1, [9])) as yfin,
    ):
        meta = ingest.refresh_symbol_news_gap("X", news_lookback_days=3, provider="yfinance")
        yfin.assert_called_once_with("X", score_finbert=False)
    assert meta.get("finbert_scored") is False


def test_refresh_symbol_news_gap_calls_qdrant_embed_when_flag_true(monkeypatch):
    monkeypatch.delenv("TRADE_ANALYSIS_EMBED_QDRANT_ON_REFRESH", raising=False)
    with (
        patch(
            "database.news_queries.fetch_max_published_at_clean_news",
            return_value=None,
        ),
        patch.object(
            ingest, "ingest_symbol_yfinance", return_value=(1, 1, [100, 101])
        ),
        patch(
            "ml.sentiment.embed_sync.embed_and_upsert_article_ids", return_value=4
        ) as emb,
    ):
        meta = ingest.refresh_symbol_news_gap(
            "ZZ",
            news_lookback_days=3,
            provider="yfinance",
            embed_new_news_in_qdrant=True,
        )
    emb.assert_called_once_with("ZZ", [100, 101])
    assert meta.get("qdrant_points_upserted") == 4
    assert meta.get("ingested_clean_article_ids") == [100, 101]


def test_refresh_symbol_news_gap_skips_qdrant_embed_by_default(monkeypatch):
    monkeypatch.delenv("TRADE_ANALYSIS_EMBED_QDRANT_ON_REFRESH", raising=False)
    with (
        patch(
            "database.news_queries.fetch_max_published_at_clean_news",
            return_value=None,
        ),
        patch.object(
            ingest, "ingest_symbol_yfinance", return_value=(1, 1, [100])
        ),
        patch("ml.sentiment.embed_sync.embed_and_upsert_article_ids") as emb,
    ):
        meta = ingest.refresh_symbol_news_gap("ZZ", news_lookback_days=3, provider="yfinance")
    emb.assert_not_called()
    assert "qdrant_points_upserted" not in meta


def test_refresh_symbol_news_gap_embeds_when_trade_analysis_default(monkeypatch):
    """Same as /trade-analysis: default_qdrant_embed_on_unset embeds without env."""
    monkeypatch.delenv("TRADE_ANALYSIS_EMBED_QDRANT_ON_REFRESH", raising=False)
    with (
        patch(
            "database.news_queries.fetch_max_published_at_clean_news",
            return_value=None,
        ),
        patch.object(
            ingest, "ingest_symbol_yfinance", return_value=(1, 1, [200, 201])
        ),
        patch(
            "ml.sentiment.embed_sync.embed_and_upsert_article_ids", return_value=3
        ) as emb,
    ):
        meta = ingest.refresh_symbol_news_gap(
            "AA",
            news_lookback_days=3,
            provider="yfinance",
            default_qdrant_embed_on_unset=True,
        )
    emb.assert_called_once_with("AA", [200, 201])
    assert meta.get("qdrant_points_upserted") == 3


def test_refresh_trade_analysis_default_respects_qdrant_embed_env_off(monkeypatch):
    monkeypatch.setenv("TRADE_ANALYSIS_EMBED_QDRANT_ON_REFRESH", "false")
    with (
        patch(
            "database.news_queries.fetch_max_published_at_clean_news",
            return_value=None,
        ),
        patch.object(
            ingest, "ingest_symbol_yfinance", return_value=(1, 1, [300])
        ),
        patch("ml.sentiment.embed_sync.embed_and_upsert_article_ids") as emb,
    ):
        meta = ingest.refresh_symbol_news_gap(
            "BB",
            news_lookback_days=3,
            provider="yfinance",
            default_qdrant_embed_on_unset=True,
        )
    emb.assert_not_called()
    assert "qdrant_points_upserted" not in meta
