import asyncio
import time
from datetime import date

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
                "stock,date,headline,url",
                "AAPL,2020-01-02T10:30:00Z,Apple positive update,https://ex/a1",
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
