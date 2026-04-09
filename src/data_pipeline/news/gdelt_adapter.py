"""GDELT doc API adapter with precision-first filtering."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from datetime import UTC, date, datetime
from typing import Iterator

import httpx
import requests

from data_pipeline.news.yfinance_adapter import NewsItemNormalized

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_FINANCE_TERMS = (
    "earnings",
    "revenue",
    "guidance",
    "profit",
    "shares",
    "stock",
    "dividend",
    "quarter",
    "nasdaq",
    "nyse",
)


def _parse_iso_day(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        # GDELT returns forms like 2026-03-27T12:34:56Z
        return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(UTC)
    except (TypeError, ValueError, OSError):
        return None


def _strict_match(symbol: str, title: str, text: str) -> bool:
    """Precision-first filter for symbol relevance."""
    s = symbol.upper()
    body = f"{title}\n{text}".upper()
    has_symbol = bool(re.search(rf"\b{s}\b", body) or re.search(rf"\${s}\b", body))
    has_finance = any(t.upper() in body for t in _FINANCE_TERMS)
    return has_symbol and has_finance


def _content_sha256(symbol: str, title: str, body: str, published_at: datetime) -> str:
    raw = f"{symbol.upper()}|{title.strip()}|{body.strip()}|{published_at.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fetch_day(symbol: str, day: date) -> list[dict]:
    # Query one UTC day for manageable payload and retries.
    q = f'"{symbol}" AND ({" OR ".join(_FINANCE_TERMS[:6])})'
    params = {
        "query": q,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": f"{day.strftime('%Y%m%d')}000000",
        "enddatetime": f"{day.strftime('%Y%m%d')}235959",
    }
    backoff = 1.0
    for _ in range(4):
        try:
            r = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 12.0)
                continue
            r.raise_for_status()
            payload = r.json()
            arts = payload.get("articles") if isinstance(payload, dict) else None
            return arts if isinstance(arts, list) else []
        except (requests.RequestException, ValueError):
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 12.0)
    return []


async def _fetch_day_async(
    symbol: str,
    day: date,
    *,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    request_timeout: float,
    retry_max: int,
    provider_sleep_s: float,
) -> tuple[list[dict], int]:
    q = f'"{symbol}" AND ({" OR ".join(_FINANCE_TERMS[:6])})'
    params = {
        "query": q,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": f"{day.strftime('%Y%m%d')}000000",
        "enddatetime": f"{day.strftime('%Y%m%d')}235959",
    }
    retries = 0
    backoff = 2
    for _ in range(retry_max + 1):
        async with sem:
            try:
                r = await client.get(
                    GDELT_DOC_API, params=params, timeout=request_timeout
                )
                if r.status_code == 429:
                    await asyncio.sleep(5)
                    continue
                if r.status_code == 500 <= r.status_code < 600:
                    retries += 1
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, 12.0)
                    continue
                r.raise_for_status()
                payload = r.json()
                arts = payload.get("articles") if isinstance(payload, dict) else None
                await asyncio.sleep(provider_sleep_s)
                return (arts if isinstance(arts, list) else []), retries
            except (httpx.HTTPError, ValueError):
                retries += 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 12.0)
    return [], retries


async def fetch_gdelt_news_async(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    request_timeout: float = 30.0,
    retry_max: int = 4,
    provider_sleep_s: float = 0.15,
) -> tuple[list[NewsItemNormalized], dict]:
    """Async fetch+normalize for [start_date, end_date]. Returns items and metrics."""
    if end_date < start_date:
        return [], {"fetched": 0, "kept": 0, "filtered": 0, "retries": 0}
    tasks = []
    cur = start_date
    while cur <= end_date:
        tasks.append(
            _fetch_day_async(
                symbol,
                cur,
                client=client,
                sem=sem,
                request_timeout=request_timeout,
                retry_max=retry_max,
                provider_sleep_s=provider_sleep_s,
            )
        )
        cur = cur.fromordinal(cur.toordinal() + 1)

    results = await asyncio.gather(*tasks)
    seen: set[str] = set()
    out: list[NewsItemNormalized] = []
    fetched = 0
    filtered = 0
    retries = 0
    for items, rcount in results:
        retries += rcount
        fetched += len(items)
        for a in items:
            if not isinstance(a, dict):
                continue
            title = str(a.get("title") or "").strip()
            body = str(a.get("seendate") or "").strip()
            text_for_score = title
            if not _strict_match(symbol, title, text_for_score):
                filtered += 1
                continue
            pub = _parse_iso_day(str(a.get("seendate") or "")) or datetime(
                start_date.year, start_date.month, start_date.day, tzinfo=UTC
            )
            url = str(a.get("url") or "").strip() or None
            ext = str(a.get("url") or "").strip()
            if not ext:
                ext = hashlib.sha256(
                    f"{symbol}|{title}|{pub.isoformat()}".encode()
                ).hexdigest()[:40]
            if ext in seen:
                continue
            seen.add(ext)
            out.append(
                NewsItemNormalized(
                    external_id=ext,
                    content_sha256=_content_sha256(symbol, title, body, pub),
                    raw_item=a,
                    symbol=symbol.upper(),
                    url=url,
                    title=title,
                    summary="",
                    published_at=pub,
                    text_for_score=text_for_score,
                )
            )
    return out, {
        "fetched": fetched,
        "kept": len(out),
        "filtered": filtered,
        "retries": retries,
    }


def iter_gdelt_news(
    symbol: str, start_date: date, end_date: date
) -> Iterator[NewsItemNormalized]:
    """Yield precision-filtered normalized rows for [start_date, end_date]."""
    if end_date < start_date:
        return
    cur = start_date
    seen: set[str] = set()
    while cur <= end_date:
        for a in _fetch_day(symbol, cur):
            if not isinstance(a, dict):
                continue
            title = str(a.get("title") or "").strip()
            body = str(a.get("seendate") or "").strip()
            text_for_score = title
            if not _strict_match(symbol, title, text_for_score):
                continue
            pub = _parse_iso_day(str(a.get("seendate") or "")) or datetime(
                cur.year, cur.month, cur.day, tzinfo=UTC
            )
            url = str(a.get("url") or "").strip() or None
            ext = str(a.get("url") or "").strip()
            if not ext:
                ext = hashlib.sha256(
                    f"{symbol}|{title}|{pub.isoformat()}".encode()
                ).hexdigest()[:40]
            if ext in seen:
                continue
            seen.add(ext)
            yield NewsItemNormalized(
                external_id=ext,
                content_sha256=_content_sha256(symbol, title, body, pub),
                raw_item=a,
                symbol=symbol.upper(),
                url=url,
                title=title,
                summary="",
                published_at=pub,
                text_for_score=text_for_score,
            )
        cur = cur.fromordinal(cur.toordinal() + 1)
