"""GDELT doc API adapter with precision-first filtering."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import UTC, date, datetime
from typing import Iterator

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
