"""Normalize yfinance ticker.news items for bronze/silver ingest."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import yfinance as yf


def _parse_pub(item: dict) -> datetime | None:
    try:
        c = item.get("content") or {}
        raw = c.get("pubDate") or c.get("displayTime")
        if not raw:
            return None
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError, OSError):
        return None


def _headline_text(item: dict) -> str:
    c = item.get("content") or {}
    title = str(c.get("title") or "").strip()
    summary = str(c.get("summary") or "").strip()
    if title and summary:
        return f"{title}. {summary}"
    return title or summary or ""


def _external_id(item: dict) -> str:
    c = item.get("content") or {}
    u = item.get("uuid") or c.get("id") or item.get("id")
    if u is not None and str(u).strip():
        return str(u).strip()
    pub = _parse_pub(item)
    title = str((c.get("title") or "")).strip()
    link = str(c.get("canonicalUrl") or c.get("linkUrl") or "").strip()
    h = hashlib.sha256()
    h.update(f"{title}|{pub.isoformat() if pub else ''}|{link}".encode())
    return h.hexdigest()[:40]


def _content_sha256(item: dict, symbol: str) -> str:
    c = item.get("content") or {}
    title = str(c.get("title") or "").strip()
    summary = str(c.get("summary") or "").strip()
    pub = _parse_pub(item)
    pub_s = pub.isoformat() if pub else ""
    raw = f"{symbol}|{title}|{summary}|{pub_s}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NewsItemNormalized:
    external_id: str
    content_sha256: str
    raw_item: dict
    symbol: str
    url: str | None
    title: str
    summary: str
    published_at: datetime
    text_for_score: str


def iter_yfinance_news(symbol: str) -> Iterator[NewsItemNormalized]:
    """Yield normalized items from the latest yfinance news list (depth capped by API)."""
    try:
        ticker = yf.Ticker(symbol)
        items = ticker.news or []
    except Exception:
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        pub = _parse_pub(item)
        if pub is None:
            continue
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        else:
            pub = pub.astimezone(timezone.utc)
        c = item.get("content") or {}
        url = c.get("canonicalUrl") or c.get("linkUrl") or c.get("url")
        title = str(c.get("title") or "").strip()
        summary = str(c.get("summary") or "").strip()
        text = _headline_text(item)
        ext = _external_id(item)
        sha = _content_sha256(item, symbol)
        yield NewsItemNormalized(
            external_id=ext,
            content_sha256=sha,
            raw_item=item,
            symbol=symbol.upper(),
            url=str(url).strip() if url else None,
            title=title,
            summary=summary,
            published_at=pub,
            text_for_score=text,
        )
