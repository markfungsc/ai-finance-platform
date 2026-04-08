"""Headlines from yfinance (no extra API key)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
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


def fetch_news_texts_for_bar_day(symbol: str, bar_day_utc: pd.Timestamp) -> list[str]:
    """
    Return article texts whose publication time falls on the same UTC calendar day
    as ``bar_day_utc`` (normalized). News list depth is limited by yfinance.
    """
    bar_day_utc = pd.Timestamp(bar_day_utc).tz_convert(timezone.utc).normalize()
    try:
        ticker = yf.Ticker(symbol)
        items = ticker.news or []
    except Exception:
        return []
    texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pub = _parse_pub(item)
        if pub is None:
            continue
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        pts = pd.Timestamp(pub).tz_convert(timezone.utc).normalize()
        if pts != bar_day_utc:
            continue
        t = _headline_text(item)
        if t:
            texts.append(t)
    return texts
