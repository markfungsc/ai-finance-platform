"""SEC filings adapter (free fallback, precision-first)."""

from __future__ import annotations

import hashlib
import os
import time
from datetime import UTC, date, datetime
from typing import Iterator

import requests

from data_pipeline.news.yfinance_adapter import NewsItemNormalized

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_DEFAULT_UA = "ai-finance-platform/1.0 (research@example.com)"


def _ua() -> str:
    return os.getenv("SEC_USER_AGENT", _DEFAULT_UA)


def _safe_date(v: str | None) -> date | None:
    if not v:
        return None
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        return None


def _fetch_json(url: str, *, timeout: int = 30) -> dict | list | None:
    headers = {"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"}
    backoff = 1.0
    for _ in range(4):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 12.0)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 12.0)
    return None


def _ticker_to_cik(symbol: str) -> str | None:
    payload = _fetch_json(SEC_TICKERS_URL)
    if not isinstance(payload, dict):
        return None
    sym = symbol.upper()
    for v in payload.values():
        if not isinstance(v, dict):
            continue
        if str(v.get("ticker", "")).upper() != sym:
            continue
        try:
            return str(int(v.get("cik_str"))).zfill(10)
        except (TypeError, ValueError):
            return None
    return None


def iter_sec_news(
    symbol: str, start_date: date, end_date: date
) -> Iterator[NewsItemNormalized]:
    """Yield filing-derived news items for [start_date, end_date]."""
    if end_date < start_date:
        return
    cik = _ticker_to_cik(symbol)
    if not cik:
        return
    payload = _fetch_json(SEC_SUBMISSIONS_URL.format(cik=cik))
    if not isinstance(payload, dict):
        return
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    filed = recent.get("filingDate", []) or []
    accession = recent.get("accessionNumber", []) or []
    primary = recent.get("primaryDocument", []) or []
    is_inline = recent.get("isInlineXBRL", []) or []
    n = min(len(forms), len(filed), len(accession), len(primary))
    for i in range(n):
        fday = _safe_date(filed[i])
        if fday is None or fday < start_date or fday > end_date:
            continue
        form = str(forms[i] or "").strip()
        # Precision preference: keep high-signal forms only.
        if form not in {"8-K", "10-Q", "10-K", "6-K"}:
            continue
        acc_raw = str(accession[i] or "").strip()
        doc = str(primary[i] or "").strip()
        if not acc_raw or not doc:
            continue
        acc_nodash = acc_raw.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
        published_at = datetime(fday.year, fday.month, fday.day, tzinfo=UTC)
        title = f"{symbol.upper()} SEC {form} filing"
        summary = ""
        sha_raw = f"{symbol}|{form}|{acc_raw}|{doc}|{fday.isoformat()}"
        yield NewsItemNormalized(
            external_id=f"sec:{acc_raw}:{doc}",
            content_sha256=hashlib.sha256(sha_raw.encode("utf-8")).hexdigest(),
            raw_item={
                "form": form,
                "filingDate": filed[i],
                "accessionNumber": acc_raw,
                "primaryDocument": doc,
                "isInlineXBRL": is_inline[i] if i < len(is_inline) else None,
                "cik": cik,
            },
            symbol=symbol.upper(),
            url=url,
            title=title,
            summary=summary,
            published_at=published_at,
            text_for_score=title,
        )
