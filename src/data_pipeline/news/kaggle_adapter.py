"""Kaggle historical news adapter."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date
from pathlib import Path
from typing import Iterator

import pandas as pd

from data_pipeline.news.kaggle_datasets import KAGGLE_DATASETS, KaggleDatasetSpec
from data_pipeline.news.yfinance_adapter import NewsItemNormalized


def _pick_text(*values: str) -> str:
    return " ".join(v.strip() for v in values if v and v.strip()).strip()


def _clean_text(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _load_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _dataset_sha256(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _to_timestamp_utc(raw: object, timezone: str) -> pd.Timestamp | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        ts = pd.to_datetime(raw, utc=False, errors="coerce")
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is None:
        return ts.tz_localize(timezone).tz_convert("UTC")
    return ts.tz_convert("UTC")


def iter_kaggle_news(
    symbol: str,
    *,
    dataset_path: str,
    dataset_key: str,
    start_date: date,
    end_date: date,
) -> Iterator[NewsItemNormalized]:
    """Yield normalized rows from a local Kaggle file."""
    if end_date < start_date:
        return
    spec: KaggleDatasetSpec = KAGGLE_DATASETS[dataset_key]
    df = _load_frame(dataset_path)
    if df.empty:
        return

    sha = _dataset_sha256(dataset_path)
    sym = symbol.upper()
    seen: set[str] = set()
    for _, row in df.iterrows():
        row_symbol = _clean_text(row.get(spec.symbol_col)) if spec.symbol_col else ""
        if spec.symbol_col:
            if row_symbol.upper() != sym:
                continue
        else:
            # market-level dataset fallback: tag all rows to requested symbol
            row_symbol = sym
        ts = _to_timestamp_utc(row.get(spec.published_at_col), spec.timezone)
        if ts is None:
            continue
        d = ts.date()
        if d < start_date or d > end_date:
            continue

        title = _clean_text(row.get(spec.title_col))
        summary = _clean_text(row.get(spec.summary_col)) if spec.summary_col else ""
        body = _clean_text(row.get(spec.body_col)) if spec.body_col else ""
        url = _clean_text(row.get(spec.url_col)) if spec.url_col else ""
        text = _pick_text(title, summary, body)
        if not text:
            continue

        base = f"{sym}|{ts.isoformat()}|{title}|{url}"
        external_id = hashlib.sha256(base.encode("utf-8")).hexdigest()[:40]
        if external_id in seen:
            continue
        seen.add(external_id)

        content_sha = hashlib.sha256(
            f"{sym}|{title}|{summary}|{body}|{ts.isoformat()}".encode("utf-8")
        ).hexdigest()
        raw_payload = {
            "dataset_key": spec.name,
            "dataset_source_url": spec.source_url,
            "dataset_path": dataset_path,
            "dataset_sha256": sha,
            "row": json.loads(row.to_json(date_format="iso")),
        }
        yield NewsItemNormalized(
            external_id=external_id,
            content_sha256=content_sha,
            raw_item=raw_payload,
            symbol=sym,
            url=url or None,
            title=title,
            summary=summary or body,
            published_at=ts.to_pydatetime().astimezone(UTC),
            text_for_score=text,
        )


def iter_kaggle_news_multi(
    symbol: str,
    *,
    dataset_pairs: list[tuple[str, str]],
    start_date: date,
    end_date: date,
) -> Iterator[NewsItemNormalized]:
    """Yield unioned and deterministically deduplicated rows across datasets."""
    seen: set[tuple[str, str, str, str]] = set()
    for dataset_key, dataset_path in dataset_pairs:
        for item in iter_kaggle_news(
            symbol,
            dataset_path=dataset_path,
            dataset_key=dataset_key,
            start_date=start_date,
            end_date=end_date,
        ):
            ts_minute = item.published_at.replace(second=0, microsecond=0).isoformat()
            title_fingerprint = hashlib.sha256(
                f"{item.title.strip().lower()}|{(item.url or '').strip().lower()}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            key = (item.symbol, ts_minute, title_fingerprint)
            if key in seen:
                continue
            seen.add(key)
            yield item

