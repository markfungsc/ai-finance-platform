"""Ingest news into bronze/silver Postgres tables."""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import date

import httpx

from constants import TRAIN_SYMBOLS
from data_pipeline.news.gdelt_adapter import fetch_gdelt_news_async, iter_gdelt_news
from data_pipeline.news.kaggle_adapter import iter_kaggle_news_multi
from data_pipeline.news.kaggle_datasets import KAGGLE_DATASETS
from data_pipeline.news.sec_adapter import fetch_sec_news_async, iter_sec_news
from data_pipeline.news.yfinance_adapter import iter_yfinance_news
from database.news_queries import insert_clean_article, upsert_raw_news
from log_config import get_logger

logger = get_logger(__name__)


def _score_fn_or_none(score_finbert: bool):
    if not score_finbert:
        return None
    from ml.sentiment.finbert_scorer import score_text

    def score_fn(text: str) -> float | None:
        if not text.strip():
            return None
        return float(score_text(text))

    return score_fn


def _ingest_items(items, *, source: str, score_finbert: bool) -> tuple[int, int]:
    """Returns (raw_upserts, clean_inserts_attempted)."""
    n_raw = 0
    n_clean = 0
    finbert_scalar = None
    score_fn = _score_fn_or_none(score_finbert)

    for item in items:
        rid = upsert_raw_news(
            source=source,
            external_id=item.external_id,
            content_sha256=item.content_sha256,
            raw_payload=item.raw_item,
        )
        n_raw += 1
        if score_fn is not None:
            finbert_scalar = score_fn(item.text_for_score)
        cid = insert_clean_article(
            raw_news_id=rid,
            symbol=item.symbol,
            url=item.url,
            title=item.title,
            summary=item.summary,
            published_at=item.published_at,
            content_sha256=item.content_sha256,
            finbert_scalar=finbert_scalar,
        )
        n_clean += 1
        logger.debug("clean id=%s symbol=%s", cid, item.symbol)
    return n_raw, n_clean


def ingest_symbol_yfinance(symbol: str, *, score_finbert: bool) -> tuple[int, int]:
    return _ingest_items(
        iter_yfinance_news(symbol), source="yfinance", score_finbert=score_finbert
    )


def ingest_symbol_gdelt(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
    score_finbert: bool,
) -> tuple[int, int]:
    return _ingest_items(
        iter_gdelt_news(symbol, start_date=start_date, end_date=end_date),
        source="gdelt",
        score_finbert=score_finbert,
    )


def ingest_symbol_sec(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
    score_finbert: bool,
) -> tuple[int, int]:
    return _ingest_items(
        iter_sec_news(symbol, start_date=start_date, end_date=end_date),
        source="sec",
        score_finbert=score_finbert,
    )


async def _fetch_items_async(
    symbol: str,
    *,
    provider: str,
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    request_timeout: float,
    retry_max: int,
    kaggle_dataset_paths: list[str] | None = None,
    kaggle_dataset_keys: list[str] | None = None,
) -> tuple[list, dict]:
    if provider == "gdelt":
        return await fetch_gdelt_news_async(
            symbol,
            start_date,
            end_date,
            client=client,
            sem=sem,
            request_timeout=request_timeout,
            retry_max=retry_max,
            provider_sleep_s=0.2,
        )
    if provider == "sec":
        return await fetch_sec_news_async(
            symbol,
            start_date,
            end_date,
            client=client,
            sem=sem,
            request_timeout=request_timeout,
            retry_max=retry_max,
            provider_sleep_s=0.2,
        )
    if provider == "hybrid":
        a_items, a_m = await fetch_gdelt_news_async(
            symbol,
            start_date,
            end_date,
            client=client,
            sem=sem,
            request_timeout=request_timeout,
            retry_max=retry_max,
            provider_sleep_s=0.2,
        )
        b_items, b_m = await fetch_sec_news_async(
            symbol,
            start_date,
            end_date,
            client=client,
            sem=sem,
            request_timeout=request_timeout,
            retry_max=retry_max,
            provider_sleep_s=0.2,
        )
        # Deduplicate by external_id/content hash.
        seen = set()
        out = []
        for it in [*a_items, *b_items]:
            k = (it.external_id, it.content_sha256)
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        m = {
            "fetched": int(a_m["fetched"] + b_m["fetched"]),
            "kept": int(len(out)),
            "filtered": int(a_m["filtered"] + b_m["filtered"]),
            "retries": int(a_m["retries"] + b_m["retries"]),
        }
        return out, m
    if provider == "kaggle":
        if not kaggle_dataset_paths:
            raise ValueError("--kaggle-dataset-path is required for provider=kaggle")
        keys = kaggle_dataset_keys or ["generic_financial_news"]
        if len(keys) == 1 and len(kaggle_dataset_paths) > 1:
            keys = keys * len(kaggle_dataset_paths)
        if len(keys) != len(kaggle_dataset_paths):
            raise ValueError(
                "--kaggle-dataset-key and --kaggle-dataset-path must have equal counts"
            )
        dataset_pairs = list(zip(keys, kaggle_dataset_paths, strict=True))
        items = list(
            iter_kaggle_news_multi(
                symbol,
                dataset_pairs=dataset_pairs,
                start_date=start_date,
                end_date=end_date,
            )
        )
        return items, {"fetched": len(items), "kept": len(items), "filtered": 0, "retries": 0}
    # yfinance path stays sync; run in thread.
    items = list(iter_yfinance_news(symbol))
    return items, {"fetched": len(items), "kept": len(items), "filtered": 0, "retries": 0}


async def _heartbeat_loop(state: dict, every_s: int) -> None:
    while not state.get("done"):
        await asyncio.sleep(max(1, every_s))
        elapsed = max(1e-6, time.time() - state["t0"])
        done = state["symbols_done"]
        total = state["symbols_total"]
        req_s = state["fetched_total"] / elapsed
        rows_s = state["inserted_total"] / elapsed
        eta_s = ((total - done) / done * elapsed) if done > 0 else float("nan")
        eta_txt = "n/a" if done == 0 else f"{int(eta_s)}s"
        logger.info(
            "HEARTBEAT done=%d/%d req_s=%.2f rows_s=%.2f retries=%d eta=%s",
            done,
            total,
            req_s,
            rows_s,
            state["retries_total"],
            eta_txt,
        )


async def _process_symbol(
    symbol: str,
    *,
    args,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    state: dict,
    score_finbert: bool,
) -> None:
    ts0 = time.time()
    logger.info("SYMBOL_START symbol=%s provider=%s", symbol, args.provider)
    try:
        items, m = await _fetch_items_async(
            symbol,
            provider=args.provider,
            start_date=args.from_date,
            end_date=args.to_date,
            client=client,
            sem=sem,
            request_timeout=args.request_timeout,
            retry_max=args.retry_max,
            kaggle_dataset_paths=args.kaggle_dataset_paths,
            kaggle_dataset_keys=args.kaggle_dataset_keys,
        )
        # DB writes and optional model inference can block; offload to thread.
        source = "yfinance" if args.provider == "yfinance" else args.provider
        raw, inserted = await asyncio.to_thread(
            _ingest_items, items, source=source, score_finbert=score_finbert
        )
        elapsed = time.time() - ts0
        state["fetched_total"] += int(m["fetched"])
        state["kept_total"] += int(m["kept"])
        state["filtered_total"] += int(m["filtered"])
        state["retries_total"] += int(m["retries"])
        state["inserted_total"] += int(inserted)
        state["raw_upserts_total"] += int(raw)
        state["symbols_done"] += 1
        logger.info(
            "SYMBOL_DONE symbol=%s fetched=%d kept=%d inserted=%d dup=%d retries=%d elapsed_s=%d",
            symbol,
            int(m["fetched"]),
            int(m["kept"]),
            int(inserted),
            max(int(m["kept"]) - int(inserted), 0),
            int(m["retries"]),
            int(elapsed),
        )
    except Exception as e:
        state["symbols_done"] += 1
        state["failed_symbols"] += 1
        logger.exception("SYMBOL_FAIL symbol=%s err=%s", symbol, e)


def main(argv: list[str] | None = None) -> None:
    def _split_multi(values: list[str] | None) -> list[str]:
        out: list[str] = []
        for v in values or []:
            out.extend([p.strip() for p in str(v).split(",") if p.strip()])
        return out

    ap = argparse.ArgumentParser(description="Ingest news to Postgres")
    ap.add_argument(
        "--provider",
        choices=("yfinance", "gdelt", "sec", "hybrid", "kaggle"),
        default="yfinance",
        help="Data source provider",
    )
    ap.add_argument(
        "--symbols",
        nargs="*",
        default=list(TRAIN_SYMBOLS),
        help="Symbols (default: TRAIN_SYMBOLS)",
    )
    ap.add_argument(
        "--score-finbert",
        action="store_true",
        help="Score each article with FinBERT (requires requirements-nlp)",
    )
    ap.add_argument(
        "--from-date",
        type=date.fromisoformat,
        default=date(2015, 1, 1),
        help="Start date for provider=gdelt/sec/hybrid (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--to-date",
        type=date.fromisoformat,
        default=date.today(),
        help="End date for provider=gdelt/sec/hybrid (YYYY-MM-DD)",
    )
    ap.add_argument("--kaggle-dataset-path", action="append", default=[])
    ap.add_argument("--kaggle-dataset-key", action="append", default=[])
    ap.add_argument("--heartbeat-seconds", type=int, default=20)
    ap.add_argument("--max-concurrency", type=int, default=3)
    ap.add_argument("--request-timeout", type=float, default=30.0)
    ap.add_argument("--retry-max", type=int, default=4)
    args = ap.parse_args(argv)
    args.kaggle_dataset_paths = _split_multi(args.kaggle_dataset_path)
    args.kaggle_dataset_keys = _split_multi(args.kaggle_dataset_key) or [
        "generic_financial_news"
    ]
    bad = [k for k in args.kaggle_dataset_keys if k not in KAGGLE_DATASETS]
    if bad:
        raise ValueError(
            f"Unsupported --kaggle-dataset-key: {bad}. "
            f"Allowed={sorted(KAGGLE_DATASETS.keys())}"
        )
    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    state = {
        "t0": time.time(),
        "done": False,
        "symbols_total": len(symbols),
        "symbols_done": 0,
        "failed_symbols": 0,
        "fetched_total": 0,
        "kept_total": 0,
        "filtered_total": 0,
        "inserted_total": 0,
        "raw_upserts_total": 0,
        "retries_total": 0,
    }

    async def _run():
        logger.info(
            "RUN_START provider=%s from=%s to=%s symbols=%d mode=safe",
            args.provider,
            args.from_date,
            args.to_date,
            len(symbols),
        )
        sem = asyncio.Semaphore(max(1, args.max_concurrency))
        async with httpx.AsyncClient() as client:
            hb = asyncio.create_task(_heartbeat_loop(state, args.heartbeat_seconds))
            try:
                tasks = [
                    _process_symbol(
                        sym,
                        args=args,
                        client=client,
                        sem=sem,
                        state=state,
                        score_finbert=args.score_finbert,
                    )
                    for sym in symbols
                ]
                await asyncio.gather(*tasks)
            finally:
                state["done"] = True
                await hb
        elapsed = int(time.time() - state["t0"])
        logger.info(
            "RUN_DONE fetched=%d kept=%d inserted=%d failed_symbols=%d retries=%d elapsed_s=%d",
            state["fetched_total"],
            state["kept_total"],
            state["inserted_total"],
            state["failed_symbols"],
            state["retries_total"],
            elapsed,
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
