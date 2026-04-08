"""Ingest news into bronze/silver Postgres tables."""

from __future__ import annotations

import argparse
from datetime import date

from constants import TRAIN_SYMBOLS
from data_pipeline.news.gdelt_adapter import iter_gdelt_news
from data_pipeline.news.sec_adapter import iter_sec_news
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest news to Postgres")
    ap.add_argument(
        "--provider",
        choices=("yfinance", "gdelt", "sec", "hybrid"),
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
    args = ap.parse_args(argv)
    for sym in args.symbols:
        sym_u = sym.strip().upper()
        logger.info("Ingesting news for %s via provider=%s", sym_u, args.provider)
        if args.provider == "gdelt":
            r, c = ingest_symbol_gdelt(
                sym_u,
                start_date=args.from_date,
                end_date=args.to_date,
                score_finbert=args.score_finbert,
            )
        elif args.provider == "sec":
            r, c = ingest_symbol_sec(
                sym_u,
                start_date=args.from_date,
                end_date=args.to_date,
                score_finbert=args.score_finbert,
            )
        elif args.provider == "hybrid":
            r1, c1 = ingest_symbol_gdelt(
                sym_u,
                start_date=args.from_date,
                end_date=args.to_date,
                score_finbert=args.score_finbert,
            )
            r2, c2 = ingest_symbol_sec(
                sym_u,
                start_date=args.from_date,
                end_date=args.to_date,
                score_finbert=args.score_finbert,
            )
            r, c = r1 + r2, c1 + c2
        else:
            r, c = ingest_symbol_yfinance(sym_u, score_finbert=args.score_finbert)
        logger.info("Done %s: raw upserts=%s clean rows=%s", sym_u, r, c)


if __name__ == "__main__":
    main()
