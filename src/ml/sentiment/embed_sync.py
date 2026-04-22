"""Embed ``clean_news_articles`` text into Qdrant (optional sentence-transformers)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.connection import engine
from log_config import get_logger
from ml.sentiment.qdrant_store import (
    DEFAULT_COLLECTION,
    DEFAULT_VECTOR_SIZE,
    ensure_news_collection,
    get_client,
)
from universe.resolve import resolve_ingestion_universe

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _chunks(text: str, max_chars: int = 500) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    return [t[i : i + max_chars] for i in range(0, len(t), max_chars)]


def _point_id(article_id: int, chunk_idx: int) -> int:
    return article_id * 10_000 + chunk_idx


def embed_and_upsert_symbol(symbol: str, *, limit: int | None = None) -> int:
    try:
        from qdrant_client.models import PointStruct
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Install: pip install qdrant-client sentence-transformers (add to requirements-nlp.txt)"
        ) from e

    q = text("""
        SELECT id, title, summary
        FROM clean_news_articles
        WHERE symbol = :symbol
        ORDER BY id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"symbol": symbol.upper()})
    if df.empty:
        return 0
    if limit is not None:
        df = df.head(int(limit))

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    if dim != DEFAULT_VECTOR_SIZE:
        logger.warning(
            "Model dim %s != DEFAULT_VECTOR_SIZE %s", dim, DEFAULT_VECTOR_SIZE
        )

    client = get_client()
    ensure_news_collection(client, vector_size=dim)

    points = []
    for row in df.itertuples(index=False):
        text_in = f"{row.title}. {row.summary}".strip()
        for ci, ch in enumerate(_chunks(text_in)):
            pid = _point_id(int(row.id), ci)
            vec = model.encode(ch, normalize_embeddings=True)
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            points.append(
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "article_id": int(row.id),
                        "chunk_idx": ci,
                        "symbol": symbol.upper(),
                        "model": MODEL_NAME,
                    },
                )
            )

    batch = 64
    for i in range(0, len(points), batch):
        client.upsert(collection_name=DEFAULT_COLLECTION, points=points[i : i + batch])
    logger.info("Upserted %s points for %s", len(points), symbol)
    return len(points)


def embed_and_upsert_all_symbols(
    *,
    start_at: str | None = None,
    limit: int | None = None,
) -> int:
    _mode, symbols = resolve_ingestion_universe()
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if start_at:
        start = start_at.strip().upper()
        if start not in symbols:
            raise ValueError(f"--start-at symbol not found in resolved universe: {start}")
        symbols = symbols[symbols.index(start) :]
    if limit is not None:
        symbols = symbols[: max(0, int(limit))]
    if not symbols:
        logger.warning("No symbols resolved for embed --all")
        return 0

    total_points = 0
    total_symbols = len(symbols)
    logger.info("Embedding universe symbols: count=%d", total_symbols)
    for idx, sym in enumerate(symbols, start=1):
        logger.info("[%d/%d] embedding %s", idx, total_symbols, sym)
        total_points += int(embed_and_upsert_symbol(sym))
    logger.info(
        "Completed universe embedding: symbols=%d points=%d",
        total_symbols,
        total_points,
    )
    return total_points


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Embed clean news into Qdrant")
    ap.add_argument("--symbol", help="Ticker symbol")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Embed all symbols from resolve_ingestion_universe()",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--start-at",
        default=None,
        help="When --all, start embedding from this symbol in resolved order",
    )
    args = ap.parse_args(argv)
    if args.all and args.symbol:
        raise SystemExit("Pass either --symbol or --all, not both")
    if not args.all and not args.symbol:
        raise SystemExit("Pass --symbol <TICKER> or --all")
    if args.all:
        embed_and_upsert_all_symbols(start_at=args.start_at, limit=args.limit)
    else:
        embed_and_upsert_symbol(args.symbol, limit=args.limit)


if __name__ == "__main__":
    main()
