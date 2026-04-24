"""Qdrant collection helpers for news chunk embeddings."""

from __future__ import annotations

import os
from functools import lru_cache

from log_config import get_logger

logger = get_logger(__name__)

DEFAULT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_NEWS_COLLECTION", "news_chunks_v1")
DEFAULT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
DEFAULT_EMBED_MODEL = os.getenv(
    "QDRANT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def get_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install qdrant-client (see requirements.txt)") from e
    return QdrantClient(url=DEFAULT_URL)


@lru_cache(maxsize=1)
def _get_embed_model():
    """Load and memoize embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model for retrieval: %s", DEFAULT_EMBED_MODEL)
    return SentenceTransformer(DEFAULT_EMBED_MODEL)


def ensure_news_collection(
    client=None,
    *,
    collection_name: str = DEFAULT_COLLECTION,
    vector_size: int = DEFAULT_VECTOR_SIZE,
):
    from qdrant_client.models import Distance, VectorParams

    c = client or get_client()
    cols = [x.name for x in (c.get_collections().collections or [])]
    if collection_name in cols:
        return c
    logger.info("Creating Qdrant collection %s dim=%s", collection_name, vector_size)
    c.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    return c


def retrieve_similar_news_payloads_with_meta(
    *,
    symbol: str,
    query_text: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict:
    """Retrieve similar news chunk payloads from Qdrant.

    Returns a structured payload with:
      - hits: list[dict]
      - hit_count: int
      - error: str | None

    Runtime/dependency failures do not raise; they are captured in `error` so
    callers can preserve deterministic fallbacks while exposing diagnostics.
    """
    try:
        client = get_client()
        model = _get_embed_model()
        vec = model.encode(query_text, normalize_embeddings=True).tolist()
        filt = {"must": [{"key": "symbol", "match": {"value": symbol.upper()}}]}
        # qdrant-client renamed search/query APIs across versions; support both.
        if hasattr(client, "search"):
            hits = client.search(
                collection_name=collection_name,
                query_vector=vec,
                limit=max(1, int(top_k)),
                query_filter=filt,
            )
        else:
            qr = client.query_points(
                collection_name=collection_name,
                query=vec,
                limit=max(1, int(top_k)),
                query_filter=filt,
            )
            hits = list(getattr(qr, "points", []) or [])
        out: list[dict] = []
        for h in hits:
            payload = getattr(h, "payload", None) or {}
            score = getattr(h, "score", None)
            if isinstance(payload, dict):
                row = dict(payload)
                if score is not None:
                    row["score"] = float(score)
                out.append(row)
        return {"hits": out, "hit_count": len(out), "error": None}
    except Exception as e:
        logger.debug("qdrant retrieval failed for symbol=%s", symbol, exc_info=True)
        return {"hits": [], "hit_count": 0, "error": f"qdrant_query_failed:{e}"}


def retrieve_similar_news_payloads(
    *,
    symbol: str,
    query_text: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict]:
    """Backward-compatible wrapper returning only the hit payload list."""
    result = retrieve_similar_news_payloads_with_meta(
        symbol=symbol,
        query_text=query_text,
        top_k=top_k,
        collection_name=collection_name,
    )
    return list(result.get("hits") or [])
