"""Qdrant collection helpers for news chunk embeddings."""

from __future__ import annotations

import os

from log_config import get_logger

logger = get_logger(__name__)

DEFAULT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_NEWS_COLLECTION", "news_chunks_v1")
DEFAULT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))


def get_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install qdrant-client (see requirements.txt)") from e
    return QdrantClient(url=DEFAULT_URL)


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
