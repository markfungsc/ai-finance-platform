"""Lazy FinBERT (ProsusAI/finbert) sentiment scoring."""

from __future__ import annotations

from typing import Any

from log_config import get_logger

_PIPELINE: Any = None
MODEL_NAME = "ProsusAI/finbert"


logger = get_logger(__name__)


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        try:
            from transformers import pipeline
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "Install NLP extras: pip install -r requirements-nlp.txt"
            ) from e
        _PIPELINE = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            truncation=True,
            max_length=512,
        )
    return _PIPELINE


def label_to_signed_score(result: dict) -> float:
    """Map FinBERT label + confidence to roughly [-1, 1]."""
    lab = str(result.get("label", "")).lower()
    s = float(result.get("score", 0.5))
    if "positive" in lab:
        return s
    if "negative" in lab:
        return -s
    return 0.0


def score_texts(texts: list[str], *, batch_size: int = 8) -> list[float]:
    if not texts:
        return []
    pipe = _get_pipeline()
    out: list[float] = []
    for i in range(0, len(texts), batch_size):
        chunk = [t[:4000] for t in texts[i : i + batch_size]]
        raw = pipe(chunk)
        logger.info("Raw: %s", raw)
        if isinstance(raw, dict):
            raw = [raw]
        out.extend(label_to_signed_score(r) for r in raw)
    return out


def score_text(text: str) -> float:
    return score_texts([text])[0] if text.strip() else 0.0
