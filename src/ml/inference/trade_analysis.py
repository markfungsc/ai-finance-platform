"""Trade-analysis orchestration for decision-support outputs."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from database.news_queries import fetch_recent_clean_news
from log_config import get_logger
from ml.sentiment.qdrant_store import retrieve_similar_news_payloads_with_meta

logger = get_logger(__name__)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def parse_llm_json(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("llm output must be object")
    for key in ("conviction_score", "adjustment", "risk_flags", "rationale"):
        if key not in payload:
            raise ValueError(f"missing key: {key}")
    if not isinstance(payload["risk_flags"], list):
        raise ValueError("risk_flags must be list")
    payload["conviction_score"] = clamp(float(payload["conviction_score"]), -1.0, 1.0)
    payload["adjustment"] = clamp(float(payload["adjustment"]), -0.10, 0.10)
    payload["rationale"] = str(payload["rationale"]).strip()
    return payload


def llm_reason(
    *,
    ticker: str,
    model_probability: float,
    threshold_used: float,
    sentiment_score: float,
    technical_summary: list[str],
    symbol_news_summary: str,
    macro_news_summary: str,
    market_regime: str | None,
) -> dict[str, Any] | None:
    mode = os.getenv("TRADE_ANALYSIS_LLM_MODE", "").strip().lower()
    if mode != "json_passthrough":
        return None
    raw = os.getenv("TRADE_ANALYSIS_LLM_JSON", "").strip()
    if not raw:
        return None
    try:
        return parse_llm_json(raw)
    except Exception:
        logger.exception("invalid llm passthrough json")
        return None


def build_trade_analysis(
    *,
    ticker: str,
    model_probability: float,
    threshold_used: float,
    sentiment_score: float,
    technical_summary: list[str],
    market_regime: str | None = None,
    top_k_news: int = 6,
    news_lookback_days: int = 7,
) -> dict[str, Any]:
    as_of = datetime.now(tz=UTC)
    news_df = fetch_recent_clean_news(
        ticker,
        since_utc=as_of - timedelta(days=max(1, int(news_lookback_days))),
        limit=max(1, int(top_k_news)),
    )
    symbol_rows = news_df.to_dict(orient="records") if not news_df.empty else []
    symbol_ids = [str(r.get("id")) for r in symbol_rows if r.get("id") is not None]
    symbol_news_summary = (
        " | ".join(
            str(r.get("title", "")).strip()
            for r in symbol_rows[:3]
            if str(r.get("title", "")).strip()
        )
        or "No recent symbol headlines."
    )

    retrieval = retrieve_similar_news_payloads_with_meta(
        symbol=ticker,
        query_text=f"{ticker} catalyst risk sentiment",
        top_k=max(1, int(top_k_news)),
    )
    qdrant_hits = list(retrieval.get("hits") or [])
    qdrant_hit_count = int(retrieval.get("hit_count") or 0)
    qdrant_error = retrieval.get("error")
    qdrant_ids = [
        str(r.get("article_id")) for r in qdrant_hits if r.get("article_id") is not None
    ]
    macro_news_summary = f"qdrant_hits={qdrant_hit_count}"

    risk_flags: list[str] = []
    if sentiment_score < -0.35:
        risk_flags.append("negative_sentiment")
    if market_regime and market_regime.lower() == "risk_off":
        risk_flags.append("risk_off_regime")
    if not symbol_rows:
        risk_flags.append("limited_symbol_news")

    sentiment_adjustment = clamp(sentiment_score * 0.04, -0.06, 0.06)
    conviction_score = clamp(
        (model_probability - threshold_used) * 1.5 + sentiment_adjustment, -1.0, 1.0
    )
    conviction_adjustment = clamp(conviction_score * 0.05, -0.08, 0.08)
    final_adjustment = clamp(sentiment_adjustment + conviction_adjustment, -0.10, 0.10)
    rationale = (
        f"Signal {'above' if model_probability > threshold_used else 'below'} threshold; "
        f"sentiment={'supportive' if sentiment_score > 0.15 else 'mixed/weak'}; "
        f"technical={', '.join(technical_summary[:3]) if technical_summary else 'none'}; "
        f"news={symbol_news_summary[:120]}"
    )

    llm = llm_reason(
        ticker=ticker,
        model_probability=model_probability,
        threshold_used=threshold_used,
        sentiment_score=sentiment_score,
        technical_summary=technical_summary,
        symbol_news_summary=symbol_news_summary,
        macro_news_summary=macro_news_summary,
        market_regime=market_regime,
    )
    if llm is not None:
        conviction_score = float(llm["conviction_score"])
        final_adjustment = float(llm["adjustment"])
        risk_flags = [str(x) for x in llm.get("risk_flags") or []]
        rationale = str(llm.get("rationale") or rationale)

    adjusted = clamp(model_probability + final_adjustment, 0.0, 1.0)
    label = (
        "high"
        if abs(conviction_score) >= 0.7
        else "medium_high"
        if abs(conviction_score) >= 0.35
        else "medium"
        if abs(conviction_score) >= 0.1
        else "low"
    )
    insufficient = len(symbol_rows) == 0 and len(qdrant_hits) == 0

    return {
        "ticker": ticker,
        "probability": float(model_probability),
        "best_threshold": float(threshold_used),
        "sentiment_snapshot": float(sentiment_score),
        "technical_context_tags": list(technical_summary),
        "news_summary": symbol_news_summary,
        "conviction_score": float(conviction_score),
        "conviction_label": label,
        "risk_flags": risk_flags,
        "adjusted_score": float(adjusted),
        "adjustment_breakdown": {
            "base_probability": float(model_probability),
            "sentiment_adjustment": float(sentiment_adjustment),
            "conviction_adjustment": float(conviction_adjustment),
            "final_adjustment": float(final_adjustment),
        },
        "grounding_refs": {
            "symbol_news_ids": symbol_ids,
            "qdrant_article_ids": qdrant_ids,
            "qdrant_hit_count": qdrant_hit_count,
            "qdrant_error": str(qdrant_error) if qdrant_error else None,
            "as_of_utc": as_of.isoformat(),
        },
        "insufficient_evidence": bool(insufficient),
        "confidence": float(0.35 if insufficient else 0.7),
        "rationale_brief": rationale[:280],
    }
