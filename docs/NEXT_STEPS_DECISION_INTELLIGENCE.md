# Next Steps: Decision Intelligence

This document defines the **next strategic build phase**. It is a roadmap and design target only.

## Scope and status

- **Implemented today:** scanner, RF probability outputs, threshold optimization artifacts, sentiment ingestion/attach pipeline, API/UI delivery paths.
- **Proposed next (not implemented yet):** LLM-powered decision augmentation layer for trade conviction, risk flags, and regime-aware adjustments.

No code/API behavior in this document should be interpreted as already shipped.

## Why this is the next move

The platform already covers data engineering, feature pipelines, model scoring, backtesting, scanner ranking, and sentiment ingestion. The biggest remaining product gap is not “another standalone model,” but a robust **decision-support layer** answering:

- Why should this signal be trusted today?
- What current news context strengthens or weakens conviction?
- Which risk flags should gate or down-weight a candidate?

## Design principle

Use LLMs for **decision augmentation**, not direct return prediction.

- Good use: summarization, narrative extraction, risk flagging, human-readable rationale, regime labeling.
- Avoid overclaim: “LLM predicts price better than model.”

## Proposed phases

## Phase 1: Trade explanation engine

Goal: produce structured, auditable explanations per candidate.

Outputs per ticker (target fields):

- `probability`
- `best_threshold`
- `sentiment_snapshot`
- `technical_context`
- `news_summary`
- `risk_flags`
- `conviction_label`
- `adjusted_score`
- `adjustment_breakdown`

## Phase 2: News-to-signal reasoning

Goal: score whether recent narrative should reinforce or weaken the base signal.

Concept:

- Retrieve top relevant recent news from vector search + recency filters.
- Summarize themes and catalyst strength.
- Produce bounded sentiment/conviction adjustment with reasoning.

## Phase 3: Regime-aware adaptation

Goal: add market-regime context to execution policy.

Examples:

- `risk_on`: keep baseline threshold behavior.
- `risk_off`: raise effective trade threshold.
- Event-heavy regime: increase caution, lower conviction, elevate risk flags.

## 2-week milestone target

Deliver a documentation-driven API design and implementation-ready contract for:

- `POST /trade-analysis`

### Draft request

```json
{
  "ticker": "NVDA",
  "horizon_days": 5,
  "include_regime": true
}
```

### Draft response

```json
{
  "ticker": "NVDA",
  "probability": 0.74,
  "best_threshold": 0.66,
  "sentiment": 0.58,
  "regime": "risk_on",
  "summary": "AI demand narrative remains supportive with event-risk caveats.",
  "conviction": "medium_high",
  "risk_flags": ["earnings_within_48h"],
  "adjusted_score": 0.78,
  "adjustment_breakdown": {
    "base_probability": 0.74,
    "sentiment_adjustment": 0.03,
    "llm_conviction_adjustment": 0.01
  },
  "grounding": {
    "top_news_ids": ["..."],
    "as_of_utc": "2026-04-21T00:00:00Z"
  }
}
```

## Guardrails and quality requirements

- Keep every adjustment bounded and explicitly attributable.
- Require grounded outputs (source IDs/snippets and timestamps).
- Include confidence labeling for narrative conclusions.
- Preserve deterministic fallback behavior when LLM/retrieval is unavailable.
- Do not represent qualitative output as guaranteed alpha.

## Success criteria for this phase

- Clear separation of current capabilities vs proposed architecture.
- Implementation-ready API contract and phased execution plan.
- Product narrative supports recruiter/stakeholder understanding of platform maturity.

## Suggested implementation touchpoints (later)

These are **future implementation targets**, listed for planning only:

- API orchestration: `src/api/main.py`
- Decision layer module (new): `src/ml/inference/trade_analysis.py`
- Retrieval/sentiment integration: existing modules under `src/ml/sentiment/`
- UI surfacing for rationale/risk: `src/ui/streamlit_app.py`
