# AI Finance Platform

Multi-layer market analytics platform: curated price data, feature store semantics, and a staged path to model serving, experimentation, and production MLOps.

## Overview

The system implements a **medallion-style data architecture** in PostgreSQL—raw ingest, validated analytics-ready tables, and a feature table keyed for downstream ML. Near-term priority is a **thin vertical slice**: baseline model, HTTP inference, and containerized services, then successive releases for experiment tracking, alternative data (sentiment), sequence models, RAG-based research assistants, orchestrated training, and hardened production operations.

## Current scope

| Layer | Relation | Function |
|-------|-----------|----------|
| Bronze | `raw_stock_prices` | OHLCV ingest (yfinance), `TIMESTAMPTZ`, idempotent upserts |
| Silver | `clean_stock_prices` | Deduplication, null/invalid filtering, stable schema for analytics |
| Gold | `stock_features` | Returns, rolling statistics, volatility, lags—backward-looking only |

Incremental pipelines support multi-symbol loads and time-bounded feature rebuilds (lookback window for rolling correctness). Schema and migrations: [`infra/postgres/`](infra/postgres/), [`infra/migrations/`](infra/migrations/).

## Strategic roadmap

Status: **Delivered** · **In flight** · **Planned**

| Checkpoint | Objective | Status |
|------------|------------|--------|
| **W1** | Core data path + baseline ML artifact | **In flight** — ETL and feature layer **delivered** for equities; training/serving **planned** |
| **W2** | Inference API + service packaging | **In flight** — database containerized; application API and full stack images **planned** |
| **W4** | Experiment tracking & reproducibility | **Planned** — MLflow-class runs, metrics, model lineage |
| **W6** | Transformer-based financial sentiment | **Planned** — news → scores → joinable features |
| **W8** | Time-series / gradient-boosted forecasting | **Planned** — comparative evaluation under same tracking layer |
| **W10** | RAG assistant over curated documents | **Planned** — retrieval + LLM, guardrails |
| **W12** | Automated training & promotion | **Planned** — scheduled pipelines (e.g. Airflow-class) |
| **W16** | Production operations | **Planned** — observability, scaling, policy-driven deploy |

### Phase narrative

1. **Foundation (now → W2)** — Lock the data contract, ship a minimal **train → register → predict** loop behind an API, standardize containers for local and CI.
2. **Experimentation (W3–W4)** — Centralize parameters, metrics, and artifacts; support A/B model comparison and dataset snapshots.
3. **Enriched signals (W5–W8)** — Add NLP sentiment and stronger sequence/tabular predictors; all models registered and comparable.
4. **Intelligence layer (W9–W10)** — RAG over filings/news with evaluation harnesses.
5. **Automation & production (W11–W16)** — Orchestrated retraining, multi-service deploy, monitoring, and SLO-oriented operations.

### Delivery model

Releases prioritize **end-to-end slices** (minimal model + pipeline + deploy + observe) over long periods of offline-only modeling. Each phase deepens one platform layer while keeping the full path runnable.

### Target reference architecture

```mermaid
flowchart LR
  ingest[Ingestion]
  feats[Feature store]
  train[Training]
  exp[Experiment tracking]
  reg[Model registry]
  api[Inference APIs]
  apps[Applications]
  ingest --> feats --> train
  train --> exp --> reg --> api --> apps
```

## Repository layout

```
src/
├── database/           # SQLAlchemy engine, query helpers
├── data_pipeline/
│   ├── ingestion/      # Market data → bronze
│   ├── processing/     # Bronze → silver
│   └── features/       # Silver → gold
└── scripts/
```

Core feature computation: [`src/data_pipeline/features/build_features.py`](src/data_pipeline/features/build_features.py).

## Requirements

- Python 3.11+
- Docker Compose (PostgreSQL service)

## Local development

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

make up
make migrate   # ordered SQL under infra/migrations/

export DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/ai_finance
```

## Operations (Make)

| Target | Description |
|--------|-------------|
| `make up` / `make down` | PostgreSQL via Compose |
| `make migrate` | Apply migration SQL to the running database |
| `make ingestion` | Incremental load → `raw_stock_prices` |
| `make clean` | Silver transformation |
| `make features` | Gold feature build / upsert |
| `make lint` / `make fmt` | Ruff |

## Engineering notes

- Streaming ingestion with batched writes; feature upserts executed in transactions.
- No lookahead in engineered features; incremental feature runs use a bounded history window for rolling statistics.
- Optional extension: high-throughput ingest, low-latency inference, or simulation/backtesting in a systems language alongside this Python stack.

## License

TBD
