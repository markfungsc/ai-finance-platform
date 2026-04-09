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

## Progress

What works end-to-end today (local, Postgres-backed):

- **Pipelines:** `make ingestion` → `make clean` → `make features` into `stock_features` (returns, rolling stats, lags; z-score columns for ML).
- **ML dataset:** [`src/ml/dataset.py`](src/ml/dataset.py) loads raw + z features, merges on `(symbol, timestamp)`, and builds a **forward** label from `return_5d` with a configurable shift (default 5 bars) in [`src/ml/helpers/merge_features.py`](src/ml/helpers/merge_features.py).
- **Training:** `make train` runs [`src/scripts/run_train.py`](src/scripts/run_train.py) (scikit-learn RandomForest, joblib artifact under `models/`).
- **Evaluation & backtest:** metrics via [`src/ml/evaluate.py`](src/ml/evaluate.py); `make backtest` and `make walk-forward` exercise [`src/ml/backtest/`](src/ml/backtest/).
- **Inference (CLI):** `make predict` runs [`src/ml/inference/predict.py`](src/ml/inference/predict.py) (optional merge debug output for inspection).
- **Optional news sentiment:** [`src/ml/sentiment/`](src/ml/sentiment/) — FinBERT scores on yfinance headlines, rolled to `news_sentiment_mean_z` and joined in training/predict when a Parquet cache exists; missing cache → neutral `0.0`. See **News sentiment (optional)** below.

- **Tests & lint:** `make test` (pytest), `make lint` / `make fmt` (Ruff).

Not here yet: HTTP inference API, containerized app service, experiment tracking, and automated promotion (see roadmap).



### News sentiment (optional)

1. Install NLP stack (PyTorch + Transformers): `pip install -r requirements-nlp.txt` (base `requirements.txt` includes `pyarrow` for reading/writing the cache).
2. Build cache (from repo root, `PYTHONPATH=src`): `python -m ml.sentiment --symbols AAPL MSFT` (defaults to `TRAIN_SYMBOLS`; add `--max-bars N`, `--no-score` for structure-only rows, `--output PATH`).
3. Default cache path: `data/sentiment/daily_sentiment.parquet`. Training (`load_train_dataset` / `load_dataset`) and CLI predict attach `news_sentiment_mean_z` after market context; **retrain** saved models so `FEATURE_COLUMNS` joblibs match the widened feature matrix.

**Postgres + Qdrant path (preferred for durable news):** Run `make migrate` (includes [`008_news_sentiment.sql`](infra/migrations/008_news_sentiment.sql) and [`009_daily_sentiment_horizons.sql`](infra/migrations/009_daily_sentiment_horizons.sql)). Start Qdrant with `docker compose` (`finance_qdrant` on port 6333). Ingest headlines: `make news-ingest` (yfinance latest), single-file historical Kaggle backfill (`make news-backfill-kaggle KAGGLE_PATH=data/news/historical.csv SYM=AAPL`), or dual-source Kaggle union (`make news-backfill-kaggle-dual KAGGLE_SP500_PATH=data/news/sp500.csv KAGGLE_YOGESH_PATH=data/news/yogesh.csv SYM=AAPL`). Realtime/latest path remains yfinance. Free-source backfill remains available: `make news-backfill-free FROM=2015-01-01 TO=2026-03-27 SYM=AAPL PROVIDER=hybrid` (`PROVIDER`: `gdelt`, `sec`, `hybrid`). Recompute gold features (`rollup_daily`: per-symbol rolling z for horizons/volume/volatility; see `src/ml/sentiment/rollup_daily.py`): `make sentiment-rollup`. Optional embeddings: `pip install -r requirements-nlp.txt` then `make embed-news-qdrant SYM=AAPL`. `attach_sentiment_features` reads **`daily_symbol_sentiment` in Postgres first** (when `DATABASE_URL` is set), then falls back to the Parquet cache. For no-coverage periods (e.g., pre-2015), rollup writes neutral sentiment values for full feature timelines.

Kaggle schemas supported by the adapter:
- `sp500_headlines_2008_2024`: expects columns `stock`, `date`, `headline`, `url`
- `yogeshchary_financial_news`: expects columns `ticker`, `published_at`, `headline`, `summary`, `text`, `url`
- `generic_financial_news`: expects columns `symbol`, `published_at`, `title`, `summary`, `body`, `url`
- `headline_time_ticker`: expects columns `ticker`, `date`, `headline`, `summary`, `article`, `link`

Dual-source ingest deduplicates deterministically across datasets by `(symbol, minute timestamp bucket, title/url fingerprint, content hash)`, and stores provenance in raw payload (`dataset_key`, source URL, local path, local file hash).

## Strategic roadmap

Status: **Delivered** · **In flight** · **Planned**

| Checkpoint | Objective | Status |
|------------|------------|--------|
| **W1** | Core data path + baseline ML artifact | **In flight** — data + features + **local** train / eval / backtest / CLI predict **delivered**; **serving API** and registry **planned** |
| **W2** | Inference API + service packaging | **In flight** — database containerized; **HTTP inference** and full stack images **planned** |
| **W4** | Experiment tracking & reproducibility | **Planned** — MLflow-class runs, metrics, model lineage |
| **W6** | Transformer-based financial sentiment | **In flight** — FinBERT + cache + `news_sentiment_mean_z` in dataset/predict; full data coverage TBD |
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
├── ml/                 # Dataset, training helpers, backtest, CLI predict
└── scripts/            # Entrypoints (e.g. run_train)
```

Core feature computation: [`src/data_pipeline/features/build_features.py`](src/data_pipeline/features/build_features.py). ML wiring: [`src/ml/`](src/ml/).

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
| `make train` | Baseline ML on `stock_features` (Postgres must be up; run ingestion/clean/features first if tables are empty) |
| `make backtest` | Run backtest driver on loaded dataset + saved model |
| `make walk-forward` | Walk-forward backtest test script |
| `make predict` | CLI inference script (`src/ml/inference/predict.py`) |
| `make test` | Pytest (`tests/`) |
| `make lint` / `make fmt` | Ruff |

## Engineering notes

- Streaming ingestion with batched writes; feature upserts executed in transactions.
- No lookahead in engineered features; incremental feature runs use a bounded history window for rolling statistics.
- Baseline backtesting and walk-forward helpers live under `src/ml/backtest/`; they reuse the same dataset merge and label semantics as training.
- Optional extension: high-throughput ingest, low-latency inference, or simulation/backtesting in a systems language alongside this Python stack.

## License

TBD
