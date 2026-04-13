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
- **Inference (HTTP + UI):** FastAPI app [`src/api/main.py`](src/api/main.py); Streamlit dashboard [`src/ui/streamlit_app.py`](src/ui/streamlit_app.py). After `make up`, API is at http://localhost:8000 and the UI at http://localhost:8501 (mounts `models/` and `experiments/` from the host; run `make train` first if you have no artifact). For local dev without Compose, use `make serve-api` and `make streamlit` in separate terminals.
- **Optional news sentiment:** [`src/ml/sentiment/`](src/ml/sentiment/) — FinBERT scores from durable news sources and leakage-safe as-of attach. Current training/inference sentiment block uses daily symbol + market features (`sym_*_d1`, `spy_*_d1`) with neutral fallback.

- **Tests & lint:** `make test` (pytest), `make lint` / `make fmt` (Ruff).

Not here yet: centralized **model registry**, shared **experiment tracking** (beyond local MLflow runs), and **automated promotion** (see roadmap).



### News sentiment (optional)

1. Install NLP stack (PyTorch + Transformers): `pip install -r requirements-nlp.txt` (base `requirements.txt` includes `pyarrow` for reading/writing the cache).
2. Build cache (from repo root, `PYTHONPATH=src`): `python -m ml.sentiment --symbols AAPL MSFT` (defaults to `TRAIN_SYMBOLS`; add `--max-bars N`, `--no-score` for structure-only rows, `--output PATH`).
3. Default cache path: `data/sentiment/daily_sentiment.parquet`. Training (`load_train_dataset` / `load_dataset`) and CLI/API predict attach sentiment after market context; current primary features are daily symbol + SPY streams (`sym_sentiment_d1`, `sym_news_volume_d1`, `sym_sentiment_vol_d1`, `spy_sentiment_d1`, `spy_news_volume_d1`, `spy_sentiment_vol_d1`). **Retrain** saved models whenever feature columns change.

**Postgres + Qdrant path (preferred for durable news):** Run `make migrate` (includes [`008_news_sentiment.sql`](infra/migrations/008_news_sentiment.sql) and [`009_daily_sentiment_horizons.sql`](infra/migrations/009_daily_sentiment_horizons.sql)). Start Qdrant with `docker compose` (`finance_qdrant` on port 6333). Ingest headlines: `make news-ingest` (yfinance latest), single-file historical Kaggle backfill (`make news-backfill-kaggle KAGGLE_PATH=data/news/historical.csv SYM=AAPL`), or dual-source Kaggle union (`make news-backfill-kaggle-dual KAGGLE_SP500_PATH=data/news/sp500.csv KAGGLE_YOGESH_PATH=data/news/yogesh.csv SYM=AAPL`). Realtime/latest path remains yfinance. Free-source backfill remains available: `make news-backfill-free FROM=2015-01-01 TO=2026-03-27 SYM=AAPL PROVIDER=hybrid` (`PROVIDER`: `gdelt`, `sec`, `hybrid`). Recompute gold features (`rollup_daily`: per-symbol rolling z for horizons/volume/volatility; see `src/ml/sentiment/rollup_daily.py`): `make sentiment-rollup`. Optional embeddings: `pip install -r requirements-nlp.txt` then `make embed-news-qdrant SYM=AAPL`. `attach_sentiment_features` reads **`daily_symbol_sentiment` in Postgres first** (when `DATABASE_URL` is set), then falls back to the Parquet cache. For no-coverage periods (e.g., pre-2015), rollup writes neutral sentiment values for full feature timelines.

Kaggle schemas supported by the adapter:
- `sp500_headlines_2008_2024`: expects columns `stock`, `date`, `headline`, `url`
- `yogeshchary_financial_news`: expects columns `ticker`, `published_at`, `headline`, `summary`, `text`, `url`
- `generic_financial_news`: expects columns `symbol`, `published_at`, `title`, `summary`, `body`, `url`
- `headline_time_ticker`: expects columns `ticker`, `date`, `headline`, `summary`, `article`, `link`

Dual-source ingest deduplicates deterministically across datasets by `(symbol, minute timestamp bucket, title/url fingerprint)`, and stores provenance in raw payload (`dataset_key`, source URL, local path, local file hash).

### Explainability in Streamlit (current)

- Prediction tab: score/threshold reason text, top feature magnitudes for latest row, and global model feature importance.
- Backtest tab: omission explainability table with reason codes plus indicator/volume/sentiment context tags.
- These are fast built-in explanations (no heavy SHAP dependency) intended for operational debugging and strategy iteration.

### Recommended next step

If sentiment impact appears weak, run an ablation comparison before adding new data sources:
- baseline model (no sentiment),
- symbol-only sentiment,
- symbol + SPY daily sentiment,
- compare precision/recall, expectancy, and drawdown stability per split.

## Strategic roadmap

Status: **Delivered** · **In flight** · **Planned**

| Checkpoint | Objective | Status |
|------------|------------|--------|
| **W1** | Core data path + baseline ML artifact | **In flight** — data + features + **local** train / eval / backtest / CLI predict **delivered**; **HTTP API** in Compose **delivered**; **model registry** **planned** |
| **W2** | Inference API + service packaging | **In flight** — **Compose stack** (Postgres, API, Streamlit UI, Qdrant) **delivered**; hardened CI images and deploy **planned** |
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
├── api/                # FastAPI inference HTTP service
├── database/           # SQLAlchemy engine, query helpers
├── data_pipeline/
│   ├── ingestion/      # Market data → bronze
│   ├── processing/     # Bronze → silver
│   └── features/       # Silver → gold
├── ml/                 # Dataset, training helpers, backtest, CLI predict
├── ui/                 # Streamlit dashboard
└── scripts/            # Entrypoints (e.g. run_train)
```

Core feature computation: [`src/data_pipeline/features/build_features.py`](src/data_pipeline/features/build_features.py). ML wiring: [`src/ml/`](src/ml/).

## Requirements

- Python 3.11+
- Docker Compose (see [`infra/docker-compose.yml`](infra/docker-compose.yml))

## Local development

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

make up
make migrate   # ordered SQL under infra/migrations/

export DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/ai_finance
```

### S&P 500 universe (onboarding + backfill)

1. Fetch the current constituent list from Wikipedia into [`data/universe/sp500_symbols.txt`](data/universe/sp500_symbols.txt):
   - `make universe-fetch-sp500` (runs `python -m universe`).
2. Ingest OHLCV for **S&P 500 + market context** (QQQ, SPY, ^VIX):
   - `make ingestion-sp500` (sets `INGESTION_UNIVERSE=sp500`).
3. Silver + gold:
   - `make clean`
   - First-time full feature history: `make features-sp500-backfill` (sets `INGESTION_UNIVERSE=sp500` and `FEATURES_BACKFILL=1`). Later use `make features-sp500` for incremental updates.
4. Check coverage (optional): `make universe-preflight-sp500` or `make universe-preflight INGESTION_UNIVERSE=sp500` (defaults to `subscriptions` if unset).

**Preflight shows 505/506 (or similar)?** The command exits with status 1 until every resolved symbol has at least one `stock_features` row. The stderr output lists **symbols missing from `stock_features`** (often one thin-history name). Re-run `make features-sp500`, or for stubborn tickers `export INGESTION_UNIVERSE=sp500 FEATURES_BACKFILL=1` and run the features script so the full clean history is recomputed.

Environment:

- `INGESTION_UNIVERSE`: `subscriptions` (default) or `sp500`.
- `SP500_SYMBOLS_FILE`: path to the ticker file (default `data/universe/sp500_symbols.txt`).
- `FEATURES_BACKFILL=1`: recompute features from full `clean_stock_prices` history for every symbol in the resolved universe.

Market context (QQQ, SPY, ^VIX) is merged into training features but is **not** part of `TRAIN_SYMBOLS`. After pulling new context columns, **retrain** and refresh `FEATURE_COLUMNS_PATH` / saved models so inference matches the feature vector.

**Compose services** (from repo root, `make up` runs [`infra/docker-compose.yml`](infra/docker-compose.yml)):

| Port | Service |
|------|---------|
| 5432 | PostgreSQL (`finance_postgres`) |
| 8000 | FastAPI (`finance_api`) |
| 8501 | Streamlit (`finance_ui`) |
| 6333 | Qdrant (`finance_qdrant`) |

## Operations (Make)

| Target | Description |
|--------|-------------|
| `make up` / `make down` | Postgres, API, Streamlit UI, and Qdrant via Compose |
| `make logs` | Follow Compose logs |
| `make migrate` | Apply migration SQL to the running database |
| `make ingestion` | Incremental load → `raw_stock_prices` (uses `INGESTION_UNIVERSE`, default subscriptions) |
| `make universe-fetch-sp500` | Write `data/universe/sp500_symbols.txt` from Wikipedia |
| `make ingestion-sp500` | Ingest with `INGESTION_UNIVERSE=sp500` |
| `make universe-preflight` | Report clean/features coverage; pass `INGESTION_UNIVERSE=sp500` or it defaults to `subscriptions` |
| `make universe-preflight-sp500` | Same with `INGESTION_UNIVERSE=sp500` (no variable on the command line) |
| `make features-sp500` | Gold features with `INGESTION_UNIVERSE=sp500` (incremental) |
| `make features-sp500-backfill` | Full feature recompute for SP500 universe |
| `make features-backfill` | Full feature recompute for current `INGESTION_UNIVERSE` |
| `make clean` | Silver transformation |
| `make features` | Gold feature build / upsert |
| `make train` | Baseline ML on `stock_features` (Postgres must be up; run ingestion/clean/features first if tables are empty) |
| `make backtest` | Run backtest driver on loaded dataset + saved model |
| `make walk-forward` | Walk-forward backtest test script |
| `make predict` | CLI inference script (`src/ml/inference/predict.py`) |
| `make serve-api` | Run FastAPI locally on port 8000 (no Docker) |
| `make streamlit` | Run Streamlit locally on port 8501 (expects API reachable; set `API_BASE_URL` / `INFERENCE_API_BASE_URL` if needed) |
| `make test` | Pytest (`tests/`) |
| `make lint` / `make fmt` | Ruff |

## Engineering notes

- Streaming ingestion with batched writes; feature upserts executed in transactions.
- No lookahead in engineered features; incremental feature runs use a bounded history window for rolling statistics.
- Baseline backtesting and walk-forward helpers live under `src/ml/backtest/`; they reuse the same dataset merge and label semantics as training.
- Optional extension: high-throughput ingest, low-latency inference, or simulation/backtesting in a systems language alongside this Python stack.

## License

TBD
