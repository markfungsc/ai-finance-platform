# Operations Guide

Practical runbook for day-to-day execution: setup, scanner/API workflows, experiments, and sentiment/news operations.

## What You Can Run Today

- Medallion-style market data path in Postgres: `raw_stock_prices` -> `clean_stock_prices` -> `stock_features` / `stock_features_zscore`
- Model training and walk-forward/backtest tooling
- FastAPI inference service with explainability and scanner endpoints
- Streamlit dashboard with Predict, Scanner, Threshold Grid, and Backtest tabs
- Experiment runner that writes threshold artifacts consumed by API/UI
- Optional news sentiment ingestion, daily rollups, cache, and Qdrant embeddings

## Quickstart

### 1) Local Python workflow (fastest for dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

export DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/ai_finance
```

Run infra and migrations:

```bash
make up
make migrate
```

Run API and UI in separate terminals:

```bash
make serve-api
make streamlit
```

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

### 2) Data -> model -> experiments flow

```bash
make ingestion
make clean
make features
make train
make experiments
make view-results
```

## S&P 500 Universe Flow

```bash
make universe-fetch-sp500
make ingestion-sp500
make clean
make features-sp500-backfill   # first full history build
make features-sp500            # later incremental updates
make universe-preflight-sp500  # optional coverage check
```

Key variables:

- `INGESTION_UNIVERSE`: `subscriptions` (default) or `sp500`
- `SP500_SYMBOLS_FILE`: default `data/universe/sp500_symbols.txt`
- `FEATURES_BACKFILL=1`: force full feature rebuild from clean history

## Stock Scanner Workflow

Scanner uses an async two-step lifecycle:

1. Refresh market/features data for the resolved ingestion universe
2. Run ranking scan on pooled symbols

The UI button "Run scanner" calls:

- `POST /scanner/refresh/start`
- `GET /scanner/refresh/status` (poll until `succeeded` or `skipped_up_to_date`)
- `POST /scanner/scan/start`
- `GET /scanner/scan/status` (poll until `succeeded`)

Scanner controls:

- `top_n` (1-50)
- `max_symbols` (optional cap)
- `max_workers` (optional, bounded server-side)
- `min_probability` (optional filter)

Important behavior:

- Scan requires refresh completion first; otherwise returns `503`
- Symbols with stale or missing data are skipped and reported
- If refresh universe coverage does not match scanner pool (for example, not using SP500 when expected), scan returns `503` with guidance

## API Endpoints (Core)

- `GET /health`
- `POST /predict_symbol`
- `POST /predict_symbol_explain`
- `GET /threshold_grid`
- `POST /scanner/refresh/start`
- `GET /scanner/refresh/status`
- `POST /scanner/scan/start`
- `GET /scanner/scan/status`
- `POST /scan_symbols` (synchronous compatibility path)
- `GET /backtest/indicators`
- `GET /backtest/trades`

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict_symbol_explain \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL"}'
```

## Experiments and Artifacts

`make experiments` runs pooled backtesting/optimization and writes threshold artifacts used by API and Streamlit.

Default cutoff enforcement:

- `EXPERIMENT_END_DATE` defaults to `2026-03-27` in `make experiments`
- The command passes `--end-date ${EXPERIMENT_END_DATE}` to `run_experiment.py`
- Data loading applies inclusive filter `timestamp <= end_date`

Artifacts written:

- `models/swing-trade/random_forest_pooled_best_threshold.json`
- `models/swing-trade/random_forest_pooled_threshold_grid.json`
- `experiments/artifacts/swing-trade/pooled_random_forest/threshold_grid.json`
- split-level files under `experiments/artifacts/swing-trade/pooled_random_forest/split_*/`
- `experiments/results.csv`

Use:

```bash
make experiments
make view-results
```

## News and Sentiment Pipeline (Optional)

Ingestion and backfills:

- `make news-ingest`
- `make news-backfill-free FROM=2015-01-01 TO=2026-03-27 SYM=AAPL PROVIDER=hybrid`
- `make news-backfill-free-finbert`
- `make news-backfill-kaggle KAGGLE_PATH=data/news/historical.csv`
- `make news-backfill-kaggle-finbert`
- `make news-backfill-kaggle-dual`
- `make news-backfill-kaggle-dual-finbert`

Aggregation and downstream:

- `make sentiment-rollup`
- `make sentiment-cache`
- `make embed-news-qdrant SYM=AAPL`

Typical sentiment data surfaces:

- Postgres: `raw_news_articles`, `clean_news_articles`, `daily_symbol_sentiment`
- Local cache: `data/sentiment/daily_sentiment.parquet`
- Vector store: Qdrant collection (default `news_chunks_v1`)

Useful knobs:

- `FROM`, `TO`, `SYM`, `PROVIDER`
- `KAGGLE_PATH`, `KAGGLE_KEY`, `KAGGLE_SP500_PATH`, `KAGGLE_YOGESH_PATH`
- `HEARTBEAT`, `CONCURRENCY`, `TIMEOUT`, `RETRY_MAX`
- `QDRANT_URL`, `QDRANT_NEWS_COLLECTION`, `QDRANT_VECTOR_SIZE`

## Operations (Make Targets)

| Target | Purpose |
|---|---|
| `make up` / `make down` / `make logs` | Docker compose lifecycle |
| `make migrate` | Apply SQL migrations |
| `make universe-fetch-sp500` | Fetch SP500 universe file |
| `make universe-preflight` / `make universe-preflight-sp500` | Coverage checks by universe mode |
| `make ingestion` / `make ingestion-sp500` | Raw market ingest |
| `make clean` | Build/refresh silver table |
| `make features` / `make features-sp500` | Incremental feature build |
| `make features-backfill` / `make features-sp500-backfill` | Full feature recompute |
| `make train` | Train model artifacts |
| `make backtest` / `make walk-forward` | Backtest execution paths |
| `make experiments` / `make view-results` | Experiment runs and result viewing |
| `make predict` | CLI inference |
| `make serve-api` / `make streamlit` | Local API/UI serving |
| `make news-ingest` and `make news-backfill-*` | News ingestion/backfill |
| `make sentiment-rollup` / `make sentiment-cache` | Sentiment aggregation/cache |
| `make embed-news-qdrant` | Push embeddings to Qdrant |
| `make test` | Run pytest |
| `make lint` / `make fmt` | Ruff checks/format |

## Environment Variables (Most Used)

Core runtime:

- `DATABASE_URL`
- `LOG_LEVEL`

API artifacts and model loading:

- `MODEL_PATH`
- `FEATURE_COLUMNS_PATH`
- `SCALER_PATH` (optional)
- `BEST_THRESHOLD_PATH` (optional override)
- `THRESHOLD_GRID_PATH` (optional override)

Scanner and inference behavior:

- `SCAN_MAX_WORKERS`
- `PREDICT_UI_MAX_BARS`
- `SCANNER_STATUS_POLL_SECONDS` (used by UI polling)
- `MARKET_CLOSE_UTC_HOUR` (fallback close-time heuristic)

Experiment runner:

- `EXPERIMENT_END_DATE` (default enforced to `2026-03-27` in `make experiments`)
- `THRESHOLD_SELECTION_MODE`
- `THRESHOLD_MULTI_TOP_K_START`
- `THRESHOLD_MULTI_TOP_K_MAX`
- `THRESHOLD_MULTI_METRICS`

UI/API connectivity and browser access:

- `API_BASE_URL` (Streamlit -> API)
- `CORS_ALLOW_ORIGINS` (API)

Backtest artifact browsing:

- `BACKTEST_ARTIFACTS_ROOT` (UI local path)
