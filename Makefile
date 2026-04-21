.PHONY: lint fmt up down logs migrate universe-fetch-sp500 universe-preflight universe-preflight-sp500 ingestion ingestion-sp500 clean features features-sp500 features-backfill features-sp500-backfill train walk-forward backtest experiments view-results predict serve-api streamlit test activate-vm news-ingest news-backfill-free news-backfill-free-finbert news-backfill-kaggle news-backfill-kaggle-finbert news-backfill-kaggle-dual news-backfill-kaggle-dual-finbert sentiment-rollup embed-news-qdrant

up:
	cd infra && docker compose up -d

down:
	cd infra && docker compose down

logs:
	cd infra && docker compose logs -f

migrate:
	for f in infra/migrations/*.sql; do \
		echo Running $$f; \
		cd infra && docker exec -i finance_postgres psql -U postgres -d ai_finance < $$f; \
	done

# Fetch current S&P 500 tickers from Wikipedia → data/universe/sp500_symbols.txt
universe-fetch-sp500:
	export PYTHONPATH=src && python -m universe

# DB coverage vs resolved INGESTION_UNIVERSE (requires Postgres + migrate).
# Default universe is subscriptions; override: ``make universe-preflight INGESTION_UNIVERSE=sp500``
universe-preflight:
	export PYTHONPATH=src && \
	export INGESTION_UNIVERSE="$(or $(INGESTION_UNIVERSE),subscriptions)" && \
	python -m universe.preflight

# Same as preflight with INGESTION_UNIVERSE=sp500 (no need to pass on the command line)
universe-preflight-sp500:
	export PYTHONPATH=src && export INGESTION_UNIVERSE=sp500 && python -m universe.preflight

ingestion:
	export PYTHONPATH=src && python src/data_pipeline/ingestion/run_ingestion.py

# Same as ingestion with INGESTION_UNIVERSE=sp500 (needs universe-fetch-sp500 first)
ingestion-sp500:
	export PYTHONPATH=src && export INGESTION_UNIVERSE=sp500 && python src/data_pipeline/ingestion/run_ingestion.py

clean:
	export PYTHONPATH=src && python src/data_pipeline/processing/clean_prices.py
	
features:
	export PYTHONPATH=src && python src/data_pipeline/features/build_features.py

features-sp500:
	export PYTHONPATH=src && export INGESTION_UNIVERSE=sp500 && python src/data_pipeline/features/build_features.py

# Full recompute features from clean history for all symbols in current universe
features-backfill:
	export PYTHONPATH=src && export FEATURES_BACKFILL=1 && python src/data_pipeline/features/build_features.py

# First SP500 historical feature build: same as features-backfill with INGESTION_UNIVERSE=sp500
features-sp500-backfill:
	export PYTHONPATH=src && export INGESTION_UNIVERSE=sp500 && export FEATURES_BACKFILL=1 && python src/data_pipeline/features/build_features.py

train:
	export PYTHONPATH=src && python src/scripts/run_train.py

walk-forward:
	export PYTHONPATH=src && python src/ml/backtest/test_walk_forward.py

backtest:
	export PYTHONPATH=src && python src/ml/backtest/run_backtest.py

experiments:
	export PYTHONPATH=src && export THRESHOLD_SELECTION_MODE=multi_top_k && export EXPERIMENT_END_DATE=$${EXPERIMENT_END_DATE:-2026-03-27} && python src/ml/experiments/run_experiment.py --end-date $${EXPERIMENT_END_DATE}

view-results:
	export PYTHONPATH=src && python src/ml/experiments/view_results.py

predict:
	export PYTHONPATH=src && python src/ml/inference/predict.py

sentiment-cache:
	export PYTHONPATH=src && python src/ml/sentiment/__main__.py

news-ingest:
	export PYTHONPATH=src && python src/data_pipeline/news/ingest.py --score-finbert

news-backfill-free:
	@echo 'Usage: make news-backfill-free FROM=2015-01-01 TO=2026-03-27 SYM=AAPL PROVIDER=hybrid' && export PYTHONPATH=src && python src/data_pipeline/news/ingest.py --provider $${PROVIDER:-hybrid} --score-finbert --from-date $${FROM:-2015-01-01} --to-date $${TO:-$$(date +%F)} --symbols $${SYM:-AAPL}

news-backfill-free-finbert:
	@echo 'Usage: make news-backfill-free-finbert FROM=2015-01-01 TO=2026-03-27 PROVIDER=hybrid'
	@export PYTHONPATH=src && python src/data_pipeline/news/ingest.py \
		--provider $${PROVIDER:-gdelt} \
		--score-finbert \
		--from-date $${FROM:-2015-01-01} \
		--to-date $${TO:-$$(date +%F)} \
		--heartbeat-seconds $${HEARTBEAT:-10} \
		--max-concurrency $${CONCURRENCY:-1} \
		--request-timeout $${TIMEOUT:-30} \
		--retry-max $${RETRY_MAX:-6}

news-backfill-kaggle:
	@echo 'Usage: make news-backfill-kaggle KAGGLE_PATH=data/news/historical.csv [KAGGLE_KEY=generic_financial_news] [SYM=AAPL] [FROM=2015-01-01] [TO=2026-03-27]'
	@export PYTHONPATH=src && python src/data_pipeline/news/ingest.py \
		--provider kaggle \
		--kaggle-dataset-path $${KAGGLE_PATH} \
		--kaggle-dataset-key $${KAGGLE_KEY:-generic_financial_news} \
		--from-date $${FROM:-2015-01-01} \
		--to-date $${TO:-$$(date +%F)} \
		--symbols $${SYM:-SPY}

news-backfill-kaggle-finbert:
	@echo 'Usage: make news-backfill-kaggle-finbert KAGGLE_PATH=data/news/sp500.csv [KAGGLE_KEY=sp500_headlines_2008_2024] [FROM=2008-01-01] [TO=2024-12-31]'
	@export PYTHONPATH=src && python src/data_pipeline/news/ingest.py \
		--provider kaggle \
		--kaggle-dataset-path $${KAGGLE_PATH:-data/news/sp500.csv} \
		--kaggle-dataset-key $${KAGGLE_KEY:-sp500_headlines_2008_2024} \
		--score-finbert \
		--from-date $${FROM:-2008-01-01} \
		--to-date $${TO:-2024-12-31} \
		--heartbeat-seconds $${HEARTBEAT:-10} \
		--max-concurrency $${CONCURRENCY:-1} \
		--request-timeout $${TIMEOUT:-30} \
		--retry-max $${RETRY_MAX:-4} \
		--symbols $${SYM:-AAPL}

news-backfill-kaggle-dual:
	@echo 'Usage: make news-backfill-kaggle-dual KAGGLE_SP500_PATH=data/news/sp500.csv KAGGLE_YOGESH_PATH=data/news/yogesh.csv [SYM=AAPL]'
	@export PYTHONPATH=src && python src/data_pipeline/news/ingest.py \
		--provider kaggle \
		--kaggle-dataset-key sp500_headlines_2008_2024 \
		--kaggle-dataset-path $${KAGGLE_SP500_PATH} \
		--kaggle-dataset-key yogeshchary_financial_news \
		--kaggle-dataset-path $${KAGGLE_YOGESH_PATH} \
		--from-date $${FROM:-2008-01-01} \
		--to-date $${TO:-$$(date +%F)} \
		--symbols $${SYM:-AAPL}

news-backfill-kaggle-dual-finbert:
	@echo 'Usage: make news-backfill-kaggle-dual-finbert KAGGLE_SP500_PATH=data/news/sp500.csv KAGGLE_YOGESH_PATH=data/news/yogesh.csv'
	@export PYTHONPATH=src && python src/data_pipeline/news/ingest.py \
		--provider kaggle \
		--kaggle-dataset-key sp500_headlines_2008_2024 \
		--kaggle-dataset-path $${KAGGLE_SP500_PATH:-data/news/sp500.csv} \
		--kaggle-dataset-key yogeshchary_financial_news \
		--kaggle-dataset-path $${KAGGLE_YOGESH_PATH:-data/news/yogesh.csv} \
		--score-finbert \
		--from-date $${FROM:-2008-01-01} \
		--to-date $${TO:-$$(date +%F)} \
		--heartbeat-seconds $${HEARTBEAT:-10} \
		--max-concurrency $${CONCURRENCY:-1} \
		--request-timeout $${TIMEOUT:-30} \
		--retry-max $${RETRY_MAX:-4}

sentiment-rollup:
	export PYTHONPATH=src && python src/ml/sentiment/rollup_daily.py

embed-news-qdrant:
	@echo 'Usage: make embed-news-qdrant SYM=AAPL' && export PYTHONPATH=src && python src/ml/sentiment/embed_sync.py --symbol $${SYM:-AAPL}

serve-api:
	export PYTHONPATH=src && uvicorn api.main:app --host 0.0.0.0 --port 8000

streamlit:
	export PYTHONPATH=src && streamlit run src/ui/streamlit_app.py --server.port 8501

lint:
	ruff check src

lint-fix:
	ruff check src --fix

fmt:
	ruff format src

test:
	export PYTHONPATH=src && pytest tests

activate-vm:
	source .venv/bin/activate