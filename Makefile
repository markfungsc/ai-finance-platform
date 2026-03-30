.PHONY: lint fmt up down logs migrate ingestion clean features train walk-forward backtest experiments view-results predict serve-api test activate-vm

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

ingestion:
	export PYTHONPATH=src && python src/data_pipeline/ingestion/run_ingestion.py

clean:
	export PYTHONPATH=src && python src/data_pipeline/processing/clean_prices.py
	
features:
	export PYTHONPATH=src && python src/data_pipeline/features/build_features.py

train:
	export PYTHONPATH=src && python src/scripts/run_train.py

walk-forward:
	export PYTHONPATH=src && python src/ml/backtest/test_walk_forward.py

backtest:
	export PYTHONPATH=src && python src/ml/backtest/run_backtest.py

experiments:
	export PYTHONPATH=src && python src/ml/experiments/run_experiment.py

view-results:
	export PYTHONPATH=src && python src/ml/experiments/view_results.py

predict:
	export PYTHONPATH=src && python src/ml/inference/predict.py

serve-api:
	export PYTHONPATH=src && uvicorn api.main:app --host 0.0.0.0 --port 8000

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