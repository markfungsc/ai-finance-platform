.PHONY: lint fmt

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

predict:
	export PYTHONPATH=src && python src/ml/inference/predict.py

lint:
	ruff check src

fmt:
	ruff format src

test:
	export PYTHONPATH=src && pytest tests

activate-vm:
	source .venv/bin/activate