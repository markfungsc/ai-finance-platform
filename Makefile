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

run:
	export PYTHONPATH=src && python src/data_pipeline/ingestion/run_ingestion.py

clean:
	export PYTHONPATH=src && python src/data_pipeline/processing/clean_prices.py

lint:
	ruff check src

fmt:
	ruff format src