import sys

from sqlalchemy.exc import OperationalError

from ml.evaluate import evaluate_model
from ml.train import train_model


def main() -> None:
    try:
        _, preds, y_test = train_model("AAPL")
    except OperationalError:
        print(
            "Cannot reach PostgreSQL (connection refused or host unreachable).\n"
            "\n"
            "Start the database and apply migrations:\n"
            "  make up\n"
            "  make migrate\n"
            "\n"
            "Ensure feature data exists for training:\n"
            "  make ingestion && make clean && make features\n"
            "\n"
            "Set DATABASE_URL (or use .env), e.g.:\n"
            "  export DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/ai_finance",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    evaluate_model(preds, y_test)


if __name__ == "__main__":
    main()
