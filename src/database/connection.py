import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. For Docker Compose, set it in the api service environment."
    )

# connect_timeout avoids hanging forever if the DB is unreachable (e.g. Postgres not ready).
engine = create_engine(
    DATABASE_URL,
    connect_args={"connect_timeout": 15},
    pool_pre_ping=True,
)
