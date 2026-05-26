#!/usr/bin/env python3
"""Apply ordered SQL files in infra/migrations/ using DATABASE_URL.

Replaces Docker-only ``make migrate`` for RDS, EC2, and local Postgres with a URL.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import psycopg2

_DO_BLOCK_RE = re.compile(r"DO\s+\$\$.*?\$\$\s*;", re.IGNORECASE | re.DOTALL)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _migrations_dir() -> Path:
    return _repo_root() / "infra" / "migrations"


def _normalize_database_url(url: str) -> str:
    """Convert SQLAlchemy-style URLs to a libpq DSN for psycopg2."""
    for prefix in ("postgresql+psycopg2://", "postgres+psycopg2://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


def _strip_line_comments(sql: str) -> str:
    """Remove ``--`` line comments so semicolons in comments are not split points."""
    lines: list[str] = []
    for line in sql.splitlines():
        if "--" in line:
            line = line[: line.index("--")]
        lines.append(line)
    return "\n".join(lines)


def _split_sql_statements(sql: str) -> list[str]:
    """Split migration file into executable statements (project DDL style)."""
    statements: list[str] = []
    for chunk in _strip_line_comments(sql).split(";"):
        body = chunk.strip()
        if not body:
            continue
        statements.append(body + ";")
    return statements


def _migration_statements(sql: str) -> list[str]:
    """Split SQL into statements; keep each ``DO $$ ... $$;`` block intact."""
    statements: list[str] = []
    pos = 0
    for match in _DO_BLOCK_RE.finditer(sql):
        before = sql[pos : match.start()]
        if before.strip():
            statements.extend(_split_sql_statements(before))
        statements.append(match.group(0).strip())
        pos = match.end()
    tail = sql[pos:]
    if tail.strip():
        statements.extend(_split_sql_statements(tail))
    return statements


def main() -> int:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 1

    migration_dir = _migrations_dir()
    if not migration_dir.is_dir():
        print(f"Missing migrations directory: {migration_dir}", file=sys.stderr)
        return 1

    paths = sorted(migration_dir.glob("*.sql"))
    if not paths:
        print(f"No .sql files in {migration_dir}", file=sys.stderr)
        return 1

    dsn = _normalize_database_url(database_url)
    print(f"Applying {len(paths)} migration file(s) to database ...")

    try:
        conn = psycopg2.connect(dsn)
    except psycopg2.Error as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 1

    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for path in paths:
                sql = path.read_text(encoding="utf-8")
                statements = _migration_statements(sql)
                print(f"Running {path.name} ({len(statements)} statement(s))")
                for statement in statements:
                    cur.execute(statement)
    except psycopg2.Error as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print("Migrations complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
