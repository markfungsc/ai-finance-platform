from types import SimpleNamespace

import pandas as pd

from database import queries


class _DummyConnCtx:
    def __enter__(self):
        return SimpleNamespace()

    def __exit__(self, exc_type, exc, tb):
        return False


def test_read_features_chunk_without_end_date(monkeypatch):
    captured = {}

    def _fake_read_sql(query, conn, params):
        captured["sql"] = query.text
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(queries.engine, "connect", lambda: _DummyConnCtx())
    monkeypatch.setattr(queries.pd, "read_sql", _fake_read_sql)

    queries._read_features_chunk(["AAPL", "MSFT"])

    assert "AND p.timestamp <= :end_date" not in captured["sql"]
    assert "end_date" not in captured["params"]


def test_read_features_chunk_with_end_date(monkeypatch):
    captured = {}

    def _fake_read_sql(query, conn, params):
        captured["sql"] = query.text
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(queries.engine, "connect", lambda: _DummyConnCtx())
    monkeypatch.setattr(queries.pd, "read_sql", _fake_read_sql)

    queries._read_features_chunk(["AAPL"], end_date="2026-03-27")

    assert "AND p.timestamp <= :end_date" in captured["sql"]
    assert captured["params"]["end_date"] == "2026-03-27"


def test_read_features_z_chunk_without_end_date(monkeypatch):
    captured = {}

    def _fake_read_sql(query, conn, params):
        captured["sql"] = query.text
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(queries.engine, "connect", lambda: _DummyConnCtx())
    monkeypatch.setattr(queries.pd, "read_sql", _fake_read_sql)

    queries._read_features_z_chunk(["AAPL", "MSFT"])

    assert "AND timestamp <= :end_date" not in captured["sql"]
    assert "end_date" not in captured["params"]


def test_read_features_z_chunk_with_end_date(monkeypatch):
    captured = {}

    def _fake_read_sql(query, conn, params):
        captured["sql"] = query.text
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(queries.engine, "connect", lambda: _DummyConnCtx())
    monkeypatch.setattr(queries.pd, "read_sql", _fake_read_sql)

    queries._read_features_z_chunk(["AAPL"], end_date="2026-03-27")

    assert "AND timestamp <= :end_date" in captured["sql"]
    assert captured["params"]["end_date"] == "2026-03-27"
