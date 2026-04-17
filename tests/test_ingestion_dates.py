from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from data_pipeline.ingestion.load_stock_data import fetch_records


def test_fetch_records_skips_when_incremental_start_after_today():
    session = SimpleNamespace()
    now = pd.Timestamp("2026-04-17T12:00:00Z")
    last_ts = pd.Timestamp("2026-04-17T04:00:00Z")
    with patch(
        "data_pipeline.ingestion.load_stock_data.get_latest_timestamp",
        return_value=last_ts,
    ), patch(
        "data_pipeline.ingestion.load_stock_data.fetch_stock_price"
    ) as mock_fetch, patch(
        "data_pipeline.ingestion.load_stock_data.pd.Timestamp.now", return_value=now
    ):
        out = list(fetch_records("AAPL", session, backfill=False))
    assert out == []
    mock_fetch.assert_not_called()


def test_fetch_records_incremental_uses_bounded_end_date():
    session = SimpleNamespace()
    now = pd.Timestamp("2026-04-17T12:00:00Z")
    last_ts = pd.Timestamp("2026-04-15T04:00:00Z")
    df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "timestamp": pd.Timestamp("2026-04-16T04:00:00Z"),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            }
        ]
    )
    with patch(
        "data_pipeline.ingestion.load_stock_data.get_latest_timestamp",
        return_value=last_ts,
    ), patch(
        "data_pipeline.ingestion.load_stock_data.fetch_stock_price", return_value=df
    ) as mock_fetch, patch(
        "data_pipeline.ingestion.load_stock_data.pd.Timestamp.now", return_value=now
    ):
        out = list(fetch_records("AAPL", session, backfill=False))
    assert len(out) == 1
    mock_fetch.assert_called_once_with(
        "AAPL",
        start_date="2026-04-16",
        end_date=(now.date() + timedelta(days=1)).isoformat(),
    )
