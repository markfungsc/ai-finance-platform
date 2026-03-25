"""Unit tests for ml.dataset.load_dataset."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.dataset import load_dataset
from ml.features import FEATURE_COLUMNS_Z


def _make_features_df(n: int = 10) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    r5 = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * n,
            "timestamp": ts,
            "return_5d": r5,
        }
    )


def _make_z_df(timestamps: pd.DatetimeIndex, z_marker: float) -> pd.DataFrame:
    """Z rows keyed by timestamp; return_1d_z = z_marker + position in *sorted* ts for checks."""
    ts_sorted = timestamps.sort_values()
    rows = []
    for i, t in enumerate(ts_sorted):
        row = {"symbol": "AAPL", "timestamp": t, "close_z": 0.0}
        for c in FEATURE_COLUMNS_Z:
            if c == "return_1d_z":
                row[c] = float(z_marker + i)
            else:
                row[c] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


class TestLoadDataset:
    @patch("ml.dataset.fetch_features_z")
    @patch("ml.dataset.fetch_features")
    def test_x_y_same_length_and_merge_order(
        self, mock_fetch: object, mock_fetch_z: object
    ) -> None:
        n = 10
        df = _make_features_df(n)
        mock_fetch.return_value = df
        # Deliberately reverse z row order vs chronological df
        mock_fetch_z.return_value = _make_z_df(df["timestamp"], z_marker=100.0).iloc[
            ::-1
        ]

        X, y, merged = load_dataset("AAPL")

        # shift(-5): rows 0..4 have targets from return_5d at 5..9
        assert len(X) == len(y) == 5
        assert len(merged) == 5

        for i in range(5):
            assert y.iloc[i] == pytest.approx(float(i + 5))

        # Row order follows ``df`` (chronological), not shuffled z frame
        ts_sorted = df["timestamp"].sort_values()
        for i in range(5):
            assert merged["timestamp"].iloc[i] == ts_sorted.iloc[i]
            assert X["return_1d_z"].iloc[i] == pytest.approx(100.0 + i)

    @patch("ml.dataset.fetch_features_z")
    @patch("ml.dataset.fetch_features")
    def test_inner_merge_drops_rows_without_z(
        self, mock_fetch: object, mock_fetch_z: object
    ) -> None:
        df = _make_features_df(10)
        mock_fetch.return_value = df
        z_full = _make_z_df(df["timestamp"], z_marker=0.0)
        # After target dropna, rows use timestamps iloc[0:5]; omit z for the first (t0)
        kept = df["timestamp"].iloc[1:5]
        mock_fetch_z.return_value = z_full[z_full["timestamp"].isin(kept)].reset_index(
            drop=True
        )

        X, y, merged = load_dataset("AAPL")

        assert len(X) == len(y) == len(merged) == 4
        assert merged["timestamp"].iloc[0] == df["timestamp"].iloc[1]

    @patch("ml.dataset.fetch_features_z")
    @patch("ml.dataset.fetch_features")
    def test_symbol_argument_passed_to_fetchers(
        self, mock_fetch: object, mock_fetch_z: object
    ) -> None:
        df = _make_features_df(10)
        mock_fetch.return_value = df
        mock_fetch_z.return_value = _make_z_df(df["timestamp"], z_marker=0.0)

        load_dataset("MSFT")

        mock_fetch.assert_called_once_with("MSFT")
        mock_fetch_z.assert_called_once_with("MSFT")
