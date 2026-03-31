"""Unit tests for ml.dataset.load_dataset."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.dataset import load_dataset, load_train_dataset
from ml.features import FEATURE_COLUMNS_MARKET_CONTEXT_Z, FEATURE_COLUMNS_Z


def _make_features_df(n: int = 10, symbol: str = "AAPL") -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    r5 = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "timestamp": ts,
            "return_5d": r5,
        }
    )


def _make_z_df(
    timestamps: pd.DatetimeIndex, z_marker: float, symbol: str = "AAPL"
) -> pd.DataFrame:
    """Z rows keyed by timestamp; return_1d_z = z_marker + position in *sorted* ts for checks."""
    ts_sorted = timestamps.sort_values()
    rows = []
    for i, t in enumerate(ts_sorted):
        row = {"symbol": symbol, "timestamp": t, "close_z": 0.0}
        for c in FEATURE_COLUMNS_Z:
            if c == "return_1d_z":
                row[c] = float(z_marker + i)
            else:
                row[c] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _with_market_context(df_z: pd.DataFrame) -> pd.DataFrame:
    out = df_z.copy()
    for col in FEATURE_COLUMNS_MARKET_CONTEXT_Z:
        out[col] = 0.0
    return out


class TestLoadDataset:
    @patch("ml.helpers.merge_features.generate_trade_labels")
    @patch("ml.dataset.attach_market_context")
    @patch("ml.dataset.fetch_features_z")
    @patch("ml.dataset.fetch_features")
    def test_x_y_same_length_and_merge_order(
        self,
        mock_fetch: object,
        mock_fetch_z: object,
        mock_attach_context: object,
        mock_labels: object,
    ) -> None:
        n = 10
        df = _make_features_df(n)
        mock_fetch.return_value = df
        # Deliberately reverse z row order vs chronological df
        mock_fetch_z.return_value = _make_z_df(df["timestamp"], z_marker=100.0).iloc[
            ::-1
        ]
        mock_attach_context.side_effect = _with_market_context
        mock_labels.side_effect = lambda d: d.assign(
            trade_success=d["return_5d"].shift(-5)
        ).dropna(subset=["trade_success"])

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

    @patch("ml.helpers.merge_features.generate_trade_labels")
    @patch("ml.dataset.attach_market_context")
    @patch("ml.dataset.fetch_features_z")
    @patch("ml.dataset.fetch_features")
    def test_inner_merge_drops_rows_without_z(
        self,
        mock_fetch: object,
        mock_fetch_z: object,
        mock_attach_context: object,
        mock_labels: object,
    ) -> None:
        df = _make_features_df(10)
        mock_fetch.return_value = df
        z_full = _make_z_df(df["timestamp"], z_marker=0.0)
        # After target dropna, rows use timestamps iloc[0:5]; omit z for the first (t0)
        kept = df["timestamp"].iloc[1:5]
        mock_fetch_z.return_value = z_full[z_full["timestamp"].isin(kept)].reset_index(
            drop=True
        )
        mock_attach_context.side_effect = _with_market_context
        mock_labels.side_effect = lambda d: d.assign(trade_success=d["return_5d"])

        X, y, merged = load_dataset("AAPL")

        assert len(X) == len(y) == len(merged) == 4
        assert merged["timestamp"].iloc[0] == df["timestamp"].iloc[1]

    @patch("ml.helpers.merge_features.generate_trade_labels")
    @patch("ml.dataset.attach_market_context")
    @patch("ml.dataset.fetch_features_z")
    @patch("ml.dataset.fetch_features")
    def test_symbol_argument_passed_to_fetchers(
        self,
        mock_fetch: object,
        mock_fetch_z: object,
        mock_attach_context: object,
        mock_labels: object,
    ) -> None:
        df = _make_features_df(10)
        mock_fetch.return_value = df
        mock_fetch_z.return_value = _make_z_df(df["timestamp"], z_marker=0.0)
        mock_attach_context.side_effect = _with_market_context
        mock_labels.side_effect = lambda d: d.assign(trade_success=d["return_5d"])

        load_dataset("MSFT")

        mock_fetch.assert_called_once_with("MSFT")
        mock_fetch_z.assert_called_once_with("MSFT")

    @patch("ml.helpers.merge_features.generate_trade_labels")
    @patch("ml.dataset.attach_market_context")
    @patch("ml.dataset.fetch_features_z")
    @patch("ml.dataset.fetch_features")
    @patch("ml.dataset.TRAIN_SYMBOLS", ["AAPL", "MSFT"])
    def test_load_train_dataset_sorts_pooled_input_before_context_attach(
        self,
        mock_fetch: object,
        mock_fetch_z: object,
        mock_attach_context: object,
        mock_labels: object,
    ) -> None:
        n = 12
        df_aapl = _make_features_df(n=n, symbol="AAPL")
        df_msft = _make_features_df(n=n, symbol="MSFT")
        df_msft.loc[:, "timestamp"] = df_msft["timestamp"] + pd.Timedelta("1D")

        z_aapl = _make_z_df(df_aapl["timestamp"], z_marker=10.0, symbol="AAPL").iloc[::-1]
        z_msft = _make_z_df(df_msft["timestamp"], z_marker=20.0, symbol="MSFT").iloc[::-1]

        feature_by_symbol = {"AAPL": df_aapl, "MSFT": df_msft}
        z_by_symbol = {"AAPL": z_aapl, "MSFT": z_msft}
        mock_fetch.side_effect = lambda symbol: feature_by_symbol[symbol]
        mock_fetch_z.side_effect = lambda symbol: z_by_symbol[symbol]

        def _assert_sorted_and_attach(df_z_in: pd.DataFrame) -> pd.DataFrame:
            sorted_copy = (
                df_z_in.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
            )
            pd.testing.assert_frame_equal(
                df_z_in.reset_index(drop=True), sorted_copy, check_like=False
            )
            return _with_market_context(df_z_in)

        mock_attach_context.side_effect = _assert_sorted_and_attach
        mock_labels.side_effect = lambda d: d.assign(trade_success=d["return_5d"])

        X, y, merged = load_train_dataset()

        assert len(X) == len(y) == len(merged)
        assert len(X) > 0
        mock_attach_context.assert_called_once()
