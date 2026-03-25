"""Unit tests for data_pipeline.features.build_features."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data_pipeline.features.build_features import (
    compute_features,
    rowwise_cross_sectional_zscore,
    run_feature_pipeline,
)
from database.queries import STOCK_FEATURES_VALUE_COLUMNS


def _synthetic_ohlc_frame(n_rows: int = 150) -> pd.DataFrame:
    """Enough rows for sma_100 / volatility_100 warm-up."""
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B", tz="UTC")
    close = np.linspace(100.0, 100.0 + n_rows * 0.1, n_rows)
    return pd.DataFrame(
        {
            "symbol": ["TEST"] * n_rows,
            "timestamp": idx,
            "close": close,
        }
    )


class TestComputeFeatures:
    def test_adds_expected_columns(self) -> None:
        df = compute_features(_synthetic_ohlc_frame())
        for col in STOCK_FEATURES_VALUE_COLUMNS:
            assert col in df.columns

    def test_warmup_then_finite(self) -> None:
        df = compute_features(_synthetic_ohlc_frame(150))
        tail = df.iloc[-1]
        assert pd.notna(tail["return_1d"])
        assert pd.notna(tail["sma_100"])
        assert pd.notna(tail["volatility_100"])

    def test_return_1d_matches_pct_change(self) -> None:
        df = compute_features(_synthetic_ohlc_frame(20))
        expected = df["close"].pct_change(1).iloc[-1]
        assert df["return_1d"].iloc[-1] == pytest.approx(expected)


class TestRowwiseCrossSectionalZscore:
    def test_constant_features_zero_z(self) -> None:
        df = pd.DataFrame({"a": [5.0, 5.0], "b": [5.0, 5.0], "c": [5.0, 5.0]})
        out = rowwise_cross_sectional_zscore(df, ["a", "b", "c"])
        assert np.allclose(out["a_z"], 0.0)
        assert np.allclose(out["b_z"], 0.0)
        assert np.allclose(out["c_z"], 0.0)

    def test_single_row_manual_stats(self) -> None:
        df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
        out = rowwise_cross_sectional_zscore(df, ["x", "y", "z"])
        mean = 2.0
        std = np.sqrt(((1 - mean) ** 2 + (2 - mean) ** 2 + (3 - mean) ** 2) / 3)
        assert out["x_z"].iloc[0] == pytest.approx((1.0 - mean) / std)
        assert out["y_z"].iloc[0] == pytest.approx((2.0 - mean) / std)
        assert out["z_z"].iloc[0] == pytest.approx((3.0 - mean) / std)

    def test_rows_independent(self) -> None:
        df = pd.DataFrame({"u": [0.0, 10.0], "v": [0.0, 10.0]})
        out = rowwise_cross_sectional_zscore(df, ["u", "v"])
        assert out["u_z"].iloc[0] == pytest.approx(0.0)
        assert out["v_z"].iloc[0] == pytest.approx(0.0)
        assert out["u_z"].iloc[1] == pytest.approx(0.0)
        assert out["v_z"].iloc[1] == pytest.approx(0.0)


class TestRunFeaturePipeline:
    @patch("data_pipeline.features.build_features.upsert_features_z")
    @patch("data_pipeline.features.build_features.upsert_features")
    @patch(
        "data_pipeline.features.build_features.delete_incomplete_stock_feature_rows",
        return_value=0,
    )
    @patch("data_pipeline.features.build_features.fetch_clean_data")
    def test_backfill_calls_upserts_with_z_columns(
        self,
        mock_fetch: object,
        _mock_delete: object,
        mock_upsert: object,
        mock_upsert_z: object,
    ) -> None:
        mock_fetch.return_value = _synthetic_ohlc_frame(150)

        run_feature_pipeline("TEST", backfill=True)

        mock_fetch.assert_called_once_with("TEST", None)
        mock_upsert.assert_called_once()
        mock_upsert_z.assert_called_once()

        base_batch = mock_upsert.call_args[0][0]
        z_batch = mock_upsert_z.call_args[0][0]
        assert len(base_batch) > 0
        assert len(z_batch) == len(base_batch)
        assert "return_1d_z" in z_batch[0]
        assert "close_z" in z_batch[0]
