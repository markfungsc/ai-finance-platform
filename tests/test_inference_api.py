"""Tests for ml.inference.api_inference and api.main FastAPI app."""

import importlib
import json
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from constants import THRESHOLD


class _ProbaModel:
    def predict_proba(self, X):
        return np.array([[0.4, 0.63]])


class _RegModel:
    def predict(self, X):
        return np.array([1.5])


@patch("ml.inference.api_inference.load_dataset")
def test_predict_trade_success_probability_predict_proba(mock_load):
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]})
    mock_load.return_value = (X, pd.Series([0, 1]), pd.DataFrame())
    from ml.inference.api_inference import predict_trade_success_probability

    p = predict_trade_success_probability(
        "AAPL",
        _ProbaModel(),
        ["a", "b"],
        scaler=None,
        quiet=True,
    )
    assert p == pytest.approx(0.63)


@patch("ml.inference.api_inference.load_dataset")
def test_predict_trade_success_probability_regressor_clip(mock_load):
    X = pd.DataFrame({"a": [0.0], "b": [2.0]})
    mock_load.return_value = (X, pd.Series([0]), pd.DataFrame())
    from ml.inference.api_inference import predict_trade_success_probability

    p = predict_trade_success_probability("X", _RegModel(), ["a", "b"])
    assert p == pytest.approx(1.0)


@patch("ml.inference.api_inference.load_dataset")
def test_predict_empty_raises(mock_load):
    mock_load.return_value = (pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())
    from ml.inference.api_inference import predict_trade_success_probability

    with pytest.raises(ValueError, match="No feature rows"):
        predict_trade_success_probability("AAPL", _ProbaModel(), ["a"])


@patch("ml.inference.api_inference.load_dataset")
def test_predict_missing_columns_raises(mock_load):
    X = pd.DataFrame({"a": [1.0]})
    mock_load.return_value = (X, pd.Series([0]), pd.DataFrame())
    from ml.inference.api_inference import predict_trade_success_probability

    with pytest.raises(ValueError, match="Missing feature"):
        predict_trade_success_probability("AAPL", _ProbaModel(), ["a", "b"])


@patch("ml.inference.api_inference.load_dataset")
def test_predict_with_scaler(mock_load):
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 2.0]})
    mock_load.return_value = (X, pd.Series([0, 1]), pd.DataFrame())
    scaler = StandardScaler()
    scaler.fit(np.array([[0.0, 0.0], [1.0, 2.0]]))

    class _M:
        def predict_proba(self, X):
            assert X.shape == (1, 2)
            return np.array([[0.5, 0.5]])

    from ml.inference.api_inference import predict_trade_success_probability

    predict_trade_success_probability("AAPL", _M(), ["a", "b"], scaler=scaler)


def test_fastapi_health_and_predict(tmp_path, monkeypatch):
    model_path = tmp_path / "m.pkl"
    feat_path = tmp_path / "f.pkl"
    joblib.dump(_ProbaModel(), model_path)
    joblib.dump(["a", "b"], feat_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("FEATURE_COLUMNS_PATH", str(feat_path))
    monkeypatch.delenv("SCALER_PATH", raising=False)

    import api.main as api_main

    importlib.reload(api_main)

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}

        with patch.object(
            api_main, "predict_trade_success_probability", return_value=0.63
        ):
            r = client.post("/predict_symbol", json={"symbol": "AAPL"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["probability_trade_success"] == pytest.approx(0.63)
    assert payload["threshold_used"] == pytest.approx(THRESHOLD)
    assert payload["should_trade"] is True


def test_fastapi_predict_value_error_422(tmp_path, monkeypatch):
    model_path = tmp_path / "m.pkl"
    feat_path = tmp_path / "f.pkl"
    joblib.dump(_ProbaModel(), model_path)
    joblib.dump(["a", "b"], feat_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("FEATURE_COLUMNS_PATH", str(feat_path))

    import api.main as api_main

    importlib.reload(api_main)

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as client:
        with patch.object(
            api_main,
            "predict_trade_success_probability",
            side_effect=ValueError("No feature rows for symbol 'ZZZ'"),
        ):
            r = client.post("/predict_symbol", json={"symbol": "ZZZ"})
    assert r.status_code == 422
    assert "No feature rows" in r.json()["detail"]


def test_startup_fails_when_artifacts_missing(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_model.pkl"
    missing_feat = tmp_path / "missing_features.pkl"
    monkeypatch.setenv("MODEL_PATH", str(missing_model))
    monkeypatch.setenv("FEATURE_COLUMNS_PATH", str(missing_feat))
    monkeypatch.delenv("SCALER_PATH", raising=False)
    import api.main as api_main

    importlib.reload(api_main)
    from fastapi.testclient import TestClient

    with pytest.raises(RuntimeError, match="Artifact not found"):
        with TestClient(api_main.app):
            pass


def test_threshold_grid_endpoint_fallback(tmp_path, monkeypatch):
    model_path = tmp_path / "m.pkl"
    feat_path = tmp_path / "f.pkl"
    joblib.dump(_ProbaModel(), model_path)
    joblib.dump(["a", "b"], feat_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("FEATURE_COLUMNS_PATH", str(feat_path))
    monkeypatch.delenv("SCALER_PATH", raising=False)
    monkeypatch.delenv("THRESHOLD_GRID_PATH", raising=False)

    import api.main as api_main

    importlib.reload(api_main)

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as client:
        r = client.get("/threshold_grid")

    assert r.status_code == 200
    payload = r.json()
    assert payload["best_threshold"] == pytest.approx(THRESHOLD)
    assert payload["grid"] == []


def test_threshold_grid_endpoint_loads(tmp_path, monkeypatch):
    model_path = tmp_path / "m.pkl"
    feat_path = tmp_path / "f.pkl"
    joblib.dump(_ProbaModel(), model_path)
    joblib.dump(["a", "b"], feat_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("FEATURE_COLUMNS_PATH", str(feat_path))
    monkeypatch.delenv("SCALER_PATH", raising=False)
    monkeypatch.delenv("THRESHOLD_GRID_PATH", raising=False)

    best_path = model_path.with_name(f"{model_path.stem}_best_threshold.json")
    best_path.write_text(json.dumps({"best_threshold": 0.42}), encoding="utf-8")

    grid_path = model_path.with_name(f"{model_path.stem}_threshold_grid.json")
    grid_payload = {
        "best_threshold": 0.42,
        "grid": [
            {
                "threshold": 0.3,
                "avg_cum_return": 1.1,
                "avg_profit_factor": 1.5,
                "avg_win_rate": 0.2,
                "avg_max_drawdown": -0.1,
                "total_trades": 10,
                "avg_expectancy": 0.01,
            }
        ],
    }
    grid_path.write_text(json.dumps(grid_payload), encoding="utf-8")

    import api.main as api_main

    importlib.reload(api_main)

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as client:
        r = client.get("/threshold_grid")

    assert r.status_code == 200
    payload = r.json()
    assert payload["best_threshold"] == pytest.approx(0.42)
    assert len(payload["grid"]) == 1
    assert payload["grid"][0]["threshold"] == pytest.approx(0.3)
