"""FastAPI inference server: load artifacts once, predict per symbol."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from constants import EXPERIMENT_STRATEGY_SLUG, THRESHOLD
from log_config import get_logger
from ml.inference.api_inference import predict_trade_success_probability
from ml.models.save_loads import load_feature_columns, load_model, load_scaler

logger = get_logger(__name__)

# Defaults match pooled experiments (see run_experiment / backtest saved names).
# Project root: .../src/api/main.py -> parents[2] == repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_REL = (
    Path("models") / EXPERIMENT_STRATEGY_SLUG / "random_forest_pooled.pkl"
)
_DEFAULT_FEATURE_COL_REL = (
    Path("models")
    / EXPERIMENT_STRATEGY_SLUG
    / "random_forest_pooled_feature_columns.pkl"
)


class PredictSymbolRequest(BaseModel):
    symbol: str = Field(..., min_length=1, examples=["AAPL"])


class PredictSymbolResponse(BaseModel):
    probability_trade_success: float
    threshold_used: float
    should_trade: bool


def _resolve_artifact_path(env_var: str, default_relative: Path) -> str:
    """Resolve path from env, or default under repo root; require an existing file."""
    raw = os.environ.get(env_var)
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = _REPO_ROOT / path
    else:
        path = _REPO_ROOT / default_relative
    if not path.is_file():
        raise RuntimeError(
            f"Artifact not found: {path} (set {env_var} to override the default)"
        )
    return str(path.resolve())


def _load_best_threshold(*, model_path: str) -> float:
    """
    Load persisted optimized best threshold if available.
    Falls back to the default constant threshold when the artifact is missing.
    """
    raw = os.environ.get("BEST_THRESHOLD_PATH")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return float(payload["best_threshold"])
        logger.warning("BEST_THRESHOLD_PATH set but file missing: %s", path)
        return float(THRESHOLD)

    # Derive candidate path from the model path for testability:
    # e.g. random_forest_pooled.pkl -> random_forest_pooled_best_threshold.json
    mp = Path(model_path)
    candidate = mp.with_name(f"{mp.stem}_best_threshold.json")
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return float(payload["best_threshold"])

    return float(THRESHOLD)


def _load_artifacts():
    model_path = _resolve_artifact_path("MODEL_PATH", _DEFAULT_MODEL_REL)
    feature_path = _resolve_artifact_path(
        "FEATURE_COLUMNS_PATH", _DEFAULT_FEATURE_COL_REL
    )
    model = load_model(model_path)
    feature_cols = load_feature_columns(feature_path)
    best_threshold = _load_best_threshold(model_path=model_path)
    scaler = None
    scaler_raw = os.environ.get("SCALER_PATH")
    if scaler_raw:
        sp = Path(scaler_raw)
        if not sp.is_absolute():
            sp = _REPO_ROOT / sp
        if not sp.is_file():
            raise RuntimeError(f"SCALER_PATH={scaler_raw!s} is not an existing file")
        scaler = load_scaler(str(sp.resolve()))
    logger.info(
        "Loaded artifacts: model=%s feature_columns=%d scaler=%s best_threshold=%s",
        model_path,
        len(feature_cols),
        "yes" if scaler is not None else "no",
        best_threshold,
    )
    return model, feature_cols, scaler, best_threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, feature_cols, scaler, best_threshold = _load_artifacts()
    app.state.model = model
    app.state.feature_cols = feature_cols
    app.state.scaler = scaler
    app.state.best_threshold = best_threshold
    yield


app = FastAPI(title="AI Finance inference", lifespan=lifespan)


@app.get("/health")
def health():
    logger.debug("health check")
    return {"status": "ok"}


@app.post("/predict_symbol", response_model=PredictSymbolResponse)
def predict_symbol(body: PredictSymbolRequest):
    symbol = body.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="symbol must be non-empty")

    try:
        p = predict_trade_success_probability(
            symbol,
            app.state.model,
            app.state.feature_cols,
            app.state.scaler,
            quiet=True,
        )
    except ValueError as e:
        logger.info("predict_symbol failed validation symbol=%s error=%s", symbol, e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except SQLAlchemyError as e:
        logger.exception("predict_symbol database error symbol=%s", symbol)
        raise HTTPException(status_code=503, detail=f"database error: {e}") from e

    logger.info(
        "predict_symbol ok symbol=%s probability_trade_success=%.6f",
        symbol,
        p,
    )

    threshold_used = float(app.state.best_threshold)
    should_trade = p > threshold_used
    return PredictSymbolResponse(
        probability_trade_success=p,
        threshold_used=threshold_used,
        should_trade=bool(should_trade),
    )
