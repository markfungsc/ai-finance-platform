"""FastAPI inference server: load artifacts once, predict per symbol."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from constants import EXPERIMENT_STRATEGY_SLUG, THRESHOLD
from database.queries import fetch_features_window
from log_config import get_logger
from ml.analysis.explanations import (
    global_feature_importance,
    threshold_explanation,
    top_feature_magnitudes,
)
from ml.dataset import load_dataset
from ml.inference.api_inference import predict_trade_success_probability
from ml.models.save_loads import load_feature_columns, load_model, load_scaler
from ui.backtest_tab import load_backtest_csv

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

BACKTEST_INDICATOR_COLUMNS: list[str] = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ema_10",
    "ema_20",
    "sma_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_mavg_20",
    "bb_hband_20",
    "bb_lband_20",
    "bb_width_20",
    "bb_pband_20",
]


class PredictSymbolRequest(BaseModel):
    symbol: str = Field(..., min_length=1, examples=["AAPL"])


class PredictSymbolResponse(BaseModel):
    probability_trade_success: float
    threshold_used: float
    should_trade: bool


class PredictSymbolExplainResponse(BaseModel):
    symbol: str
    probability_trade_success: float
    threshold_used: float
    should_trade: bool
    reason: str
    top_feature_magnitudes: list[dict]
    global_feature_importance: list[dict]


class ThresholdGridResponse(BaseModel):
    best_threshold: float
    grid: list[dict]


class BacktestIndicatorsRow(BaseModel):
    symbol: str
    timestamp: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    ema_10: float | None = None
    ema_20: float | None = None
    sma_20: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    bb_mavg_20: float | None = None
    bb_hband_20: float | None = None
    bb_lband_20: float | None = None
    bb_width_20: float | None = None
    bb_pband_20: float | None = None


class BacktestTradeRow(BaseModel):
    symbol: str
    entry_timestamp: str | None = None
    exit_timestamp: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    trade_return: float
    trade_compounded_equity: float


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


def _load_threshold_grid(*, model_path: str) -> list[dict]:
    """
    Load persisted optimized threshold grid if available.

    Expected on disk:
      - MODEL_STEM_threshold_grid.json
      - or a file set via THRESHOLD_GRID_PATH
    """
    raw = os.environ.get("THRESHOLD_GRID_PATH")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload.get("grid") or []
            if isinstance(payload, list):
                return payload
        logger.warning("THRESHOLD_GRID_PATH set but file missing: %s", str(path))
        return []

    mp = Path(model_path)
    candidate = mp.with_name(f"{mp.stem}_threshold_grid.json")
    if not candidate.is_file():
        return []

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload.get("grid") or []
    if isinstance(payload, list):
        return payload
    return []


def _load_artifacts():
    model_path = _resolve_artifact_path("MODEL_PATH", _DEFAULT_MODEL_REL)
    feature_path = _resolve_artifact_path(
        "FEATURE_COLUMNS_PATH", _DEFAULT_FEATURE_COL_REL
    )
    model = load_model(model_path)
    feature_cols = load_feature_columns(feature_path)
    best_threshold = _load_best_threshold(model_path=model_path)
    threshold_grid = _load_threshold_grid(model_path=model_path)
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
        "Loaded artifacts: model=%s feature_columns=%d scaler=%s best_threshold=%s threshold_grid_rows=%d",
        model_path,
        len(feature_cols),
        "yes" if scaler is not None else "no",
        best_threshold,
        len(threshold_grid),
    )
    return model, feature_cols, scaler, best_threshold, threshold_grid


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, feature_cols, scaler, best_threshold, threshold_grid = _load_artifacts()
    app.state.model = model
    app.state.feature_cols = feature_cols
    app.state.scaler = scaler
    app.state.best_threshold = best_threshold
    app.state.threshold_grid = threshold_grid
    yield


app = FastAPI(title="AI Finance inference", lifespan=lifespan)

cors_raw = os.environ.get("CORS_ALLOW_ORIGINS")
if cors_raw:
    origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    if not origins:
        origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


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


@app.post("/predict_symbol_explain", response_model=PredictSymbolExplainResponse)
def predict_symbol_explain(body: PredictSymbolRequest):
    symbol = body.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="symbol must be non-empty")
    try:
        X, _y, _df_merged = load_dataset(symbol, debug_merge=False, quiet=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"failed loading features: {e}") from e
    if X.empty:
        raise HTTPException(status_code=422, detail=f"No feature rows for symbol {symbol!r}")

    feature_cols = app.state.feature_cols
    row = X.iloc[[-1]].copy()
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing feature columns for {symbol!r}: {missing[:20]}",
        )
    x_aligned = row[feature_cols]
    if app.state.scaler is not None:
        x_model = app.state.scaler.transform(x_aligned.to_numpy())
    else:
        x_model = x_aligned
    if hasattr(app.state.model, "predict_proba"):
        p = float(app.state.model.predict_proba(x_model)[0, 1])
    else:
        p = float(np.clip(app.state.model.predict(x_model)[0], 0.0, 1.0))
    threshold_used = float(app.state.best_threshold)
    should_trade = bool(p > threshold_used)
    reason = threshold_explanation(p, threshold_used)

    top_mag_df = top_feature_magnitudes(row.iloc[0], feature_cols, top_n=10)
    global_imp_df = global_feature_importance(app.state.model, feature_cols, top_n=10)
    return PredictSymbolExplainResponse(
        symbol=symbol,
        probability_trade_success=p,
        threshold_used=threshold_used,
        should_trade=should_trade,
        reason=reason,
        top_feature_magnitudes=top_mag_df.to_dict(orient="records"),
        global_feature_importance=global_imp_df.to_dict(orient="records"),
    )


@app.get("/threshold_grid", response_model=ThresholdGridResponse)
def threshold_grid():
    return ThresholdGridResponse(
        best_threshold=float(app.state.best_threshold),
        grid=app.state.threshold_grid or [],
    )


@app.get("/backtest/indicators", response_model=list[BacktestIndicatorsRow])
def backtest_indicators(
    artifacts_root: str = Query(..., description="Backtest artifacts root folder"),
    split_id: int = Query(..., ge=0, description="Split id, e.g. 0 for split_000"),
    symbol: str = Query(..., min_length=1),
):
    """
    Return OHLCV + indicator rows for a given split and symbol.
    """
    sym = symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=422, detail="symbol must be non-empty")

    root = Path(artifacts_root).expanduser()
    split_dir = root / f"split_{split_id:03d}"
    try:
        df = load_backtest_csv(split_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if "timestamp" not in df.columns:
        raise HTTPException(
            status_code=422, detail="backtest.csv missing timestamp column"
        )

    # Normalize timestamp dtype from backtest.csv to timezone-aware datetime
    # (load_backtest_csv already attempts this, but we enforce it here to be
    # robust to any upstream changes or alternate loaders).
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    if "symbol" not in df.columns:
        df.loc[:, "symbol"] = sym

    df_sym = df[df["symbol"].astype(str).str.upper() == sym].copy()
    if df_sym.empty:
        return []

    ts_min = df_sym["timestamp"].min()
    ts_max = df_sym["timestamp"].max()
    feats = fetch_features_window([sym], ts_min, ts_max)
    if feats.empty:
        # Fall back to whatever is in backtest.csv
        merged = df_sym
    else:
        feats = feats.copy()
        # Normalize DB timestamp dtype to match df_sym (timezone-aware datetime).
        feats["timestamp"] = pd.to_datetime(
            feats["timestamp"], utc=True, errors="coerce"
        )
        merged = df_sym.merge(
            feats,
            on=["symbol", "timestamp"],
            how="left",
            suffixes=("", "_db"),
        )
        # Prefer DB indicator columns when present.
        for col in BACKTEST_INDICATOR_COLUMNS:
            if col in ("symbol", "timestamp"):
                continue
            db_col = f"{col}_db"
            if db_col in merged.columns:
                merged[col] = merged[db_col].where(
                    merged[db_col].notna(), merged.get(col)
                )
                merged = merged.drop(columns=[db_col])

    merged = merged.sort_values("timestamp")
    out_cols = BACKTEST_INDICATOR_COLUMNS
    rows: list[BacktestIndicatorsRow] = []
    for _, r in merged.iterrows():
        payload = {k: r[k] if k in merged.columns else None for k in out_cols}
        payload["timestamp"] = (
            str(payload["timestamp"]) if payload["timestamp"] is not None else None
        )
        rows.append(BacktestIndicatorsRow(**payload))
    return rows


@app.get("/backtest/trades", response_model=list[BacktestTradeRow])
def backtest_trades(
    artifacts_root: str = Query(..., description="Backtest artifacts root folder"),
    split_id: int = Query(..., ge=0, description="Split id, e.g. 0 for split_000"),
    symbol: str | None = Query(
        None, description="Optional symbol filter; when omitted returns all symbols"
    ),
):
    """
    Return per-trade rows (entry/exit) for a given split, optionally filtered by symbol.
    """
    from ui.backtest_tab import build_trade_pnl_table  # local import to avoid cycles

    root = Path(artifacts_root).expanduser()
    split_dir = root / f"split_{split_id:03d}"
    try:
        df = load_backtest_csv(split_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if symbol:
        sym = symbol.strip().upper()
        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper() == sym]

    tbl = build_trade_pnl_table(df)
    if tbl.empty:
        return []

    # Defensive guard: ensure numeric fields are finite before serialization.
    for col in ("trade_return", "trade_compounded_equity"):
        if col in tbl.columns:
            mask = np.isfinite(tbl[col].astype(float).to_numpy())
            if not mask.all():
                logger.warning(
                    "backtest_trades dropping %d rows with non-finite %s "
                    "for artifacts_root=%s split_id=%d symbol=%s",
                    int((~mask).sum()),
                    col,
                    artifacts_root,
                    split_id,
                    symbol or "*",
                )
                tbl = tbl[mask]
    if tbl.empty:
        return []

    rows: list[BacktestTradeRow] = []
    for _, r in tbl.iterrows():
        try:
            # Required numeric fields must be finite floats
            tr = float(r["trade_return"])
            te = float(r["trade_compounded_equity"])
            if not (np.isfinite(tr) and np.isfinite(te)):
                continue

            # Optional price fields: coerce to float when finite, else None
            ep_raw = r.get("entry_price")
            xp_raw = r.get("exit_price")
            ep = (
                float(ep_raw)
                if ep_raw is not None and np.isfinite(float(ep_raw))
                else None
            )
            xp = (
                float(xp_raw)
                if xp_raw is not None and np.isfinite(float(xp_raw))
                else None
            )

            payload = {
                "symbol": str(r.get("symbol", "")),
                "entry_timestamp": str(r["entry_timestamp"])
                if r.get("entry_timestamp") is not None
                else None,
                "exit_timestamp": str(r["exit_timestamp"])
                if r.get("exit_timestamp") is not None
                else None,
                "entry_price": ep,
                "exit_price": xp,
                "trade_return": tr,
                "trade_compounded_equity": te,
            }
            rows.append(BacktestTradeRow(**payload))
        except Exception as e:  # pragma: no cover - extremely defensive
            logger.warning(
                "backtest_trades skipping malformed row for artifacts_root=%s split_id=%d symbol=%s error=%s",
                artifacts_root,
                split_id,
                symbol or "*",
                e,
            )
            continue
    return rows
