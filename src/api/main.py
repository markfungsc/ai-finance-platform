"""FastAPI inference server: load artifacts once, predict per symbol."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, Thread

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from constants import EXPERIMENT_STRATEGY_SLUG, THRESHOLD
from data_pipeline.features.build_features import run_feature_pipeline
from data_pipeline.ingestion.run_ingestion import run_ingestion
from data_pipeline.processing.clean_prices import clean_prices
from database.connection import engine
from database.queries import fetch_features, fetch_features_window
from log_config import get_logger
from ml.analysis.explanations import (
    global_feature_importance,
    indicator_context_tags,
    threshold_explanation,
    top_feature_magnitudes,
)
from ml.dataset import get_pooled_dataset_symbols, load_dataset
from ml.inference.api_inference import (
    predict_trade_success_probability,
    scanner_evaluate_symbol,
)
from ml.models.save_loads import load_feature_columns, load_model, load_scaler
from ui.backtest_tab import load_backtest_csv
from universe.resolve import resolve_ingestion_symbols

logger = get_logger(__name__)
Session = sessionmaker(bind=engine)

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

_refresh_lock = Lock()
_refresh_thread: Thread | None = None
_refresh_state: dict[str, object | None] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "elapsed_ms": 0,
    "error": None,
    "latest_common_timestamp": None,
    "last_market_close_utc": None,
    "stale_symbols": None,
}
_scan_lock = Lock()
_scan_thread: Thread | None = None
_scan_state: dict[str, object | None] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "elapsed_ms": 0,
    "error": None,
    "result": None,
}

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

# Subset of columns for Predict tab chart JSON (must exist in fetch_features rows).
_PREDICT_CHART_SERIALIZE_COLUMNS: tuple[str, ...] = tuple(
    c for c in BACKTEST_INDICATOR_COLUMNS if c != "symbol"
)


def _serialize_predict_chart_history(df: pd.DataFrame) -> list[dict]:
    cols = [c for c in _PREDICT_CHART_SERIALIZE_COLUMNS if c in df.columns]
    if not cols:
        return []
    sub = df[cols].copy()
    return json.loads(sub.to_json(orient="records", date_format="iso"))


def _maybe_tail_predict_history(df: pd.DataFrame) -> pd.DataFrame:
    """Optional cap via PREDICT_UI_MAX_BARS; otherwise full history (max available in DB)."""
    raw = os.environ.get("PREDICT_UI_MAX_BARS")
    if not raw or not raw.strip():
        return df
    try:
        n = int(raw)
    except ValueError:
        return df
    if n <= 0:
        return df
    return df.tail(n).reset_index(drop=True)


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
    chart_history: list[dict] = Field(default_factory=list)
    latest_bar_timestamp: str | None = None
    indicator_context_tags: list[str] = Field(default_factory=list)


class ThresholdGridResponse(BaseModel):
    best_threshold: float
    grid: list[dict]


class ScannerRequest(BaseModel):
    top_n: int = Field(default=5, ge=1, le=50)
    max_symbols: int | None = Field(default=None, ge=1, le=2000)
    max_workers: int | None = Field(default=None, ge=1, le=32)
    min_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class ScannerRow(BaseModel):
    symbol: str
    probability_trade_success: float
    threshold_used: float
    should_trade: bool


class ScannerErrorRow(BaseModel):
    symbol: str
    error: str


class ScannerSkippedRow(BaseModel):
    symbol: str
    reason: str


class ScannerResponse(BaseModel):
    top: list[ScannerRow]
    errors: list[ScannerErrorRow]
    skipped_missing_data: list[ScannerSkippedRow] = Field(default_factory=list)
    skipped_missing_data_count: int = 0
    evaluated_count: int
    error_count: int
    refresh_status: str
    refresh_elapsed_ms: int
    scan_elapsed_ms: int
    duration_ms: int


class ScannerRefreshStatusResponse(BaseModel):
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_ms: int = 0
    error: str | None = None
    latest_common_timestamp: str | None = None
    last_market_close_utc: str | None = None
    stale_symbols: list[str] | None = None


class ScannerScanStatusResponse(BaseModel):
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_ms: int = 0
    error: str | None = None
    result: ScannerResponse | None = None


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


_FRESHNESS_TABLES = frozenset(
    {"clean_stock_prices", "stock_features", "stock_features_zscore"}
)


def _ts_to_utc(ts) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _max_timestamp_per_symbol(table_name: str) -> dict[str, pd.Timestamp]:
    """Latest bar timestamp per symbol for ``table_name`` (UTC)."""
    if table_name not in _FRESHNESS_TABLES:
        raise ValueError(f"unsupported table for freshness: {table_name!r}")
    query = text(
        f"SELECT symbol, MAX(timestamp) AS ts FROM {table_name} GROUP BY symbol"
    )
    out: dict[str, pd.Timestamp] = {}
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    for sym, ts in rows:
        if sym is None or ts is None:
            continue
        out[str(sym).strip().upper()] = _ts_to_utc(ts)
    return out


def _last_market_close_utc(now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now_utc or pd.Timestamp.now(tz="UTC")
    close_hour_raw = os.environ.get("MARKET_CLOSE_UTC_HOUR", "21").strip()
    try:
        close_hour = int(close_hour_raw)
    except ValueError:
        close_hour = 21
    close_hour = max(0, min(23, close_hour))

    close_today = now.normalize() + pd.Timedelta(hours=close_hour)
    if now.weekday() < 5 and now >= close_today:
        candidate = close_today
    else:
        candidate = close_today - pd.Timedelta(days=1)

    while candidate.weekday() >= 5:
        candidate -= pd.Timedelta(days=1)
    return candidate


def _is_market_data_fresh_for_symbols(
    symbols: list[str],
) -> tuple[bool, pd.Timestamp | None, pd.Timestamp, list[str]]:
    """True iff every expected symbol has clean + features + z rows through ``last_close``.

    Uses per-symbol ``min(ts_clean, ts_feat, ts_feat_z) >= last_close`` so one lagging
    ticker cannot be hidden by a global MAX(timestamp).
    """
    last_close = _last_market_close_utc()
    if not symbols:
        return False, None, last_close, []

    norm = [s.strip().upper() for s in symbols if s and str(s).strip()]
    if not norm:
        return False, None, last_close, []

    d_clean = _max_timestamp_per_symbol("clean_stock_prices")
    d_feat = _max_timestamp_per_symbol("stock_features")
    d_z = _max_timestamp_per_symbol("stock_features_zscore")

    stale: list[str] = []
    per_symbol_floor: list[pd.Timestamp] = []
    for sym in norm:
        t1 = d_clean.get(sym)
        t2 = d_feat.get(sym)
        t3 = d_z.get(sym)
        if t1 is None or t2 is None or t3 is None:
            stale.append(sym)
            continue
        floor = min(t1, t2, t3)
        if floor < last_close:
            stale.append(sym)
            continue
        per_symbol_floor.append(floor)

    fresh = len(stale) == 0 and len(per_symbol_floor) == len(norm)
    bottleneck = (
        min(per_symbol_floor) if len(per_symbol_floor) == len(norm) and norm else None
    )
    if stale:
        sample = stale[:15]
        logger.info(
            "market data stale for %d/%d symbols (sample): %s",
            len(stale),
            len(norm),
            sample,
        )
    return fresh, bottleneck, last_close, stale


def _refresh_snapshot() -> dict[str, object | None]:
    with _refresh_lock:
        return dict(_refresh_state)


def _refresh_update(**kwargs: object) -> None:
    with _refresh_lock:
        _refresh_state.update(kwargs)


def _scan_snapshot() -> dict[str, object | None]:
    with _scan_lock:
        return dict(_scan_state)


def _scan_update(**kwargs: object) -> None:
    with _scan_lock:
        _scan_state.update(kwargs)


def _run_refresh_pipeline_for_symbols(symbols: list[str]) -> None:
    run_ingestion(symbols)
    with Session() as session:
        for symbol in symbols:
            clean_prices(session, symbol)
    for symbol in symbols:
        run_feature_pipeline(symbol, backfill=False)


def _refresh_worker(symbols: list[str]) -> None:
    global _refresh_thread
    t0 = time.perf_counter()
    try:
        _run_refresh_pipeline_for_symbols(symbols)
        fresh_after, latest_after, close_after, stale_after = _is_market_data_fresh_for_symbols(
            symbols
        )
        if not fresh_after:
            logger.warning(
                "refresh pipeline finished but market data still stale for %d symbol(s) "
                "(sample): %s — refresh marked succeeded; scanner will skip unusable symbols",
                len(stale_after),
                stale_after[:20],
            )
        _refresh_update(
            status="succeeded",
            finished_at=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            error=None,
            latest_common_timestamp=latest_after.isoformat()
            if latest_after is not None
            else None,
            last_market_close_utc=close_after.isoformat(),
            stale_symbols=list(stale_after) if stale_after else None,
        )
    except Exception as e:
        _refresh_update(
            status="failed",
            finished_at=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            error=str(e),
            stale_symbols=None,
        )
        logger.exception("scanner refresh job failed")
    finally:
        with _refresh_lock:
            _refresh_thread = None


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
        raise HTTPException(
            status_code=422, detail=f"failed loading features: {e}"
        ) from e
    if X.empty:
        raise HTTPException(
            status_code=422, detail=f"No feature rows for symbol {symbol!r}"
        )

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

    df_hist = fetch_features(symbol).sort_values("timestamp").reset_index(drop=True)
    df_hist = _maybe_tail_predict_history(df_hist)
    chart_history = _serialize_predict_chart_history(df_hist)

    last_merged = _df_merged.iloc[-1]
    latest_ts = pd.Timestamp(last_merged["timestamp"])
    latest_bar_timestamp = latest_ts.isoformat()

    vol_roll = (
        pd.to_numeric(df_hist["volume"], errors="coerce")
        .rolling(20, min_periods=1)
        .mean()
    )
    v_last = vol_roll.iloc[-1] if len(vol_roll) else np.nan
    tag_row = last_merged.copy()
    tag_row["volume_sma_20"] = float(v_last) if pd.notna(v_last) else np.nan
    ctx_tags = indicator_context_tags(tag_row)

    return PredictSymbolExplainResponse(
        symbol=symbol,
        probability_trade_success=p,
        threshold_used=threshold_used,
        should_trade=should_trade,
        reason=reason,
        top_feature_magnitudes=top_mag_df.to_dict(orient="records"),
        global_feature_importance=global_imp_df.to_dict(orient="records"),
        chart_history=chart_history,
        latest_bar_timestamp=latest_bar_timestamp,
        indicator_context_tags=ctx_tags,
    )


@app.get("/threshold_grid", response_model=ThresholdGridResponse)
def threshold_grid():
    return ThresholdGridResponse(
        best_threshold=float(app.state.best_threshold),
        grid=app.state.threshold_grid or [],
    )


@app.post("/scanner/refresh/start", response_model=ScannerRefreshStatusResponse)
def scanner_refresh_start():
    global _refresh_thread
    snap = _refresh_snapshot()
    if snap.get("status") == "running":
        return ScannerRefreshStatusResponse(**snap)

    symbols = resolve_ingestion_symbols()
    fresh, latest_common, last_close, _stale = _is_market_data_fresh_for_symbols(symbols)
    if fresh:
        _refresh_update(
            status="skipped_up_to_date",
            started_at=pd.Timestamp.now(tz="UTC").isoformat(),
            finished_at=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_ms=0,
            error=None,
            latest_common_timestamp=latest_common.isoformat()
            if latest_common is not None
            else None,
            last_market_close_utc=last_close.isoformat(),
            stale_symbols=None,
        )
        return ScannerRefreshStatusResponse(**_refresh_snapshot())

    _refresh_update(
        status="running",
        started_at=pd.Timestamp.now(tz="UTC").isoformat(),
        finished_at=None,
        elapsed_ms=0,
        error=None,
        latest_common_timestamp=latest_common.isoformat()
        if latest_common is not None
        else None,
        last_market_close_utc=last_close.isoformat(),
        stale_symbols=None,
    )
    _refresh_thread = Thread(target=_refresh_worker, args=(symbols,), daemon=True)
    _refresh_thread.start()
    return ScannerRefreshStatusResponse(**_refresh_snapshot())


@app.get("/scanner/refresh/status", response_model=ScannerRefreshStatusResponse)
def scanner_refresh_status():
    return ScannerRefreshStatusResponse(**_refresh_snapshot())


def _run_scan_core(body: ScannerRequest) -> ScannerResponse:
    started = time.perf_counter()
    refresh = _refresh_snapshot()
    refresh_status = str(refresh.get("status", "idle"))
    if refresh_status not in {"succeeded", "skipped_up_to_date"}:
        detail = "refresh must be completed before scanning"
        if refresh_status == "failed" and refresh.get("error"):
            detail = f"refresh failed: {refresh.get('error')}"
        elif refresh_status == "running":
            detail = "refresh is running; wait until completion"
        raise HTTPException(status_code=503, detail=detail)

    symbols = get_pooled_dataset_symbols()
    if body.max_symbols is not None:
        symbols = symbols[: int(body.max_symbols)]
    if not symbols:
        return ScannerResponse(
            top=[],
            errors=[],
            skipped_missing_data=[],
            skipped_missing_data_count=0,
            evaluated_count=0,
            error_count=0,
            refresh_status=refresh_status,
            refresh_elapsed_ms=int(refresh.get("elapsed_ms", 0)),
            scan_elapsed_ms=0,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    top_n = int(body.top_n)
    min_p = float(body.min_probability) if body.min_probability is not None else None
    env_workers = os.environ.get("SCAN_MAX_WORKERS", "4").strip()
    try:
        default_workers = int(env_workers)
    except ValueError:
        default_workers = 4
    workers = max(1, min(len(symbols), int(body.max_workers or default_workers)))
    threshold_used = float(app.state.best_threshold)

    scan_start = time.perf_counter()
    scored: list[ScannerRow] = []
    errors: list[ScannerErrorRow] = []
    skipped_missing: list[ScannerSkippedRow] = []
    last_close = _last_market_close_utc()

    def _evaluate_symbol(sym: str) -> tuple[float | None, str | None]:
        return scanner_evaluate_symbol(
            sym,
            app.state.model,
            app.state.feature_cols,
            app.state.scaler,
            last_market_close_utc=last_close,
            quiet=True,
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_evaluate_symbol, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                p, skip_reason = fut.result()
                if skip_reason is not None:
                    skipped_missing.append(
                        ScannerSkippedRow(symbol=sym, reason=skip_reason)
                    )
                    logger.info(
                        "scanner: symbol=%s not scored — excluded from ranking (%s)",
                        sym,
                        skip_reason,
                    )
                    continue
                assert p is not None
                if min_p is not None and p < min_p:
                    continue
                scored.append(
                    ScannerRow(
                        symbol=sym,
                        probability_trade_success=p,
                        threshold_used=threshold_used,
                        should_trade=bool(p > threshold_used),
                    )
                )
            except Exception as e:
                errors.append(ScannerErrorRow(symbol=sym, error=str(e)))

    scored.sort(
        key=lambda r: (-float(r.probability_trade_success), str(r.symbol).upper())
    )
    top = scored[:top_n]
    scan_elapsed_ms = int((time.perf_counter() - scan_start) * 1000)
    duration_ms = int((time.perf_counter() - started) * 1000)
    return ScannerResponse(
        top=top,
        errors=errors,
        skipped_missing_data=skipped_missing,
        skipped_missing_data_count=len(skipped_missing),
        evaluated_count=len(symbols),
        error_count=len(errors),
        refresh_status=refresh_status,
        refresh_elapsed_ms=int(refresh.get("elapsed_ms", 0)),
        scan_elapsed_ms=scan_elapsed_ms,
        duration_ms=duration_ms,
    )


def _scan_worker(body: ScannerRequest) -> None:
    global _scan_thread
    t0 = time.perf_counter()
    try:
        result = _run_scan_core(body)
        _scan_update(
            status="succeeded",
            finished_at=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            error=None,
            result=result.model_dump(),
        )
    except Exception as e:
        _scan_update(
            status="failed",
            finished_at=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            error=str(e),
            result=None,
        )
        logger.exception("scanner scan job failed")
    finally:
        with _scan_lock:
            _scan_thread = None


@app.post("/scanner/scan/start", response_model=ScannerScanStatusResponse)
def scanner_scan_start(body: ScannerRequest):
    global _scan_thread
    snap = _scan_snapshot()
    if snap.get("status") == "running":
        return ScannerScanStatusResponse(**snap)

    _scan_update(
        status="running",
        started_at=pd.Timestamp.now(tz="UTC").isoformat(),
        finished_at=None,
        elapsed_ms=0,
        error=None,
        result=None,
    )
    _scan_thread = Thread(target=_scan_worker, args=(body,), daemon=True)
    _scan_thread.start()
    return ScannerScanStatusResponse(**_scan_snapshot())


@app.get("/scanner/scan/status", response_model=ScannerScanStatusResponse)
def scanner_scan_status():
    return ScannerScanStatusResponse(**_scan_snapshot())


@app.post("/scan_symbols", response_model=ScannerResponse)
def scan_symbols(body: ScannerRequest):
    # Backward-compatible synchronous endpoint.
    return _run_scan_core(body)


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
