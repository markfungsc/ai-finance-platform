"""Inference helpers for API: latest-bar trade-success probability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from log_config import get_logger
from ml.dataset import load_dataset

logger = get_logger(__name__)


def _ts_to_utc(ts) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _probability_from_model(model, X: np.ndarray | pd.DataFrame) -> float:
    """Match backtest logic: predict_proba positive class, else clip regression to [0, 1]."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return float(probs[0, 1])
    preds = model.predict(X)
    return float(np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)[0])


def predict_trade_success_probability(
    symbol: str,
    model,
    feature_cols: list[str],
    scaler=None,
    *,
    quiet: bool = True,
) -> float:
    """
    Load features for ``symbol``, take the latest row, align to ``feature_cols``,
    optionally apply ``scaler``, and return P(trade success).

    Raises:
        ValueError: if there are no rows or required feature columns are missing.
    """
    X, _y, _df_merged = load_dataset(symbol, debug_merge=False, quiet=quiet)
    if X.empty:
        raise ValueError(f"No feature rows for symbol {symbol!r}")

    row = X.iloc[[-1]].copy()
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns for {symbol!r}: {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )

    X_aligned = row[feature_cols]
    if scaler is not None:
        X_model = scaler.transform(X_aligned.to_numpy())
    else:
        X_model = X_aligned

    return _probability_from_model(model, X_model)


def scanner_evaluate_symbol(
    symbol: str,
    model,
    feature_cols: list[str],
    scaler,
    *,
    last_market_close_utc: pd.Timestamp,
    quiet: bool = True,
) -> tuple[float | None, str | None]:
    """
    Score one symbol for the stock scanner, or skip it when recent model inputs are unusable.

    Returns:
        ``(probability, None)`` when the symbol is ranked.
        ``(None, reason)`` when the symbol is intentionally skipped (missing/stale data — not an error).

    Skips when: no feature rows, missing required columns, NaN in any model input on the latest
    row, or latest feature **calendar date** (UTC) is before the calendar date of
    ``last_market_close_utc`` (missing recent bars). Old historical gaps do not skip the symbol if
    the latest bar is current.

    Set ``LOG_LEVEL=DEBUG`` for merged timestamp bounds and staleness details (``[scanner_evaluate]``).
    """
    lc_in = _ts_to_utc(last_market_close_utc)
    logger.debug(
        "[scanner_evaluate] start symbol=%s last_market_close_utc=%s",
        symbol,
        lc_in.isoformat(),
    )

    X, _y, df_merged = load_dataset(symbol, debug_merge=False, quiet=quiet)
    if X.empty:
        logger.info(
            "[scanner_evaluate] skip symbol=%s reason=no_rows X_empty=True",
            symbol,
        )
        return None, (
            "no feature rows — not scored (missing data). "
            "Symbol excluded from scanner ranking."
        )

    ts_col = pd.to_datetime(df_merged["timestamp"], utc=True)
    ts_min = ts_col.min()
    ts_max = ts_col.max()
    ts_iloc_last = _ts_to_utc(df_merged.iloc[-1]["timestamp"])
    iloc_matches_max = ts_max == ts_iloc_last

    logger.debug(
        "[scanner_evaluate] symbol=%s merged_rows=%d X_rows=%d ts_min=%s ts_max=%s "
        "iloc_last_ts=%s iloc_last_matches_ts_max=%s",
        symbol,
        len(df_merged),
        len(X),
        ts_min.isoformat(),
        ts_max.isoformat(),
        ts_iloc_last.isoformat(),
        iloc_matches_max,
    )
    if not iloc_matches_max:
        logger.warning(
            "[scanner_evaluate] symbol=%s df_merged not ordered by time: "
            "iloc[-1] ts=%s != ts_max=%s (staleness check uses iloc[-1])",
            symbol,
            ts_iloc_last.isoformat(),
            pd.Timestamp(ts_max).isoformat(),
        )

    row = X.iloc[[-1]].copy()
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        logger.info(
            "[scanner_evaluate] skip symbol=%s reason=missing_columns count=%d sample=%s",
            symbol,
            len(missing),
            missing[:20],
        )
        return None, (
            f"missing model columns {missing[:15]!r}"
            + (" ..." if len(missing) > 15 else "")
            + " — not scored. Symbol excluded from scanner ranking."
        )

    X_aligned = row[feature_cols]
    nan_cols = [c for c in feature_cols if bool(pd.isna(X_aligned[c].iloc[0]))]
    if nan_cols:
        logger.info(
            "[scanner_evaluate] skip symbol=%s reason=nan_inputs cols=%s",
            symbol,
            nan_cols[:20],
        )
        return None, (
            f"NaN in model inputs for columns {nan_cols[:12]!r}"
            + (" ..." if len(nan_cols) > 12 else "")
            + " — not scored. Symbol excluded from scanner ranking."
        )

    latest_ts = ts_iloc_last
    lc = lc_in
    latest_day = latest_ts.normalize().date()
    close_day = lc.normalize().date()
    ts_max_day = _ts_to_utc(ts_max).normalize().date()
    logger.debug(
        "[scanner_evaluate] staleness symbol=%s latest_day=%s close_day=%s "
        "latest_ts=%s ts_max_date=%s",
        symbol,
        latest_day,
        close_day,
        latest_ts.isoformat(),
        ts_max_day,
    )
    if latest_day < close_day:
        logger.info(
            "[scanner_evaluate] skip symbol=%s reason=stale latest_day=%s close_day=%s "
            "latest_ts=%s merged_ts_max=%s ts_max_day=%s",
            symbol,
            latest_day,
            close_day,
            latest_ts.isoformat(),
            pd.Timestamp(ts_max).isoformat(),
            ts_max_day,
        )
        return None, (
            f"latest feature bar date {latest_day} is before last market session date {close_day} "
            "(missing recent data). Symbol excluded from scanner ranking."
        )

    if scaler is not None:
        X_model = scaler.transform(X_aligned.to_numpy())
    else:
        X_model = X_aligned

    p = _probability_from_model(model, X_model)
    logger.debug(
        "[scanner_evaluate] ok symbol=%s p=%.6f score_row_ts=%s",
        symbol,
        p,
        latest_ts.isoformat(),
    )
    return float(p), None
