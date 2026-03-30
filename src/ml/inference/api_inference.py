"""Inference helpers for API: latest-bar trade-success probability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.dataset import load_dataset


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
