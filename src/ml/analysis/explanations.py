"""Fast explainability helpers for Streamlit prediction/backtest views."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def threshold_explanation(probability: float, threshold: float) -> str:
    delta = float(probability) - float(threshold)
    if delta > 0:
        return (
            f"Signal emitted: model score {probability:.4f} is above threshold "
            f"{threshold:.4f} (margin +{delta:.4f})."
        )
    return (
        f"Signal omitted: model score {probability:.4f} is below/equal threshold "
        f"{threshold:.4f} (margin {delta:.4f})."
    )


def global_feature_importance(
    model: Any, feature_names: list[str], top_n: int = 12
) -> pd.DataFrame:
    vals = getattr(model, "feature_importances_", None)
    if vals is None:
        return pd.DataFrame(columns=["feature", "importance"])
    arr = np.asarray(vals, dtype=float)
    n = min(len(arr), len(feature_names))
    df = pd.DataFrame({"feature": feature_names[:n], "importance": arr[:n]})
    return (
        df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    )


def top_feature_magnitudes(
    row: pd.Series, feature_names: list[str], top_n: int = 8
) -> pd.DataFrame:
    vals: list[dict[str, float | str]] = []
    for c in feature_names:
        if c not in row.index:
            continue
        v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
        if pd.isna(v):
            continue
        vals.append({"feature": c, "value": float(v), "abs_value": float(abs(v))})
    if not vals:
        return pd.DataFrame(columns=["feature", "value", "abs_value"])
    return (
        pd.DataFrame(vals)
        .sort_values("abs_value", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def indicator_context_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    ema10 = pd.to_numeric(pd.Series([row.get("ema_10")]), errors="coerce").iloc[0]
    ema20 = pd.to_numeric(pd.Series([row.get("ema_20")]), errors="coerce").iloc[0]
    if not pd.isna(ema10) and not pd.isna(ema20):
        tags.append("EMA short above long" if ema10 > ema20 else "EMA short below long")

    macd = pd.to_numeric(pd.Series([row.get("macd")]), errors="coerce").iloc[0]
    macd_sig = pd.to_numeric(pd.Series([row.get("macd_signal")]), errors="coerce").iloc[
        0
    ]
    if not pd.isna(macd) and not pd.isna(macd_sig):
        tags.append("MACD bullish" if macd > macd_sig else "MACD bearish")

    rsi = pd.to_numeric(pd.Series([row.get("rsi_14")]), errors="coerce").iloc[0]
    if not pd.isna(rsi):
        if rsi >= 70:
            tags.append("RSI overbought")
        elif rsi <= 30:
            tags.append("RSI oversold")
        else:
            tags.append("RSI neutral")

    vol = pd.to_numeric(pd.Series([row.get("volume")]), errors="coerce").iloc[0]
    vol_sma = pd.to_numeric(
        pd.Series([row.get("volume_sma_20")]), errors="coerce"
    ).iloc[0]
    if not pd.isna(vol) and not pd.isna(vol_sma) and vol_sma != 0:
        tags.append("Volume above 20D avg" if vol > vol_sma else "Volume below 20D avg")

    sym_sent = pd.to_numeric(
        pd.Series([row.get("sym_sentiment_d1")]), errors="coerce"
    ).iloc[0]
    spy_sent = pd.to_numeric(
        pd.Series([row.get("spy_sentiment_d1")]), errors="coerce"
    ).iloc[0]
    if not pd.isna(sym_sent):
        tags.append(
            "Symbol sentiment positive"
            if sym_sent > 0
            else "Symbol sentiment negative/flat"
        )
    if not pd.isna(spy_sent):
        tags.append(
            "Market sentiment positive"
            if spy_sent > 0
            else "Market sentiment negative/flat"
        )
    return tags
