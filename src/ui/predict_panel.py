"""Reusable Streamlit blocks for the Predict tab (charts + explainability)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ui.charts import plotly_split_panels


def dataframe_from_chart_history(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def render_predict_price_ta_chart(
    df_chart: pd.DataFrame,
    symbol: str,
    latest_bar_timestamp: str | None,
) -> None:
    """Price + volume + TA panels (shared with Backtest tab chart helper)."""
    if df_chart.empty or "timestamp" not in df_chart.columns:
        st.info("No chart history in API response.")
        return
    mark_ts: pd.Timestamp | str | None = None
    if latest_bar_timestamp:
        mark_ts = pd.Timestamp(latest_bar_timestamp)
    fig = plotly_split_panels(
        df_chart,
        symbol,
        trades=None,
        mark_prediction_timestamp=mark_ts,
        price_subplot_title=(
            f"Price & TA — {symbol} "
            "(full history from DB; dashed line = bar used for prediction)"
        ),
    )
    st.plotly_chart(fig, width="stretch")


def render_signal_omission_explainability(
    *,
    symbol: str,
    latest_bar_timestamp: str | None,
    probability: float,
    threshold: float,
    should_trade: bool,
    reason: str,
    indicator_tags: list[str],
) -> None:
    """Model threshold reason + same-style indicator tags as the Backtest tab."""
    with st.expander(
        "Signal / omission explainability (model + indicator context)",
        expanded=True,
    ):
        st.caption(
            "Threshold line compares model score to the deployed threshold. "
            "Indicator tags summarize EMA, MACD, RSI, volume vs 20D avg, and sentiment "
            "(when present on the merged feature row)."
        )
        ind_expl = (
            ", ".join(indicator_tags)
            if indicator_tags
            else "No indicator tags returned."
        )
        row = {
            "timestamp": latest_bar_timestamp,
            "symbol": symbol,
            "signal": 1 if should_trade else 0,
            "model_reason": reason,
            "indicator_context": ind_expl,
            "P(trade success)": f"{probability:.6f}",
            "threshold": f"{threshold:.6f}",
        }
        st.dataframe(pd.DataFrame([row]), width="stretch", height=120)
        if indicator_tags:
            st.markdown("**Context tags**")
            for t in indicator_tags:
                st.markdown(f"- {t}")
