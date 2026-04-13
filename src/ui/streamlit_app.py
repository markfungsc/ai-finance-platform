"""
Streamlit dashboard: prediction, threshold grid, and local backtest CSV artifacts.

Run API first:  make serve-api
Then:           make streamlit
Or:             streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

from ui.backtest_tab import default_artifacts_root
from ui.backtest_tab import render as render_backtest
from ui.predict_panel import (
    dataframe_from_chart_history,
    render_predict_price_ta_chart,
    render_signal_omission_explainability,
)


def _api_base() -> str:
    return os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _get(path: str, timeout: int = 60) -> dict[str, Any]:
    r = requests.get(f"{_api_base()}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    r = requests.post(
        f"{_api_base()}{path}",
        json=body,
        timeout=timeout,
    )
    if r.status_code >= 400:
        try:
            err = r.json()
            detail = err.get("detail", err)
        except Exception:
            detail = r.text
        raise requests.HTTPError(f"{r.status_code}: {detail}", response=r)
    return r.json()


def main() -> None:
    st.set_page_config(
        page_title="AI Finance (Swing Trade Strategy) — inference", layout="wide"
    )
    st.title("AI Finance (Swing Trade Strategy) — inference dashboard")

    with st.sidebar:
        st.subheader("API")
        default_base = _api_base()
        base = st.text_input("Base URL", value=default_base, key="api_base_input")
        if base:
            os.environ["API_BASE_URL"] = base.rstrip("/")

        if st.button("Check health"):
            try:
                h = _get("/health", timeout=5)
                st.success(h)
            except Exception as e:
                st.error(str(e))

        st.subheader("Backtest artifacts")
        st.caption(
            "Per-split `backtest.csv` from `make experiments`. "
            "Override with env `BACKTEST_ARTIFACTS_ROOT`."
        )
        art_default = str(default_artifacts_root())
        art_input = st.text_input("Artifacts folder", value=art_default, key="art_root")

    tab_predict, tab_grid, tab_backtest = st.tabs(
        ["Predict", "Threshold grid", "Backtest"]
    )

    with tab_predict:
        col1, col2 = st.columns([1, 2])
        with col1:
            symbol = st.text_input("Symbol", value="AAPL").strip().upper()
        with col2:
            run = st.button("Predict", type="primary")

        if run and symbol:
            try:
                data = _post("/predict_symbol_explain", {"symbol": symbol})
                p = float(data["probability_trade_success"])
                t = float(data["threshold_used"])
                trade = bool(data["should_trade"])
                m1, m2, m3 = st.columns(3)
                m1.metric("P(trade success)", f"{p:.4f}")
                m2.metric("Threshold used", f"{t:.4f}")
                m3.metric("Should trade", "Yes" if trade else "No")
                if trade:
                    st.success("Probability is above the deployed threshold.")
                else:
                    st.warning("Probability is at or below the deployed threshold.")

                st.subheader("Price & TA (max historical data from API)")
                df_px = dataframe_from_chart_history(data.get("chart_history") or [])
                render_predict_price_ta_chart(
                    df_px,
                    symbol,
                    data.get("latest_bar_timestamp"),
                )

                render_signal_omission_explainability(
                    symbol=symbol,
                    latest_bar_timestamp=data.get("latest_bar_timestamp"),
                    probability=p,
                    threshold=t,
                    should_trade=trade,
                    reason=str(data.get("reason", "")),
                    indicator_tags=list(data.get("indicator_context_tags") or []),
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Top Feature Magnitudes (latest row)**")
                    tfm = pd.DataFrame(data.get("top_feature_magnitudes") or [])
                    st.dataframe(tfm, width="stretch", height=260)
                with c2:
                    st.markdown("**Global Feature Importance (model)**")
                    gfi = pd.DataFrame(data.get("global_feature_importance") or [])
                    st.dataframe(gfi, width="stretch", height=260)
            except requests.HTTPError as e:
                st.error(str(e))
            except Exception as e:
                st.exception(e)

    with tab_grid:
        if st.button("Load threshold grid"):
            st.session_state["load_grid"] = True

        if st.session_state.get("load_grid"):
            try:
                data = _get("/threshold_grid")
                best = float(data["best_threshold"])
                grid = data.get("grid") or []
                st.metric("Best threshold (API)", f"{best:.4f}")
                if not grid:
                    st.info(
                        "Grid is empty. Run an experiment that writes "
                        "`*_threshold_grid.json` next to the model, or set "
                        "`THRESHOLD_GRID_PATH` on the API."
                    )
                else:
                    df = pd.DataFrame(grid)
                    if "threshold" in df.columns:
                        df = df.sort_values("threshold")
                    st.subheader("avg_cum_return")
                    if "avg_cum_return" in df.columns:
                        st.line_chart(
                            df.set_index("threshold")[["avg_cum_return"]],
                            height=220,
                        )
                    st.subheader("avg_profit_factor")
                    if "avg_profit_factor" in df.columns:
                        st.line_chart(
                            df.set_index("threshold")[["avg_profit_factor"]],
                            height=220,
                        )
                    st.subheader("Win rate % and max drawdown %")
                    plot_df = pd.DataFrame(
                        index=df["threshold"] if "threshold" in df.columns else None
                    )
                    if "avg_win_rate" in df.columns:
                        plot_df["win_rate_pct"] = (
                            df["avg_win_rate"].to_numpy(dtype=float) * 100.0
                        )
                    if "avg_max_drawdown" in df.columns:
                        plot_df["max_drawdown_pct"] = (
                            df["avg_max_drawdown"].to_numpy(dtype=float) * 100.0
                        )
                    plot_df = plot_df.dropna(axis=1, how="all")
                    if not plot_df.empty:
                        st.line_chart(plot_df, height=220)
                    with st.expander("Raw grid rows"):
                        st.dataframe(df, width="stretch")
            except Exception as e:
                st.exception(e)

    with tab_backtest:
        render_backtest(art_input)


if __name__ == "__main__":
    main()
