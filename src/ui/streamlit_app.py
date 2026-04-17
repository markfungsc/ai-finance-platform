"""
Streamlit dashboard: prediction, threshold grid, and local backtest CSV artifacts.

Run API first:  make serve-api
Then:           make streamlit
Or:             streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import os
import time
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

    tab_predict, tab_scanner, tab_grid, tab_backtest = st.tabs(
        ["Predict", "Scanner", "Threshold grid", "Backtest"]
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

    with tab_scanner:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            top_n = st.number_input("Top N", min_value=1, max_value=50, value=5, step=1)
        with c2:
            max_symbols = st.number_input(
                "Max symbols (0 = all)",
                min_value=0,
                max_value=2000,
                value=0,
                step=1,
            )
        with c3:
            max_workers = st.number_input(
                "Max workers (0 = server default)",
                min_value=0,
                max_value=32,
                value=0,
                step=1,
            )
        min_prob = st.slider("Min probability filter", 0.0, 1.0, 0.0, 0.01)
        run_scan = st.button("Run scanner", type="primary")

        if run_scan:
            body: dict[str, Any] = {"top_n": int(top_n)}
            if int(max_symbols) > 0:
                body["max_symbols"] = int(max_symbols)
            if int(max_workers) > 0:
                body["max_workers"] = int(max_workers)
            if float(min_prob) > 0.0:
                body["min_probability"] = float(min_prob)

            status_ph = st.empty()
            poll_s = float(
                os.environ.get("SCANNER_STATUS_POLL_SECONDS", "5").strip() or "5"
            )
            poll_s = max(1.0, min(30.0, poll_s))
            with st.spinner("Starting refresh job..."):
                try:
                    _post("/scanner/refresh/start", {}, timeout=20)
                except requests.HTTPError as e:
                    st.error(str(e))
                    data = None
                except Exception as e:
                    st.exception(e)
                    data = None
                else:
                    data = None
                    # Poll refresh status, then run scan when ready.
                    refresh_max_polls = int((60 * 60) / poll_s)
                    for _ in range(refresh_max_polls):
                        s = _get("/scanner/refresh/status", timeout=20)
                        rs = str(s.get("status", "idle"))
                        status_ph.info(
                            "Refresh status: "
                            f"{rs} | elapsed_ms={int(s.get('elapsed_ms', 0))}"
                        )
                        if rs in {"succeeded", "skipped_up_to_date"}:
                            stale_syms = s.get("stale_symbols") or []
                            if stale_syms:
                                st.warning(
                                    "Refresh completed but market data is still stale for: "
                                    + ", ".join(str(x) for x in stale_syms)
                                    + ". Those symbols are excluded from scanner ranking "
                                    "when data is missing for prediction."
                                )
                            _post("/scanner/scan/start", body, timeout=20)
                            scan_max_polls = int((120 * 60) / poll_s)
                            for _ in range(scan_max_polls):
                                ss = _get("/scanner/scan/status", timeout=20)
                                scan_status = str(ss.get("status", "idle"))
                                status_ph.info(
                                    "Refresh status: "
                                    f"{rs} | Scan status: {scan_status} "
                                    f"| scan_elapsed_ms={int(ss.get('elapsed_ms', 0))}"
                                )
                                if scan_status == "succeeded":
                                    data = ss.get("result")
                                    break
                                if scan_status == "failed":
                                    st.error(f"Scan failed: {ss.get('error')}")
                                    break
                                time.sleep(poll_s)
                            break
                        if rs == "failed":
                            st.error(f"Refresh failed: {s.get('error')}")
                            break
                        time.sleep(poll_s)

            if data:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Evaluated", int(data.get("evaluated_count", 0)))
                m2.metric("Errors", int(data.get("error_count", 0)))
                m3.metric(
                    "Skipped (missing data)",
                    int(data.get("skipped_missing_data_count", 0)),
                )
                m4.metric("Refresh status", str(data.get("refresh_status", "unknown")))
                m5.metric("Elapsed (ms)", int(data.get("duration_ms", 0)))

                top_df = pd.DataFrame(data.get("top") or [])
                if top_df.empty:
                    st.info("No scanner candidates found.")
                else:
                    st.subheader("Top scanner candidates")
                    st.dataframe(top_df, width="stretch", height=260)
                    st.download_button(
                        "Download top CSV",
                        data=top_df.to_csv(index=False),
                        file_name="scanner_top.csv",
                        mime="text/csv",
                    )

                skip_df = pd.DataFrame(data.get("skipped_missing_data") or [])
                if not skip_df.empty:
                    st.info(
                        "The following symbols were not scored and are excluded from the "
                        "scanner ranking due to missing or stale market/feature data."
                    )
                    st.dataframe(skip_df, width="stretch", height=200)

                err_df = pd.DataFrame(data.get("errors") or [])
                if not err_df.empty:
                    with st.expander("Per-symbol errors"):
                        st.dataframe(err_df, width="stretch", height=200)

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
