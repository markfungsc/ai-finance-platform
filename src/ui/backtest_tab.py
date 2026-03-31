"""Streamlit backtest tab: Plotly OHLC-style charts, raw TA, trade PnL per symbol."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def experiments_host_root() -> Path:
    """
    Host-side root for experiments used by Streamlit when reading CSVs directly.
    """
    return repo_root() / "experiments"


def default_artifacts_root() -> Path:
    raw = os.environ.get("BACKTEST_ARTIFACTS_ROOT")
    if raw:
        return Path(raw).expanduser()
    return experiments_host_root() / "artifacts/swing-trade/pooled_random_forest"


def _api_base_url() -> str:
    return os.environ.get("INFERENCE_API_BASE_URL", "http://localhost:8000")


def _artifacts_root_for_api(artifacts_root: Path) -> Path:
    """
    Map a host artifacts_root to the path visible from the API container.

    When the API runs in Docker, experiments are mounted at /app/experiments
    (see docker-compose). We derive a relative path from the host experiments
    root and join it under the container root so that:

        host:      /home/.../experiments/artifacts/...
        container: /app/experiments/artifacts/...

    For local/non-Docker runs, or when the mapping cannot be determined,
    we fall back to the original artifacts_root.
    """
    raw = os.environ.get("BACKTEST_ARTIFACTS_ROOT_API")
    if raw:
        return Path(raw).expanduser()

    host_exp_root = experiments_host_root()
    try:
        rel = artifacts_root.resolve().relative_to(host_exp_root.resolve())
    except ValueError:
        # Not under the expected experiments root; just pass through.
        return artifacts_root

    container_root = Path(
        os.environ.get("BACKTEST_EXPERIMENTS_ROOT_CONTAINER", "/app/experiments")
    )
    return container_root / rel


def fetch_indicators_api(
    artifacts_root: Path, split_id: int, symbol: str
) -> pd.DataFrame:
    url = f"{_api_base_url().rstrip('/')}/backtest/indicators"
    params: dict[str, str | int] = {
        "artifacts_root": str(_artifacts_root_for_api(artifacts_root)),
        "split_id": split_id,
        "symbol": symbol,
    }
    resp = requests.get(url, params=params, timeout=15)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail: str | None = None
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                d = payload.get("detail")
                # FastAPI may return detail as str or list[dict]
                if isinstance(d, str):
                    detail = d
        except Exception:
            detail = None
        msg = f"{e}"
        if detail:
            msg = f"{msg} - detail: {detail}"
        raise requests.HTTPError(msg, response=resp) from e
    data = resp.json()
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def fetch_trades_api(
    artifacts_root: Path, split_id: int, symbol: str | None = None
) -> pd.DataFrame:
    url = f"{_api_base_url().rstrip('/')}/backtest/trades"
    params: dict[str, str | int] = {
        "artifacts_root": str(_artifacts_root_for_api(artifacts_root)),
        "split_id": split_id,
    }
    if symbol:
        params["symbol"] = symbol
    resp = requests.get(url, params=params, timeout=15)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail: str | None = None
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                d = payload.get("detail")
                if isinstance(d, str):
                    detail = d
        except Exception:
            detail = None
        msg = f"{e}"
        if detail:
            msg = f"{msg} - detail: {detail}"
        raise requests.HTTPError(msg, response=resp) from e
    data = resp.json()
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def dedupe_pooled_timestamp_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .copy()
    )


def discover_splits(artifacts_root: Path) -> list[tuple[int, Path]]:
    if not artifacts_root.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in sorted(artifacts_root.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"split_(\d+)$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def load_backtest_csv(split_dir: Path) -> pd.DataFrame:
    path = split_dir / "backtest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path).copy()
    if "timestamp" in df.columns:
        df.loc[:, "timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        )
    return df


def final_cum_returns(df: pd.DataFrame) -> tuple[float, float]:
    pooled = "symbol" in df.columns and df["symbol"].nunique() > 1
    if pooled and "timestamp" in df.columns:
        d = dedupe_pooled_timestamp_for_plot(df)
        cs = float(d["cum_strategy_return"].iloc[-1]) if len(d) else 1.0
        mcol = (
            "cum_market_return_pooled_eqw"
            if "cum_market_return_pooled_eqw" in d.columns
            else "cum_market_return"
        )
        cm = float(d[mcol].iloc[-1]) if len(d) and mcol in d.columns else 1.0
        return cs, cm
    if len(df) == 0:
        return 1.0, 1.0
    cs = float(df["cum_strategy_return"].iloc[-1])
    cm = float(df["cum_market_return"].iloc[-1])
    return cs, cm


def attach_db_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated: kept for compatibility; indicators now fetched via API in render."""
    return df


def build_trade_pnl_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trade table from signal rows (entry-centric, with simple inferred exits).

    This treats each row with signal==1 as an entry and uses strategy_return on that row
    as the trade return. Exit timestamp/price are approximated as the next bar.
    """
    if "signal" not in df.columns or "strategy_return" not in df.columns:
        return pd.DataFrame()
    base = (
        df.sort_values(["symbol", "timestamp"])
        if {
            "symbol",
            "timestamp",
        }
        <= set(df.columns)
        else df.copy()
    )
    base = base.reset_index(drop=True)

    if base.empty:
        return base

    # Approximate exit at next bar for visualization; strategy_return is already TP/SL.
    exit_ts = base["timestamp"].shift(-1) if "timestamp" in base.columns else None
    exit_px = base["close"].shift(-1) if "close" in base.columns else None

    tr = base[base["signal"] == 1].copy()
    if tr.empty:
        return tr

    sym_col = "symbol" if "symbol" in tr.columns else None
    if sym_col is None:
        tr.loc[:, "_symbol"] = "__single__"
        sym_col = "_symbol"

    rows: list[dict] = []
    for sym, g in tr.groupby(sym_col, sort=False):
        g = g.sort_values("timestamp") if "timestamp" in g.columns else g
        r = g["strategy_return"].astype(float).to_numpy()
        finite = np.isfinite(r)
        if not finite.all():
            g = g.iloc[finite].copy()
            r = r[finite]
        if not len(g):
            continue
        cum = np.cumprod(1.0 + r)
        for i in range(len(g)):
            idx = g.index[i]
            rows.append(
                {
                    "symbol": str(sym) if sym != "__single__" else "—",
                    "entry_timestamp": g["timestamp"].iloc[i]
                    if "timestamp" in g.columns
                    else None,
                    "exit_timestamp": exit_ts.iloc[idx]
                    if exit_ts is not None
                    else None,
                    "entry_price": g["close"].iloc[i] if "close" in g.columns else None,
                    "exit_price": exit_px.iloc[idx] if exit_px is not None else None,
                    "trade_return": float(r[i]),
                    "trade_compounded_equity": float(cum[i]),
                }
            )
    return pd.DataFrame(rows)


def plotly_summary_merged(summ: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summ["split"],
            y=summ["cum_strategy_end"],
            mode="lines+markers",
            name="Strategy end cum",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summ["split"],
            y=summ["cum_market_end"],
            mode="lines+markers",
            name="Market end cum",
        )
    )
    fig.update_layout(
        title="All splits: strategy vs market (end cumulative return)",
        xaxis_title="Split",
        yaxis_title="Cumulative return (×)",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plotly_trade_equity_by_symbol(df: pd.DataFrame) -> go.Figure | None:
    t = build_trade_pnl_table(df)
    if t.empty or "symbol" not in t.columns:
        return None
    fig = go.Figure()
    for sym, g in t.groupby("symbol"):
        g = g.sort_values("timestamp") if "timestamp" in g.columns else g
        fig.add_trace(
            go.Scatter(
                x=g["timestamp"] if "timestamp" in g.columns else g.index,
                y=g["trade_compounded_equity"],
                mode="lines+markers",
                name=str(sym),
            )
        )
    fig.update_layout(
        title="Compounded trade equity (per symbol, sequential entries)",
        xaxis_title="Time",
        yaxis_title="Running ∏(1 + trade return)",
        height=380,
        hovermode="x unified",
    )
    return fig


def plotly_split_panels(df_sym: pd.DataFrame, title_sym: str) -> go.Figure:
    g = (
        df_sym.sort_values("timestamp").copy()
        if "timestamp" in df_sym.columns
        else df_sym.copy()
    )
    x = g["timestamp"] if "timestamp" in g.columns else g.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            f"Price & entries — {title_sym}",
            "MACD",
            "RSI (14)",
        ),
    )

    if "high" in g.columns and "low" in g.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=g["high"],
                name="High",
                line=dict(width=0.8, color="rgba(150,150,150,0.7)"),
                legendgroup="hl",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=g["low"],
                name="Low",
                line=dict(width=0.8, color="rgba(150,150,150,0.7)"),
                legendgroup="hl",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x, y=g["close"], name="Close", line=dict(color="#1f77b4", width=2)
        ),
        row=1,
        col=1,
    )
    for col, nm, dash in (
        ("ema_10", "EMA 10", None),
        ("ema_20", "EMA 20", "dash"),
        ("sma_20", "SMA 20", "dot"),
        ("bb_mavg_20", "BB mavg 20", "dot"),
        ("bb_hband_20", "BB high 20", "dash"),
        ("bb_lband_20", "BB low 20", "dash"),
    ):
        if col in g.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=g[col],
                    name=nm,
                    line=dict(dash=dash),
                ),
                row=1,
                col=1,
            )

    if "signal" in g.columns:
        sig = g["signal"].to_numpy() == 1
        ret = (
            g["strategy_return"].astype(float).to_numpy()
            if "strategy_return" in g.columns
            else np.zeros(len(g))
        )
        colors = np.where(ret >= 0, "#2ca02c", "#d62728")
        tx = (
            g["timestamp"].to_numpy() if "timestamp" in g.columns else np.arange(len(g))
        )
        # Entry markers
        fig.add_trace(
            go.Scatter(
                x=tx[sig],
                y=g["close"].to_numpy()[sig],
                mode="markers",
                name="Entry",
                marker=dict(
                    size=10,
                    symbol="triangle-up",
                    color=colors[sig],
                    line=dict(width=1, color="white"),
                ),
            ),
            row=1,
            col=1,
        )
        # Approximate exits one bar after entry where available.
        if "close" in g.columns:
            exit_idx = np.where(sig)[0] + 1
            exit_idx = exit_idx[exit_idx < len(g)]
            if len(exit_idx):
                fig.add_trace(
                    go.Scatter(
                        x=tx[exit_idx],
                        y=g["close"].to_numpy()[exit_idx],
                        mode="markers",
                        name="Exit",
                        marker=dict(
                            size=9,
                            symbol="triangle-down",
                            color=colors[exit_idx],
                            line=dict(width=1, color="white"),
                        ),
                    ),
                    row=1,
                    col=1,
                )

    if {"macd", "macd_signal"} <= set(g.columns):
        fig.add_trace(
            go.Bar(
                x=x,
                y=g.get("macd_hist", (g["macd"] - g["macd_signal"])),
                name="MACD hist",
                marker_color="lightblue",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=g["macd"], name="MACD", line=dict(color="#1f77b4")),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=g["macd_signal"], name="MACD sig", line=dict(color="#ff7f0e")
            ),
            row=2,
            col=1,
        )

    if "rsi_14" in g.columns:
        fig.add_trace(
            go.Scatter(x=x, y=g["rsi_14"], name="RSI 14", line=dict(color="#9467bd")),
            row=3,
            col=1,
        )

    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=3, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_layout(height=820, showlegend=True, legend=dict(orientation="h", y=1.14))
    return fig


def render(artifacts_root_str: str) -> None:
    artifacts_root = Path(artifacts_root_str.strip()).expanduser()
    splits = discover_splits(artifacts_root)

    if not splits:
        st.warning(
            f"No `split_*` folders with `backtest.csv` under:\n`{artifacts_root}`\n\n"
            "Run `make experiments` or set **Backtest artifacts** to your run folder."
        )
        return

    view_options = ["Summary (all splits)"] + [f"Split {n:03d}" for n, _ in splits]
    choice = st.selectbox("View", view_options, key="backtest_view")

    if choice == "Summary (all splits)":
        rows = []
        for n, d in splits:
            try:
                df = load_backtest_csv(d)
                cs, cm = final_cum_returns(df)
                signals = (
                    int((df["signal"] == 1).sum()) if "signal" in df.columns else 0
                )
                rows.append(
                    {
                        "split": n,
                        "cum_strategy_end": cs,
                        "cum_market_end": cm,
                        "signal_rows": signals,
                        "rows": len(df),
                    }
                )
            except Exception as e:
                rows.append({"split": n, "error": str(e)})

        summ = pd.DataFrame(rows)
        st.dataframe(summ, use_container_width=True)
        plot_df = summ[summ["cum_strategy_end"].notna()].copy()
        if not plot_df.empty and "cum_strategy_end" in plot_df.columns:
            st.plotly_chart(plotly_summary_merged(plot_df), use_container_width=True)
        if (
            "cum_strategy_end" in summ.columns
            and summ["cum_strategy_end"].notna().any()
        ):
            avg_s = float(summ["cum_strategy_end"].mean())
            avg_m = float(summ["cum_market_end"].mean())
            c1, c2 = st.columns(2)
            c1.metric("Mean cum_strategy (splits)", f"{avg_s:.4f}")
            c2.metric("Mean cum_market (splits)", f"{avg_m:.4f}")
        return

    snum = int(choice.replace("Split ", ""))
    split_dir = dict(splits)[snum]
    try:
        df = load_backtest_csv(split_dir)
    except Exception as e:
        st.error(str(e))
        return

    cs, cm = final_cum_returns(df)
    sig_n = int((df["signal"] == 1).sum()) if "signal" in df.columns else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("End cum_strategy", f"{cs:.4f}")
    m2.metric("End cum_market", f"{cm:.4f}")
    m3.metric("Entry signals", sig_n)
    m4.metric("Rows", len(df))

    st.caption(
        "TA is recomputed from OHLC via `ta` (not CSV z-scores). Markers are **entries** "
        "(green/red = realized PnL sign on that bar)."
    )

    sub_eq, sub_px, sub_tr = st.tabs(
        ["Equity (strategy vs market)", "Price & TA", "Trades PnL"]
    )

    with sub_eq:
        pooled = "symbol" in df.columns and df["symbol"].nunique() > 1
        fig = go.Figure()
        if pooled and "timestamp" in df.columns:
            eq = dedupe_pooled_timestamp_for_plot(df)
            mkt = (
                "cum_market_return_pooled_eqw"
                if "cum_market_return_pooled_eqw" in eq.columns
                else "cum_market_return"
            )
            fig.add_trace(
                go.Scatter(
                    x=eq["timestamp"],
                    y=eq["cum_strategy_return"],
                    name="Strategy",
                    mode="lines",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=eq["timestamp"],
                    y=eq[mkt],
                    name="Market",
                    mode="lines",
                )
            )
        else:
            x = df["timestamp"] if "timestamp" in df.columns else df.index
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["cum_strategy_return"],
                    name="Strategy",
                    mode="lines",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["cum_market_return"],
                    name="Market",
                    mode="lines",
                )
            )
        fig.update_layout(
            title="Cumulative return: strategy vs market",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with sub_px:
        symbols: list[str] = []
        if "symbol" in df.columns:
            symbols = sorted(df["symbol"].dropna().astype(str).unique())

        if len(symbols) > 1:
            sym_pick = st.selectbox("Symbol", symbols, key="split_symbol")
        else:
            sym_pick = symbols[0] if symbols else "—"

        try:
            if not symbols:
                st.error("No symbol information found in backtest.csv.")
            else:
                df_plot_ind = fetch_indicators_api(artifacts_root, snum, sym_pick)
                if df_plot_ind.empty:
                    st.error(
                        "No indicator data returned from API for this symbol/split."
                    )
                else:
                    # Ensure timestamp is datetime for plotting
                    if "timestamp" in df_plot_ind.columns:
                        df_plot_ind["timestamp"] = pd.to_datetime(
                            df_plot_ind["timestamp"], utc=True, errors="coerce"
                        )
                    fig_p = plotly_split_panels(df_plot_ind, str(sym_pick))
                    st.plotly_chart(fig_p, use_container_width=True)
        except Exception as e:
            st.exception(e)

    with sub_tr:
        # Fetch per-trade table via API, defaulting to All symbols first.
        try:
            tbl = fetch_trades_api(artifacts_root, snum, None)
        except Exception as e:
            st.exception(e)
            tbl = pd.DataFrame()
        if tbl.empty:
            st.info("No rows with `signal == 1` for trade table.")
        else:
            symbols_tr = sorted(tbl["symbol"].dropna().astype(str).unique())
            sym_filter = st.selectbox(
                "Trades — symbol", ["All"] + symbols_tr, key="trades_symbol"
            )
            if sym_filter != "All":
                tbl_view = tbl[tbl["symbol"].astype(str) == sym_filter]
            else:
                tbl_view = tbl

            st.subheader("Per-trade PnL (entry/exit, compounded equity)")
            st.dataframe(tbl_view, use_container_width=True, height=340)

            fig_te = None
            if sym_filter == "All":
                fig_te = plotly_trade_equity_by_symbol(df)
            else:
                df_sym = df[df.get("symbol", "").astype(str) == sym_filter]
                fig_te = plotly_trade_equity_by_symbol(df_sym)
            if fig_te is not None:
                st.plotly_chart(fig_te, use_container_width=True)

    with st.expander("Preview backtest.csv"):
        st.dataframe(df.head(50), use_container_width=True)
