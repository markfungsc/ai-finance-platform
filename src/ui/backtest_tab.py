"""Streamlit backtest tab: Plotly OHLC-style charts, raw TA, trade PnL per symbol."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
from plotly.subplots import make_subplots


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_artifacts_root() -> Path:
    raw = os.environ.get("BACKTEST_ARTIFACTS_ROOT")
    if raw:
        return Path(raw).expanduser()
    return repo_root() / "experiments/artifacts/swing-trade/pooled_random_forest"


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


def add_raw_indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").copy() if "timestamp" in g.columns else g.copy()
    close = g["close"].astype(float)

    g.loc[:, "ema_10"] = ta.trend.ema_indicator(close, window=10)
    g.loc[:, "ema_20"] = ta.trend.ema_indicator(close, window=20)
    g.loc[:, "sma_20"] = ta.trend.sma_indicator(close, window=20)
    g.loc[:, "rsi_14"] = ta.momentum.rsi(close, window=14)
    g.loc[:, "macd_line"] = ta.trend.macd(close, window_slow=26, window_fast=12)
    g.loc[:, "macd_signal"] = ta.trend.macd_signal(
        close, window_slow=26, window_fast=12, window_sign=9
    )
    g.loc[:, "macd_hist"] = ta.trend.macd_diff(
        close, window_slow=26, window_fast=12, window_sign=9
    )
    return g


def build_trade_pnl_table(df: pd.DataFrame) -> pd.DataFrame:
    if "signal" not in df.columns or "strategy_return" not in df.columns:
        return pd.DataFrame()
    tr = df.loc[df["signal"] == 1].copy()
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
        cum = np.cumprod(1.0 + r)
        for i in range(len(g)):
            rows.append(
                {
                    "symbol": str(sym) if sym != "__single__" else "—",
                    "timestamp": g["timestamp"].iloc[i]
                    if "timestamp" in g.columns
                    else None,
                    "strategy_return": float(r[i]),
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
    g = add_raw_indicators(df_sym)
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
        fig.add_trace(
            go.Scatter(
                x=tx[sig],
                y=g["close"].to_numpy()[sig],
                mode="markers",
                name="Entry signal",
                marker=dict(
                    size=11,
                    symbol="triangle-up",
                    color=colors[sig],
                    line=dict(width=1, color="white"),
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(x=x, y=g["macd_hist"], name="MACD hist", marker_color="lightblue"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=g["macd_line"], name="MACD", line=dict(color="#1f77b4")),
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
            df_plot = df[df["symbol"].astype(str) == sym_pick].copy()
        else:
            df_plot = df.copy()
            sym_pick = symbols[0] if symbols else "—"

        if "close" not in df_plot.columns:
            st.error("No `close` column — cannot plot TA.")
        else:
            try:
                fig_p = plotly_split_panels(df_plot, str(sym_pick))
                st.plotly_chart(fig_p, use_container_width=True)
            except Exception as e:
                st.exception(e)

    with sub_tr:
        tbl = build_trade_pnl_table(df)
        if tbl.empty:
            st.info("No rows with `signal == 1` for trade table.")
        else:
            st.subheader("Each entry: compounded equity within symbol (chronological)")
            st.dataframe(tbl, use_container_width=True, height=340)
            fig_te = plotly_trade_equity_by_symbol(df)
            if fig_te is not None:
                st.plotly_chart(fig_te, use_container_width=True)

    with st.expander("Preview backtest.csv"):
        st.dataframe(df.head(50), use_container_width=True)
