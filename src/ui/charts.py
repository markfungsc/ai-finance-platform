"""Shared Plotly figures for Streamlit (backtest and predict tabs)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotly_split_panels(
    df_sym: pd.DataFrame,
    title_sym: str,
    trades: pd.DataFrame | None = None,
    *,
    mark_prediction_timestamp: pd.Timestamp | str | None = None,
    price_subplot_title: str | None = None,
) -> go.Figure:
    g = (
        df_sym.sort_values("timestamp").copy()
        if "timestamp" in df_sym.columns
        else df_sym.copy()
    )
    x = g["timestamp"] if "timestamp" in g.columns else g.index

    top_title = (
        price_subplot_title
        if price_subplot_title is not None
        else f"Price, volume & entries — {title_sym}"
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.46, 0.30, 0.24],
        subplot_titles=(
            top_title,
            "MACD",
            "RSI (14)",
        ),
        specs=[[{"secondary_y": True}], [{}], [{}]],
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
        secondary_y=False,
    )
    if "volume" in g.columns:
        vol = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
        fig.add_trace(
            go.Bar(
                x=x,
                y=vol,
                name="Volume",
                marker=dict(color="rgba(80, 120, 200, 0.35)"),
                showlegend=True,
            ),
            row=1,
            col=1,
            secondary_y=True,
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

    if trades is not None and not trades.empty and "entry_timestamp" in trades.columns:
        t = trades.copy()
        if "entry_timestamp" in t.columns:
            t.loc[:, "entry_timestamp"] = pd.to_datetime(
                t["entry_timestamp"], utc=True, errors="coerce"
            )
        if "exit_timestamp" in t.columns:
            t.loc[:, "exit_timestamp"] = pd.to_datetime(
                t["exit_timestamp"], utc=True, errors="coerce"
            )

        entry_px = (
            pd.to_numeric(t["entry_price"], errors="coerce")
            if "entry_price" in t.columns
            else pd.Series([np.nan] * len(t))
        )
        exit_px = (
            pd.to_numeric(t["exit_price"], errors="coerce")
            if "exit_price" in t.columns
            else pd.Series([np.nan] * len(t))
        )

        trade_ret = (
            pd.to_numeric(t["trade_return"], errors="coerce")
            if "trade_return" in t.columns
            else pd.Series([0.0] * len(t))
        )
        colors = np.where(trade_ret.to_numpy(dtype=float) >= 0, "#2ca02c", "#d62728")

        mask_buy = t["entry_timestamp"].notna().to_numpy() & np.isfinite(
            entry_px.to_numpy(dtype=float)
        )
        if mask_buy.any():
            fig.add_trace(
                go.Scatter(
                    x=t.loc[mask_buy, "entry_timestamp"],
                    y=entry_px.loc[mask_buy],
                    mode="markers",
                    name="Buy",
                    marker=dict(
                        size=10,
                        symbol="triangle-up",
                        color=colors[mask_buy],
                        line=dict(width=1, color="white"),
                    ),
                ),
                row=1,
                col=1,
            )

        mask_sell = t.get(
            "exit_timestamp", pd.Series([None] * len(t))
        ).notna().to_numpy() & np.isfinite(exit_px.to_numpy(dtype=float))
        if mask_sell.any():
            fig.add_trace(
                go.Scatter(
                    x=t.loc[mask_sell, "exit_timestamp"],
                    y=exit_px.loc[mask_sell],
                    mode="markers",
                    name="Sell",
                    marker=dict(
                        size=9,
                        symbol="triangle-down",
                        color=colors[mask_sell],
                        line=dict(width=1, color="white"),
                    ),
                ),
                row=1,
                col=1,
            )

        conn_x: list[Any] = []
        conn_y: list[Any] = []
        for i in range(len(t)):
            ets = (
                t["entry_timestamp"].iloc[i] if "entry_timestamp" in t.columns else None
            )
            xts = t["exit_timestamp"].iloc[i] if "exit_timestamp" in t.columns else None
            ep = entry_px.iloc[i]
            xp = exit_px.iloc[i]
            if (
                pd.notna(ets)
                and pd.notna(xts)
                and np.isfinite(float(ep))
                and np.isfinite(float(xp))
            ):
                conn_x.extend([ets, xts, None])
                conn_y.extend([float(ep), float(xp), None])

        if conn_x:
            fig.add_trace(
                go.Scatter(
                    x=conn_x,
                    y=conn_y,
                    mode="lines",
                    name="Trade",
                    hoverinfo="skip",
                    showlegend=False,
                    line=dict(dash="dot", width=1, color="rgba(0,0,0,0.35)"),
                ),
                row=1,
                col=1,
            )

    elif "signal" in g.columns:
        sig = g["signal"].to_numpy() == 1
        ret = (
            g["strategy_return"].astype(float).to_numpy()
            if "strategy_return" in g.columns
            else np.zeros(len(g))
        )
        sig_idx = np.flatnonzero(sig)
        if len(sig_idx):
            trade_colors = np.where(ret[sig_idx] >= 0, "#2ca02c", "#d62728")
        else:
            trade_colors = np.array([])
        tx = (
            g["timestamp"].to_numpy() if "timestamp" in g.columns else np.arange(len(g))
        )
        fig.add_trace(
            go.Scatter(
                x=tx[sig],
                y=g["close"].to_numpy()[sig],
                mode="markers",
                name="Entry",
                marker=dict(
                    size=10,
                    symbol="triangle-up",
                    color=trade_colors,
                    line=dict(width=1, color="white"),
                ),
            ),
            row=1,
            col=1,
        )
        if "close" in g.columns and len(sig_idx):
            exit_idx = sig_idx + 1
            exit_idx = exit_idx[exit_idx < len(g)]
            if len(exit_idx):
                exit_colors = trade_colors[: len(exit_idx)]
                fig.add_trace(
                    go.Scatter(
                        x=tx[exit_idx],
                        y=g["close"].to_numpy()[exit_idx],
                        mode="markers",
                        name="Exit",
                        marker=dict(
                            size=9,
                            symbol="triangle-down",
                            color=exit_colors,
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

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="Volume",
        row=1,
        col=1,
        secondary_y=True,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(title_text="MACD", row=2, col=1, automargin=True)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100], automargin=True)

    fig.update_layout(
        height=1020,
        margin=dict(t=64, b=48),
        showlegend=True,
        legend=dict(orientation="h", y=1.18),
        hovermode="x unified",
        uirevision=f"price_ta_{title_sym}",
        dragmode="pan",
    )

    xaxis_extras: dict[str, Any] = {
        "rangeslider": dict(visible=True, thickness=0.06),
    }
    if "timestamp" in g.columns:
        ts_check = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
        if ts_check.notna().any():
            xaxis_extras["rangeselector"] = dict(
                buttons=[
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            )
    fig.update_xaxes(
        row=1,
        col=1,
        **xaxis_extras,
    )

    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikedash="dot",
        spikecolor="rgba(60,60,60,0.45)",
        spikesnap="cursor",
    )

    if mark_prediction_timestamp is not None:
        mts = pd.Timestamp(mark_prediction_timestamp)
        if pd.notna(mts):
            for row_i in (1, 2, 3):
                fig.add_vline(
                    x=mts,
                    line_width=2,
                    line_dash="dash",
                    line_color="rgba(46, 160, 67, 0.9)",
                    row=row_i,
                    col=1,
                )

    return fig
