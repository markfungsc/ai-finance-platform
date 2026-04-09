"""Streamlit backtest tab: Plotly OHLC-style charts, raw TA, trade PnL per symbol."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from ml.analysis.explanations import indicator_context_tags, threshold_explanation


def _prob_from_backtest_row(row: pd.Series) -> float | None:
    """Artifact CSV uses ``prob_trade_success``; API/UI may use ``probability_trade_success``."""
    for col in ("prob_trade_success", "probability_trade_success"):
        if col not in row.index:
            continue
        x = pd.to_numeric(row[col], errors="coerce")
        if pd.isna(x):
            continue
        return float(x)
    return None


def _row_entry_price(row: pd.Series, *, has_entry_price: bool) -> float | None:
    """Prefer engine ``entry_price``; fall back to bar ``close`` when missing."""
    ep: float | None = None
    if has_entry_price and "entry_price" in row.index:
        v = row["entry_price"]
        ep = float(v) if v is not None and np.isfinite(float(v)) else None
    if ep is None and "close" in row.index:
        v = row["close"]
        ep = float(v) if v is not None and np.isfinite(float(v)) else None
    return ep


def _row_exit_raw(x_row: pd.Series, *, has_exit_price: bool) -> float | None:
    if has_exit_price and "exit_price" in x_row.index:
        v = x_row["exit_price"]
        return float(v) if v is not None and np.isfinite(float(v)) else None
    if "close" in x_row.index:
        v = x_row["close"]
        return float(v) if v is not None and np.isfinite(float(v)) else None
    return None


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


def _enrich_work_with_chart_indicators(
    work: pd.DataFrame,
    artifacts_root: Path,
    split_id: int,
) -> pd.DataFrame:
    """
    Left-join OHLC/TA from ``/backtest/indicators`` (same source as Price & TA tab).

    ``backtest.csv`` rows are mostly model features; ``indicator_context_tags`` needs
    raw ``ema_10``, ``macd``, ``rsi_14``, etc., from the DB via the API.
    """
    if work.empty or "timestamp" not in work.columns or "symbol" not in work.columns:
        return work
    parts: list[pd.DataFrame] = []
    for sym in sorted(work["symbol"].dropna().astype(str).unique()):
        sub = work.loc[work["symbol"].astype(str) == sym].copy()
        try:
            ind = fetch_indicators_api(artifacts_root, split_id, sym)
        except Exception:
            parts.append(sub)
            continue
        if ind.empty:
            parts.append(sub)
            continue
        ind = ind.copy()
        ind["timestamp"] = pd.to_datetime(ind["timestamp"], utc=True, errors="coerce")
        ind["symbol"] = ind["symbol"].astype(str)
        sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
        sub["symbol"] = sub["symbol"].astype(str)
        overlap = (set(sub.columns) & set(ind.columns)) - {"timestamp", "symbol"}
        sub_trim = sub.drop(columns=list(overlap), errors="ignore")
        merged = sub_trim.merge(ind, on=["timestamp", "symbol"], how="left")
        parts.append(merged)
    if not parts:
        return work
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["timestamp", "symbol"], kind="mergesort")


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
    """One row per completed round-trip (entry from flat), not per raw signal bar.

    If ``entry_trade`` and ``exit_trade`` exist (from ``basic_backtest``), each entry is
    paired to its exact exit bar and uses engine-provided ``entry_price``/``exit_price``.

    If exit columns are missing (legacy artifacts), exit timestamp/price are approximated
    as the next bar for display.
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

    sym_col = "symbol" if "symbol" in base.columns else None
    if sym_col is None:
        base.loc[:, "_symbol"] = "__single__"
        sym_col = "_symbol"

    has_exit_price = "exit_price" in base.columns
    has_entry_price = "entry_price" in base.columns

    rows: list[dict] = []
    for sym, g in base.groupby(sym_col, sort=False):
        g = g.sort_values("timestamp") if "timestamp" in g.columns else g

        if "exit_trade" in g.columns and (g["exit_trade"] == 1).any():
            g_ord = g.reset_index(drop=True)
            pending: deque[int] = deque()
            pairs: list[tuple[int, int]] = []
            for i in range(len(g_ord)):
                row = g_ord.iloc[i]
                et = int(row["exit_trade"]) if "exit_trade" in row.index else 0
                ent = int(row["entry_trade"]) if "entry_trade" in row.index else 0
                # Match engine: close open position before opening a new one on the same bar.
                if et == 1 and pending:
                    epos = pending.popleft()
                    pairs.append((epos, i))
                if ent == 1:
                    pending.append(i)

            if not pairs:
                continue

            pair_returns: list[tuple[int, int, float]] = []
            for epos, xpos in pairs:
                e_row = g_ord.iloc[epos]
                r_i = float(e_row["strategy_return"])
                if not np.isfinite(r_i):
                    continue
                pair_returns.append((epos, xpos, r_i))

            if not pair_returns:
                continue

            r_list = np.array([t[2] for t in pair_returns], dtype=float)
            cum = np.cumprod(1.0 + r_list)

            for j, (epos, xpos, r_i) in enumerate(pair_returns):
                e_row = g_ord.iloc[epos]
                x_row = g_ord.iloc[xpos]
                entry_ts = e_row["timestamp"] if "timestamp" in e_row.index else None
                exit_ts = x_row["timestamp"] if "timestamp" in x_row.index else None

                ep = _row_entry_price(e_row, has_entry_price=has_entry_price)
                xp_raw = _row_exit_raw(x_row, has_exit_price=has_exit_price)

                if ep is not None and np.isfinite(ep) and np.isfinite(r_i):
                    xp = float(ep * (1.0 + r_i))
                else:
                    xp = xp_raw

                rows.append(
                    {
                        "symbol": str(sym) if sym != "__single__" else "—",
                        "entry_timestamp": entry_ts,
                        "exit_timestamp": exit_ts,
                        "entry_price": ep,
                        "exit_price": xp,
                        "trade_return": r_i,
                        "trade_compounded_equity": float(cum[j]),
                    }
                )
        else:
            if "entry_trade" in g.columns:
                entries = g[g["entry_trade"] == 1].copy()
            else:
                entries = g[g["signal"] == 1].copy()
            if entries.empty:
                continue

            # Filter out non-finite returns; prices are allowed to be NaN.
            r_series = entries["strategy_return"].astype(float)
            finite_mask = np.isfinite(r_series.to_numpy())
            if not finite_mask.all():
                entries = entries.iloc[finite_mask].copy()
            if entries.empty:
                continue

            r = entries["strategy_return"].astype(float).to_numpy()
            cum = np.cumprod(1.0 + r)

            # Legacy visualization: approximate exit on the next bar.
            exit_ts = g["timestamp"].shift(-1) if "timestamp" in g.columns else None
            exit_px = g["close"].shift(-1) if "close" in g.columns else None

            for i in range(len(entries)):
                idx_entry = entries.index[i]
                entry_ts = (
                    entries["timestamp"].iloc[i]
                    if "timestamp" in entries.columns
                    else None
                )
                exit_ts_val = exit_ts.loc[idx_entry] if exit_ts is not None else None

                e_row = entries.iloc[i]
                ep = _row_entry_price(e_row, has_entry_price=has_entry_price)

                r_i = float(r[i])
                xp_raw: float | None
                if exit_px is not None:
                    v = exit_px.loc[idx_entry]
                    xp_raw = (
                        float(v) if v is not None and np.isfinite(float(v)) else None
                    )
                else:
                    xp_raw = None

                if ep is not None and np.isfinite(ep) and np.isfinite(r_i):
                    xp = float(ep * (1.0 + r_i))
                else:
                    xp = xp_raw

                rows.append(
                    {
                        "symbol": str(sym) if sym != "__single__" else "—",
                        "entry_timestamp": entry_ts,
                        "exit_timestamp": exit_ts_val,
                        "entry_price": ep,
                        "exit_price": xp,
                        "trade_return": r_i,
                        "trade_compounded_equity": float(cum[i]),
                    }
                )

    return pd.DataFrame(rows)


def plotly_equity_per_symbol_vs_market(
    df: pd.DataFrame, symbol: str
) -> go.Figure | None:
    """
    One symbol's strategy equity vs that symbol's buy-and-hold.

    In pooled CSVs, ``cum_strategy_return`` / ``cum_market_return`` are portfolio
    benchmarks repeated on every row; per-symbol curves are rebuilt from
    ``strategy_return`` and ``close`` when multiple symbols are present.
    """
    if "timestamp" not in df.columns or "symbol" not in df.columns:
        return None
    d = df.loc[df["symbol"].astype(str) == str(symbol)].copy()
    if d.empty:
        return None
    d = d.sort_values("timestamp")
    x = d["timestamp"]

    pooled_multi = int(df["symbol"].nunique()) > 1

    if pooled_multi and "strategy_return" in d.columns:
        sr = pd.to_numeric(d["strategy_return"], errors="coerce").fillna(0.0)
        y_strat = (1.0 + sr).cumprod()
    elif "cum_strategy_return" in d.columns:
        y_strat = pd.to_numeric(d["cum_strategy_return"], errors="coerce")
    else:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_strat,
            name="Strategy (symbol)",
            mode="lines",
        )
    )

    if pooled_multi and "close" in d.columns:
        cl = pd.to_numeric(d["close"], errors="coerce")
        if cl.notna().any():
            c0 = float(cl.dropna().iloc[0])
            if c0 > 1e-12:
                y_mkt = cl / c0
            else:
                y_mkt = pd.Series(1.0, index=d.index)
            y_mkt = y_mkt.ffill().bfill().fillna(1.0)
        else:
            y_mkt = pd.Series(1.0, index=d.index)
        mkt_name = "Market (symbol buy-hold)"
    else:
        mcol = (
            "cum_market_return"
            if "cum_market_return" in d.columns
            else "cum_market_return_pooled_eqw"
        )
        if mcol not in d.columns:
            fig.update_layout(
                title=f"Cumulative return — {symbol}: strategy vs market",
                height=400,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
            )
            return fig
        y_mkt = pd.to_numeric(d[mcol], errors="coerce")
        mkt_name = "Market (symbol)"

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mkt,
            name=mkt_name,
            mode="lines",
        )
    )
    fig.update_layout(
        title=f"Cumulative return — {symbol}: strategy vs market",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def plotly_summary_strategy_market_traces(
    splits,
    strategy_y,
    market_y,
    *,
    title: str,
    strategy_name: str = "Strategy end cum",
    market_name: str = "Market end cum",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=splits,
            y=strategy_y,
            mode="lines+markers",
            name=strategy_name,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=splits,
            y=market_y,
            mode="lines+markers",
            name=market_name,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Split",
        yaxis_title="Cumulative return (×)",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plotly_summary_merged(summ: pd.DataFrame) -> go.Figure:
    return plotly_summary_strategy_market_traces(
        summ["split"],
        summ["cum_strategy_end"],
        summ["cum_market_end"],
        title="All splits: strategy vs market (end cumulative return)",
        strategy_name="Strategy end cum",
        market_name="Market end cum",
    )


def load_threshold_grid_json(artifacts_root: Path) -> dict | None:
    p = artifacts_root / "threshold_grid.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def per_split_series_from_grid_row(
    row: dict,
) -> tuple[list[int], list[float], list[float]] | None:
    ps = row.get("per_split")
    if not ps or not isinstance(ps, list):
        return None
    ordered = sorted(ps, key=lambda r: int(r["split"]))
    return (
        [int(r["split"]) for r in ordered],
        [float(r["cum_return"]) for r in ordered],
        [float(r["cum_market_return"]) for r in ordered],
    )


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


def plotly_split_panels(
    df_sym: pd.DataFrame,
    title_sym: str,
    trades: pd.DataFrame | None = None,
) -> go.Figure:
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
        vertical_spacing=0.12,
        row_heights=[0.46, 0.30, 0.24],
        subplot_titles=(
            f"Price, volume & entries — {title_sym}",
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

    # Accurate trade buy/sell markers (driven by engine exits).
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

        # Buy markers
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

        # Sell markers
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

        # Dotted connectors: entry -> exit per trade.
        conn_x: list[object] = []
        conn_y: list[object] = []
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

    # Legacy fallback: approximate entry/exit markers from `signal` rows.
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
                    color=trade_colors,
                    line=dict(width=1, color="white"),
                ),
            ),
            row=1,
            col=1,
        )
        # Approximate exits one bar after entry where available.
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

    # Trading-style navigation + hover alignment (Price & TA panel only).
    fig.update_layout(
        height=1020,
        margin=dict(t=64, b=48),
        showlegend=True,
        legend=dict(orientation="h", y=1.18),
        hovermode="x unified",
        uirevision=f"price_ta_{title_sym}",
        dragmode="pan",
    )

    # Timestamp navigator under the price chart (top row).
    xaxis_extras: dict = {
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

    # Vertical dotted guide line across all stacked panels on hover.
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikedash="dot",
        spikecolor="rgba(60,60,60,0.45)",
        spikesnap="cursor",
    )

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
        st.dataframe(summ, width="stretch")
        plot_df = summ[summ["cum_strategy_end"].notna()].copy()
        if not plot_df.empty and "cum_strategy_end" in plot_df.columns:
            grid_bundle = load_threshold_grid_json(artifacts_root)
            grid_rows: list[dict] = []
            if grid_bundle and isinstance(grid_bundle.get("grid"), list):
                for r in grid_bundle["grid"]:
                    if r.get("per_split") and per_split_series_from_grid_row(r):
                        grid_rows.append(r)
                grid_rows.sort(key=lambda row: float(row.get("threshold", 0.0)))

            strat_labels = ["Saved split backtests (CSVs)"]
            best_thr = (
                float(grid_bundle["best_threshold"])
                if grid_bundle and grid_bundle.get("best_threshold") is not None
                else None
            )
            for r in grid_rows:
                t = float(r["threshold"])
                tag = f"Grid replay τ={t:.3g}"
                if best_thr is not None and abs(t - best_thr) < 1e-9:
                    tag += " (objective pick)"
                strat_labels.append(tag)

            if len(strat_labels) > 1:
                strategy_source = st.selectbox(
                    "Strategy curve source",
                    list(range(len(strat_labels))),
                    format_func=lambda i: strat_labels[i],
                    key="summary_strategy_threshold_mode",
                )
            else:
                strategy_source = 0

            if strategy_source == 0:
                fig_sum = plotly_summary_merged(plot_df)
            else:
                row = grid_rows[strategy_source - 1]
                pcs = per_split_series_from_grid_row(row)
                xs, ys, ym = pcs  # type: ignore[misc]
                thr_f = float(row["threshold"])
                fig_sum = plotly_summary_strategy_market_traces(
                    xs,
                    ys,
                    ym,
                    title=(
                        "All splits: strategy vs market "
                        f"(threshold grid τ={thr_f:.3g}, end cum per split)"
                    ),
                    strategy_name=f"Strategy (τ={thr_f:.3g})",
                    market_name="Market (per split)",
                )
            st.plotly_chart(fig_sum, width="stretch")
            if grid_rows and strategy_source != 0:
                st.caption(
                    "Strategy/market points come from `threshold_grid.json` (replay). "
                    "Saved split CSVs still reflect the run's chosen backtest threshold."
                )
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
        "TA is recomputed from OHLC via `ta` (not CSV z-scores). Markers show buy/sell "
        "for each completed trade (green/red = realized PnL sign)."
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
        st.plotly_chart(fig, width="stretch")

        if pooled and "symbol" in df.columns:
            symbols_eq = sorted(df["symbol"].dropna().astype(str).unique())
            if len(symbols_eq) > 1:
                sym_eq = st.selectbox(
                    "Per-symbol equity (vs symbol market)",
                    symbols_eq,
                    key="equity_per_symbol",
                )
                fig_sym = plotly_equity_per_symbol_vs_market(df, sym_eq)
                if fig_sym is not None:
                    st.plotly_chart(fig_sym, width="stretch")

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
                    trades_df = pd.DataFrame()
                    try:
                        trades_df = fetch_trades_api(artifacts_root, snum, sym_pick)
                    except Exception:
                        trades_df = pd.DataFrame()

                    fig_p = plotly_split_panels(
                        df_plot_ind, str(sym_pick), trades=trades_df
                    )
                    st.plotly_chart(fig_p, width="stretch")
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
            st.dataframe(tbl_view, width="stretch", height=340)

            fig_te = None
            if sym_filter == "All":
                fig_te = plotly_trade_equity_by_symbol(df)
            else:
                df_sym = df[df.get("symbol", "").astype(str) == sym_filter]
                fig_te = plotly_trade_equity_by_symbol(df_sym)
            if fig_te is not None:
                st.plotly_chart(fig_te, width="stretch")

            if sym_filter != "All" and "symbol" in df.columns:
                work_df = df[df["symbol"].astype(str) == sym_filter].copy()
            else:
                work_df = df.copy()

            with st.expander("Signal/Omission explainability (model + context)"):
                if (
                    sym_filter == "All"
                    and "symbol" in df.columns
                    and df["symbol"].dropna().astype(str).nunique() > 1
                ):
                    st.caption(
                        "Full bar history for this split, chronological across **all** "
                        'symbols. Narrow with "Trades — symbol" and the row filter below.'
                    )
                row_kind = st.selectbox(
                    "Rows to explain",
                    ["All bars", "Omissions only", "Signal days only"],
                    key="explain_row_filter",
                )
                work = work_df.sort_values("timestamp").copy()
                if "signal" in work.columns:
                    is_signal = (
                        pd.to_numeric(work["signal"], errors="coerce").fillna(0) == 1
                    )
                else:
                    is_signal = pd.Series(False, index=work.index)
                if row_kind == "Omissions only":
                    work = work.loc[~is_signal]
                elif row_kind == "Signal days only":
                    work = work.loc[is_signal]

                work = _enrich_work_with_chart_indicators(work, artifacts_root, snum)

                grid_bundle = load_threshold_grid_json(artifacts_root)
                thr = (
                    float(grid_bundle["best_threshold"])
                    if grid_bundle and grid_bundle.get("best_threshold") is not None
                    else None
                )

                reason_rows: list[dict] = []
                for _, r in work.iterrows():
                    if "signal" in r.index and pd.notna(r["signal"]):
                        try:
                            sig = 1 if int(float(r["signal"])) == 1 else 0
                        except (TypeError, ValueError):
                            sig = 0
                    else:
                        sig = 0
                    tags = indicator_context_tags(r)
                    indicator_explanation = (
                        ", ".join(tags) if tags else "No indicator context available."
                    )
                    prob = _prob_from_backtest_row(r)
                    if prob is not None and thr is not None:
                        reason = threshold_explanation(prob, thr)
                    else:
                        reason = (
                            "No model score in artifact; see indicator explanation."
                        )
                    reason_rows.append(
                        {
                            "timestamp": r.get("timestamp"),
                            "symbol": r.get("symbol"),
                            "signal": sig,
                            "reason": reason,
                            "indicator_explanation": indicator_explanation,
                        }
                    )
                if reason_rows:
                    er = pd.DataFrame(reason_rows)
                    _explain_cols = [
                        "timestamp",
                        "symbol",
                        "signal",
                        "reason",
                        "indicator_explanation",
                    ]
                    er = er[[c for c in _explain_cols if c in er.columns]]
                    n_sig = (
                        int((er["signal"] == 1).sum()) if "signal" in er.columns else 0
                    )
                    n_omit = len(er) - n_sig
                    st.caption(
                        f"Showing {len(er)} rows ({n_sig} signal bars, {n_omit} non-signal)."
                    )
                    if len(er) > 50_000:
                        st.warning(
                            "Very large table; consider filtering by symbol or row kind."
                        )
                    st.dataframe(er, width="stretch", height=400)
                    er_omit = er[er["signal"] != 1] if "signal" in er.columns else er
                    if not er_omit.empty and "reason" in er_omit.columns:
                        top = (
                            er_omit["reason"]
                            .value_counts()
                            .rename_axis("reason")
                            .reset_index(name="count")
                        )
                        st.markdown("**Most common reasons (non-signal bars only)**")
                        st.dataframe(top, width="stretch", height=180)
                else:
                    st.info(
                        "No rows match the current symbol and row filter selection."
                    )

    with st.expander("Preview backtest.csv"):
        st.dataframe(df.head(50), width="stretch")
