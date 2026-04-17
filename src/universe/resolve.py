"""Resolve which symbols to ingest / build features for (env-driven)."""

from __future__ import annotations

import os
from pathlib import Path

from constants import MARKET_CONTEXT_SYMBOLS, SUBSCRIPTIONS
from universe.paths import DEFAULT_SP500_SYMBOLS_FILE
from universe.sp500 import read_symbol_file


def _sp500_path() -> Path:
    raw = os.environ.get("SP500_SYMBOLS_FILE", str(DEFAULT_SP500_SYMBOLS_FILE)).strip()
    return Path(raw).expanduser()


def resolve_ingestion_universe() -> tuple[str, list[str]]:
    """
    ``INGESTION_UNIVERSE`` (default ``subscriptions``):

    - ``subscriptions``: ``SUBSCRIPTIONS`` from constants only.
    - ``sp500``: S&P 500 list from ``SP500_SYMBOLS_FILE`` (default
      ``data/universe/sp500_symbols.txt``) plus market context symbols
      (QQQ, SPY, ^VIX) for downstream training context. Megacap names in
      ``SUBSCRIPTIONS`` are already constituents; no separate union mode.
    """
    mode = os.environ.get("INGESTION_UNIVERSE", "sp500").strip().lower()
    ctx = sorted(MARKET_CONTEXT_SYMBOLS)

    if mode == "subscriptions":
        return mode, sorted(SUBSCRIPTIONS)

    if mode == "sp500":
        sp = read_symbol_file(_sp500_path())
        return mode, sorted(set(sp + ctx))

    raise ValueError(
        f"INGESTION_UNIVERSE must be one of: subscriptions, sp500 (got {mode!r})"
    )


def resolve_ingestion_symbols() -> list[str]:
    _mode, symbols = resolve_ingestion_universe()
    return symbols
