"""Tests for S&P 500 universe fetch and symbol resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from universe.resolve import resolve_ingestion_symbols
from universe.sp500 import (
    normalize_yfinance_symbol,
    write_symbol_file,
)


def test_normalize_yfinance_symbol() -> None:
    assert normalize_yfinance_symbol("BRK.B") == "BRK-B"
    assert normalize_yfinance_symbol("aapl") == "AAPL"


def test_resolve_subscriptions_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGESTION_UNIVERSE", raising=False)
    syms = resolve_ingestion_symbols()
    assert "AAPL" in syms
    assert "SPY" in syms


def test_resolve_sp500_reads_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "sp500.txt"
    write_symbol_file(p, ["ZZZ", "AAA", "BRK-B"])
    monkeypatch.setenv("INGESTION_UNIVERSE", "sp500")
    monkeypatch.setenv("SP500_SYMBOLS_FILE", str(p))
    syms = resolve_ingestion_symbols()
    assert "AAA" in syms and "ZZZ" in syms
    assert "SPY" in syms and "QQQ" in syms and "^VIX" in syms

