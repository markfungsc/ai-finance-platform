"""Paths for cached universe files (repo-root relative)."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SP500_SYMBOLS_FILE = _REPO_ROOT / "data" / "universe" / "sp500_symbols.txt"
