"""Fetch S&P 500 tickers and normalize for yfinance-style symbols."""

from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_yfinance_symbol(symbol: str) -> str:
    """Map Wikipedia-style tickers to yfinance (e.g. BRK.B -> BRK-B)."""
    s = symbol.strip().upper()
    if not s:
        return s
    return s.replace(".", "-")


def fetch_sp500_symbols_from_wikipedia(
    *,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> list[str]:
    """
    Parse the current S&P 500 constituents table from Wikipedia.

    Returns tickers normalized for yfinance (class shares use ``-`` not ``.``).
    """
    sess = session or requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ai-finance-platform/1.0; +https://github.com/)"
        )
    }
    r = sess.get(WIKI_URL, timeout=timeout, headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError(
            "Wikipedia S&P 500 table not found (id=constituents). Page layout may have changed."
        )
    symbols: list[str] = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if not tds:
            continue
        raw = tds[0].get_text(strip=True)
        if not raw:
            continue
        # Drop footnotes like "BRK.B[note 1]"
        raw = re.split(r"\[", raw, maxsplit=1)[0].strip()
        sym = normalize_yfinance_symbol(raw)
        if sym and sym not in symbols:
            symbols.append(sym)
    if len(symbols) < 400:
        raise RuntimeError(
            f"Expected ~500 S&P 500 symbols; got {len(symbols)}. Refusing to write partial list."
        )
    return sorted(symbols)


def write_symbol_file(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(symbols) + ("\n" if symbols else "")
    path.write_text(text, encoding="utf-8")


def read_symbol_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Universe file not found: {path}. Run: python -m universe"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip().upper()
        if s and not s.startswith("#"):
            out.append(normalize_yfinance_symbol(s))
    return sorted(set(out))
