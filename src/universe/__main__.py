"""CLI: fetch S&P 500 tickers from Wikipedia and write ``data/universe/sp500_symbols.txt``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from universe.paths import DEFAULT_SP500_SYMBOLS_FILE
from universe.sp500 import fetch_sp500_symbols_from_wikipedia, write_symbol_file


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch S&P 500 symbols and save to a text file.")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_SP500_SYMBOLS_FILE,
        help=f"Output path (default: {DEFAULT_SP500_SYMBOLS_FILE})",
    )
    args = p.parse_args(argv)
    syms = fetch_sp500_symbols_from_wikipedia()
    write_symbol_file(args.output, syms)
    print(f"Wrote {len(syms)} symbols to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
