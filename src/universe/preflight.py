"""Optional DB coverage check for the resolved ingestion universe (scanner readiness)."""

from __future__ import annotations

import os
import sys

from database.queries import (
    count_symbols_with_clean_rows,
    count_symbols_with_stock_features,
    list_symbols_missing_stock_features,
)
from universe.resolve import resolve_ingestion_symbols


def main() -> int:
    mode = os.environ.get("INGESTION_UNIVERSE", "subscriptions").strip()
    expected = resolve_ingestion_symbols()
    n_exp = len(expected)
    n_clean = count_symbols_with_clean_rows(expected)
    n_feat = count_symbols_with_stock_features(expected)
    print(f"INGESTION_UNIVERSE={mode!r}")
    print(f"Resolved symbols: {n_exp}")
    print(f"Symbols with clean_stock_prices rows: {n_clean} / {n_exp}")
    print(f"Symbols with stock_features rows: {n_feat} / {n_exp}")
    if n_exp and n_feat < n_exp:
        missing = list_symbols_missing_stock_features(expected)
        if missing:
            print(
                "\nSymbols with no stock_features rows (investigate or re-run features):",
                file=sys.stderr,
            )
            for sym in missing:
                print(f"  - {sym}", file=sys.stderr)
        print(
            "\nTip: run ingestion → clean → features for this INGESTION_UNIVERSE; "
            "for a single ticker use FEATURES_BACKFILL=1 if history is short.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
