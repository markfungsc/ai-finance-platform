import pandas as pd

from ml.features import (
    FEATURE_COLUMNS_MARKET_CONTEXT_Z,
    FEATURE_COLUMNS_Z,
    TARGET_COLUMN,
)
from ml.helpers.generate_trade_labels import generate_trade_labels

_DEBUG_TABLE_COLS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "return_1d",
    "return_5d",
    "return_1d_z",
    "return_5d_z",
    "trade_success",
)


def _debug_print_tail(name: str, frame: pd.DataFrame, n: int = 10) -> None:
    k = min(n, len(frame))
    print(f"\n[merge_features_with_target] {name} — last {k} of {len(frame)} rows")
    if len(frame) == 0:
        print("  (empty)")
        return
    cols = [c for c in _DEBUG_TABLE_COLS if c in frame.columns]
    if not cols:
        print("  (none of: " + ", ".join(_DEBUG_TABLE_COLS) + ")")
        return
    table = frame.loc[:, cols].tail(n)
    print(table.to_string(index=False))


def merge_features_with_target(
    df: pd.DataFrame,
    df_z: pd.DataFrame,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if debug:
        _debug_print_tail("df (input)", df)
        _debug_print_tail("df_z (input)", df_z)

    df_merged = df.merge(df_z, on=["symbol", "timestamp"], how="inner")
    df_merged = df_merged.reset_index(drop=True)
    if debug:
        _debug_print_tail("df_merged before labeling", df_merged)

    # Generate swing trade labels
    df_labeled = generate_trade_labels(df_merged)
    if debug:
        _debug_print_tail("df_labeled", df_labeled)

    X = df_labeled[FEATURE_COLUMNS_Z + FEATURE_COLUMNS_MARKET_CONTEXT_Z].reset_index(
        drop=True
    )
    y = df_labeled[TARGET_COLUMN].reset_index(drop=True)
    df_labeled = df_labeled.reset_index(drop=True)

    return X, y, df_labeled
