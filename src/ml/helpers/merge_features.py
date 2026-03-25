import pandas as pd

from ml.features import FEATURE_COLUMNS_Z, TARGET_COLUMN

_DEBUG_TABLE_COLS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "return_1d",
    "return_5d",
    "return_1d_z",
    "return_5d_z",
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
    target_shift: int = 5,
    *,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if debug:
        _debug_print_tail("df (input)", df)
        _debug_print_tail("df_z (input)", df_z)

    df_raw = df.copy()
    df_raw[TARGET_COLUMN] = df_raw["return_5d"].shift(-target_shift)
    df_raw = df_raw.dropna(subset=[TARGET_COLUMN])

    if debug:
        _debug_print_tail("df_raw (after target + dropna)", df_raw)

    df_merged = df_raw.merge(df_z, on=["symbol", "timestamp"], how="inner")
    X = df_merged[FEATURE_COLUMNS_Z].reset_index(drop=True)
    y = df_merged[TARGET_COLUMN].reset_index(drop=True)

    if debug:
        _debug_print_tail("df_merged (inner join)", df_merged)

    return X, y, df_merged
