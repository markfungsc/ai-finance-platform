import pandas as pd

from log_config import get_logger
from ml.features import (
    FEATURE_COLUMNS_MARKET_CONTEXT_Z,
    FEATURE_COLUMNS_SENTIMENT_Z,
    FEATURE_COLUMNS_Z,
    TARGET_COLUMN,
)
from ml.helpers.generate_trade_labels import generate_trade_labels

logger = get_logger(__name__)


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
    logger.debug(
        "[merge_features_with_target] %s — last %d of %d rows", name, k, len(frame)
    )
    if len(frame) == 0:
        logger.debug("  (empty)")
        return
    cols = [c for c in _DEBUG_TABLE_COLS if c in frame.columns]
    if not cols:
        logger.debug("  (none of: %s)", ", ".join(_DEBUG_TABLE_COLS))
        return
    table = frame.loc[:, cols].tail(n)
    logger.debug("%s", table.to_string(index=False))


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

    X = df_labeled[
        FEATURE_COLUMNS_Z
        + FEATURE_COLUMNS_MARKET_CONTEXT_Z
        + FEATURE_COLUMNS_SENTIMENT_Z
    ].reset_index(drop=True)
    y = df_labeled[TARGET_COLUMN].reset_index(drop=True)
    df_labeled = df_labeled.reset_index(drop=True)

    return X, y, df_labeled
