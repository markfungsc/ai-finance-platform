import pandas as pd

from constants import TRAIN_SYMBOLS
from database.queries import fetch_features, fetch_features_z
from log_config import get_logger
from ml.helpers.attach_market_context import attach_market_context
from ml.helpers.merge_features import merge_features_with_target

logger = get_logger(__name__)


def load_train_dataset(debug_merge: bool = False):
    frames = []
    frames_z = []
    for symbol in TRAIN_SYMBOLS:
        df = fetch_features(symbol)
        df_z = fetch_features_z(symbol)
        frames.append(df)
        frames_z.append(df_z)

    df_all = pd.concat(frames, ignore_index=True)
    df_z_all = pd.concat(frames_z, ignore_index=True)
    df_all = df_all.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df_z_all = df_z_all.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    # single context attach
    df_z_ctx = attach_market_context(df_z_all)
    if "symbol" in df_z_ctx.columns and "timestamp" in df_z_ctx.columns:
        shared_ts = (
            df_z_ctx.groupby("timestamp")["symbol"]
            .nunique()
            .loc[lambda s: s > 1]
            .index[:3]
        )
        if len(shared_ts) > 0:
            context_cols = [
                "return_1d_z",
                "return_5d_z",
                "spy_return_1d_z",
                "spy_return_5d_z",
                "spy_volatility_20_z",
                "vix_level_z",
                "vix_change_z",
                "vix_high_vol_z",
            ]
            shown_cols = [
                c
                for c in ["timestamp", "symbol", *context_cols]
                if c in df_z_ctx.columns
            ]
            sample = df_z_ctx.loc[
                df_z_ctx["timestamp"].isin(shared_ts), shown_cols
            ].sort_values(["timestamp", "symbol"])
            logger.info(
                "[load_train_dataset] Market-context sample "
                "(same timestamp, different symbols)\n%s",
                sample.to_string(index=False),
            )
            check_cols = [c for c in shown_cols if c not in {"timestamp", "symbol"}]
            if check_cols:
                per_ts_consistent = (
                    sample.groupby("timestamp")[check_cols]
                    .nunique(dropna=False)
                    .le(1)
                    .all(axis=1)
                )
                logger.info(
                    "[load_train_dataset] Context identical across symbols "
                    "for sampled timestamps: %s",
                    bool(per_ts_consistent.all()),
                )
    X, y, df_merged = merge_features_with_target(df_all, df_z_ctx, debug=debug_merge)
    return X, y, df_merged


def load_dataset(symbol: str, debug_merge: bool = False, quiet: bool = False):
    df = fetch_features(symbol)  # contains high/low/close
    df_z = fetch_features_z(symbol)  # z-score features
    if not quiet:
        logger.info(
            "Loaded %d rows for %s z-scores; df columns: %s",
            len(df_z),
            symbol,
            list(df.columns),
        )

    # attach market context
    df_z_context = attach_market_context(df_z)
    if not quiet:
        logger.info("Loaded %d rows for %s market context", len(df_z_context), symbol)

    X, y, df_merged = merge_features_with_target(
        df,
        df_z_context,
        debug=debug_merge,
    )

    return X, y, df_merged
