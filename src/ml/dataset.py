from pathlib import Path

from constants import MARKET_CONTEXT_SYMBOLS, TRAIN_SYMBOLS
from database.queries import (
    fetch_features,
    fetch_features_many,
    fetch_features_z,
    fetch_features_z_many,
)
from log_config import get_logger
from ml.helpers.attach_market_context import attach_market_context
from ml.helpers.merge_features import merge_features_with_target
from ml.sentiment.attach import attach_sentiment_features

logger = get_logger(__name__)

_SP500_SYMBOLS_REL = Path("data") / "universe" / "sp500_symbols.txt"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_pooled_dataset_symbols() -> list[str]:
    """Symbols used for pooled training/backtest (S&P 500 list minus market-context tickers).

    Falls back to ``TRAIN_SYMBOLS`` if the universe file is missing or empty.
    """
    path = _repo_root() / _SP500_SYMBOLS_REL
    if not path.is_file():
        logger.warning("Pooled universe file missing at %s; using TRAIN_SYMBOLS", path)
        return list(TRAIN_SYMBOLS)
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            tickers.append(s)
    out = sorted({s for s in tickers if s not in MARKET_CONTEXT_SYMBOLS})
    if not out:
        logger.warning("No symbols loaded from %s; using TRAIN_SYMBOLS", path)
        return list(TRAIN_SYMBOLS)
    return out


def load_train_dataset(debug_merge: bool = False):
    syms = get_pooled_dataset_symbols()
    # Sequential fetch + no inner chunk parallelism keeps peak RAM lower (OOM on large pools).
    df_all = fetch_features_many(syms, parallel=False, max_workers=1)
    df_z_all = fetch_features_z_many(syms, parallel=False, max_workers=1)
    df_all = df_all.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df_z_all = df_z_all.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    # single context attach
    df_z_ctx = attach_market_context(df_z_all)
    df_z_ctx = attach_sentiment_features(df_z_ctx)
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
    df_z_context = attach_sentiment_features(df_z_context)
    if not quiet:
        logger.info("Loaded %d rows for %s market context", len(df_z_context), symbol)

    X, y, df_merged = merge_features_with_target(
        df,
        df_z_context,
        debug=debug_merge,
    )

    return X, y, df_merged
