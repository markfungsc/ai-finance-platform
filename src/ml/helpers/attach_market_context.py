import pandas as pd

from database.queries import fetch_features_z
from ml.features import FEATURE_COLUMNS_MARKET_CONTEXT_Z


def attach_market_context(df_z: pd.DataFrame) -> pd.DataFrame:
    spy = fetch_features_z("SPY").drop_duplicates(subset=["timestamp"])
    vix = fetch_features_z("^VIX").drop_duplicates(subset=["timestamp"])
    if "symbol" in df_z.columns:
        df_z = df_z.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    else:
        df_z = df_z.sort_values(["timestamp"]).reset_index(drop=True)

    # --- SPY FEATURES ---
    spy_features_z = spy[
        [
            "timestamp",
            "return_1d_z",
            "return_5d_z",
            "volatility_20_z",
            "sma_20_z",
            "sma_50_z",
            "ema_trend_bull_z",
            "ema_slope_20_z",
        ]
    ].copy()

    spy_features_z = spy_features_z.assign(
        spy_return_5d_z_minus_1d_z=(
            spy_features_z["return_5d_z"] - spy_features_z["return_1d_z"]
        ),
    )

    spy_features_z = spy_features_z.rename(
        columns={
            "return_1d_z": "spy_return_1d_z",
            "return_5d_z": "spy_return_5d_z",
            "volatility_20_z": "spy_volatility_20_z",
            "sma_20_z": "spy_sma_20_z",
            "sma_50_z": "spy_sma_50_z",
            "ema_trend_bull_z": "spy_ema_trend_bull_z",
            "ema_slope_20_z": "spy_ema_slope_20_z",
        }
    )
    spy_features_z = spy_features_z.sort_values("timestamp").reset_index(drop=True)

    # --- VIX FEATURES ---
    vix_features_z = vix[
        [
            "timestamp",
            "close_z",
            "return_1d_z",
            "return_5d_z",
            "volatility_20_z",
            "sma_20_z",
        ]
    ].copy()

    vix_features_z = vix_features_z.assign(
        vix_rolling_mean_50=vix_features_z["close_z"].rolling(50).mean()
    )
    vix_features_z = vix_features_z.dropna(subset=["vix_rolling_mean_50"]).copy()
    vix_features_z = vix_features_z.assign(
        vix_high_vol_z=(
            (vix_features_z["close_z"] - vix_features_z["vix_rolling_mean_50"])
            / vix_features_z["vix_rolling_mean_50"]
        )
    )

    vix_features_z = vix_features_z.rename(
        columns={
            "close_z": "vix_level_z",
            "return_1d_z": "vix_change_z",
            "return_5d_z": "vix_return_5d_z",
            "volatility_20_z": "vix_volatility_20_z",
            "sma_20_z": "vix_sma_20_z",
        }
    )
    vix_features_z = vix_features_z.sort_values("timestamp").reset_index(drop=True)

    # --- MERGE ---
    df_z = df_z.merge(spy_features_z, on="timestamp", how="left")
    # merge_asof requires both left/right keys to be sorted by the join key.
    df_z = pd.merge_asof(df_z, vix_features_z, on="timestamp", direction="backward")

    df_z = df_z.dropna(subset=FEATURE_COLUMNS_MARKET_CONTEXT_Z)
    return df_z
