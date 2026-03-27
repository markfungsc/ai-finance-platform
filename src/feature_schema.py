"""Shared feature schema constants used by ML and DB layers."""

BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_100",
    "ema_10",
    "ema_20",
    "ema_50",
    "ema_100",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_50",
    "volatility_100",
    "lag_1",
    "lag_2",
    "sma_200",
    "ema_200",
    "ema_trend_bull",
    "ema_slope_20",
    "rsi_14",
    "rsi_21",
    "macd",
    "macd_signal",
    "roc_5",
    "roc_10",
    "stochastic_k",
    "stochastic_d",
    "atr_14",
    "volatility_ratio",
    "close_vs_high_10",
    "close_vs_low_10",
)


def z_columns(cols: tuple[str, ...] | list[str]) -> list[str]:
    return [f"{c}_z" for c in cols]
