from feature_schema import BASE_FEATURE_COLUMNS, z_columns

FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)

FEATURE_COLUMNS_MARKET_CONTEXT = [
    "spy_return_1d",
    "spy_return_5d",
    "spy_volatility_20",
    "spy_sma_20",
    "spy_ema_trend_bull",
    "spy_ema_slope_20",
    "vix_level",
    "vix_change",
    "vix_return_5d",
    "vix_volatility_20",
    "vix_sma_20",
]

# Sentiment: DB/rollup uses per-symbol rolling z (see rollup_daily); names kept without _z suffix.
FEATURE_COLUMNS_SENTIMENT_Z = [
    "news_sentiment_mean_z",
    "sentiment_1h",
    "sentiment_24h",
    "sentiment_3d",
    "news_volume",
    "sentiment_volatility",
]

FEATURE_COLUMNS_MARKET_CONTEXT_Z = [
    "spy_return_1d_z",
    "spy_return_5d_z",
    "spy_volatility_20_z",
    "spy_sma_20_z",
    "spy_ema_trend_bull_z",
    "spy_ema_slope_20_z",
    "spy_return_5d_z_minus_1d_z",
    "vix_level_z",
    "vix_change_z",
    "vix_return_5d_z",
    "vix_volatility_20_z",
    "vix_sma_20_z",
    "vix_high_vol_z",
]

FEATURE_COLUMNS_Z = z_columns(BASE_FEATURE_COLUMNS)

TARGET_COLUMN = "trade_success"
