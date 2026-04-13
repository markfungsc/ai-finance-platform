from feature_schema import BASE_FEATURE_COLUMNS, z_columns

FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)

FEATURE_COLUMNS_MARKET_CONTEXT = [
    "spy_return_1d",
    "spy_return_5d",
    "spy_volatility_20",
    "spy_sma_20",
    "spy_ema_trend_bull",
    "spy_ema_slope_20",
    "qqq_return_1d",
    "qqq_return_5d",
    "qqq_volatility_20",
    "qqq_sma_20",
    "qqq_ema_trend_bull",
    "qqq_ema_slope_20",
    "vix_level",
    "vix_change",
    "vix_return_5d",
    "vix_volatility_20",
    "vix_sma_20",
]

# Sentiment daily-only stream: symbol + market(SPY) as separate features.
FEATURE_COLUMNS_SENTIMENT_Z = [
    "sym_sentiment_d1",
    "sym_news_volume_d1",
    "sym_sentiment_vol_d1",
    "spy_sentiment_d1",
    "spy_news_volume_d1",
    "spy_sentiment_vol_d1",
]

FEATURE_COLUMNS_MARKET_CONTEXT_Z = [
    "spy_return_1d_z",
    "spy_return_5d_z",
    "spy_volatility_20_z",
    "spy_sma_20_z",
    "spy_ema_trend_bull_z",
    "spy_ema_slope_20_z",
    "spy_return_5d_z_minus_1d_z",
    "qqq_return_1d_z",
    "qqq_return_5d_z",
    "qqq_volatility_20_z",
    "qqq_sma_20_z",
    "qqq_ema_trend_bull_z",
    "qqq_ema_slope_20_z",
    "qqq_return_5d_z_minus_1d_z",
    "vix_level_z",
    "vix_change_z",
    "vix_return_5d_z",
    "vix_volatility_20_z",
    "vix_sma_20_z",
    "vix_high_vol_z",
]

FEATURE_COLUMNS_Z = z_columns(BASE_FEATURE_COLUMNS)

TARGET_COLUMN = "trade_success"
