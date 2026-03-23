CREATE TABLE IF NOT EXISTS stock_prices (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(12,4),
    high NUMERIC(12,4),
    low NUMERIC(12,4),
    close NUMERIC(12,4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);


CREATE INDEX IF NOT EXISTS idx_symbol_timestamp
ON stock_prices(symbol, timestamp);

CREATE TABLE IF NOT EXISTS stock_features (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,

    close NUMERIC NOT NULL,

    -- returns
    return_1d NUMERIC,
    return_5d NUMERIC,

    -- trend
    sma_5 NUMERIC,
    sma_10 NUMERIC,
    ema_10 NUMERIC,

    -- volatility
    volatility_5 NUMERIC,

    -- lag features
    lag_1 NUMERIC,
    lag_2 NUMERIC,

    PRIMARY KEY (symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp
ON stock_features(symbol, timestamp);