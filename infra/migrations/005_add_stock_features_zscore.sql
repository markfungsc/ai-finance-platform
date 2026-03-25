create table if not exists stock_features_zscore (
    symbol text not null,
    timestamp timestamptz not null,
    close_z numeric not null,
    return_1d_z numeric not null,
    return_5d_z numeric not null,
    return_10d_z numeric not null,
    return_20d_z numeric not null,
    sma_5_z numeric not null,
    sma_10_z numeric not null,
    sma_20_z numeric not null,
    sma_50_z numeric not null,
    sma_100_z numeric not null,
    ema_10_z numeric not null,
    ema_20_z numeric not null,
    ema_50_z numeric not null,
    ema_100_z numeric not null,
    volatility_5_z numeric not null,
    volatility_10_z numeric not null,
    volatility_20_z numeric not null,
    volatility_50_z numeric not null,
    volatility_100_z numeric not null,
    lag_1_z numeric not null,
    lag_2_z numeric not null,
    primary key (symbol, timestamp)
);

create index if not exists idx_features_zscore_symbol_timestamp on stock_features_zscore (symbol, timestamp);