-- News medallion: bronze raw, silver clean, gold daily aggregates for ML joins.
-- Vectors live in Qdrant; Postgres stores metadata and reproducible keys.

CREATE TABLE IF NOT EXISTS raw_news_articles (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    raw_payload JSONB NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_news_sha ON raw_news_articles (content_sha256);

CREATE TABLE IF NOT EXISTS clean_news_articles (
    id BIGSERIAL PRIMARY KEY,
    raw_news_id BIGINT REFERENCES raw_news_articles (id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    url TEXT,
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    published_at TIMESTAMPTZ NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    finbert_scalar DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (content_sha256)
);

CREATE INDEX IF NOT EXISTS idx_clean_news_symbol_pub ON clean_news_articles (symbol, published_at);

CREATE TABLE IF NOT EXISTS daily_symbol_sentiment (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    news_sentiment_mean_z DOUBLE PRECISION NOT NULL DEFAULT 0,
    article_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_sentiment_date ON daily_symbol_sentiment (as_of_date);
