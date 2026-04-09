"""Kaggle dataset contracts and schema mappings for historical news."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KaggleDatasetSpec:
    name: str
    source_url: str
    symbol_col: str
    published_at_col: str
    title_col: str
    summary_col: str | None
    body_col: str | None
    url_col: str | None
    timezone: str = "UTC"


KAGGLE_DATASETS: dict[str, KaggleDatasetSpec] = {
    "sp500_headlines_2008_2024": KaggleDatasetSpec(
        name="sp500_headlines_2008_2024",
        source_url="https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks-news-headlines",
        symbol_col="",  # market-level dataset has no ticker column
        published_at_col="Date",
        title_col="Title",
        summary_col=None,
        body_col=None,
        url_col=None,
        timezone="UTC",
    ),
    "yogeshchary_financial_news": KaggleDatasetSpec(
        name="yogeshchary_financial_news",
        source_url="https://www.kaggle.com/datasets/yogeshchary/financial-news-dataset/data",
        symbol_col="ticker",
        published_at_col="published_at",
        title_col="headline",
        summary_col="summary",
        body_col="text",
        url_col="url",
        timezone="UTC",
    ),
    # Generic schema for curated per-symbol financial headlines.
    "generic_financial_news": KaggleDatasetSpec(
        name="generic_financial_news",
        source_url="https://www.kaggle.com/datasets",
        symbol_col="symbol",
        published_at_col="published_at",
        title_col="title",
        summary_col="summary",
        body_col="body",
        url_col="url",
        timezone="UTC",
    ),
    # Alternate common column style.
    "headline_time_ticker": KaggleDatasetSpec(
        name="headline_time_ticker",
        source_url="https://www.kaggle.com/datasets",
        symbol_col="ticker",
        published_at_col="date",
        title_col="headline",
        summary_col="summary",
        body_col="article",
        url_col="link",
        timezone="UTC",
    ),
}

