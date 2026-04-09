from datetime import UTC, date

from data_pipeline.news.kaggle_adapter import iter_kaggle_news, iter_kaggle_news_multi


def test_iter_kaggle_news_generic_schema(tmp_path):
    fp = tmp_path / "kaggle_news.csv"
    fp.write_text(
        "\n".join(
            [
                "symbol,published_at,title,summary,body,url",
                "AAPL,2020-01-02T10:30:00Z,Apple beats earnings,Strong quarter,Revenue up,https://ex/a1",
                "AAPL,2020-01-03T12:00:00Z,Apple misses guidance,Weak outlook,Margins down,https://ex/a2",
                "MSFT,2020-01-02T10:30:00Z,Microsoft update,Cloud growth,,https://ex/m1",
            ]
        ),
        encoding="utf-8",
    )

    out = list(
        iter_kaggle_news(
            "AAPL",
            dataset_path=str(fp),
            dataset_key="generic_financial_news",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 5),
        )
    )
    assert len(out) == 2
    assert out[0].symbol == "AAPL"
    assert out[0].published_at.tzinfo == UTC
    assert out[0].summary != ""
    assert "Apple beats earnings" in out[0].text_for_score


def test_iter_kaggle_news_alt_schema(tmp_path):
    fp = tmp_path / "kaggle_news_alt.csv"
    fp.write_text(
        "\n".join(
            [
                "ticker,date,headline,summary,article,link",
                "AAPL,2021-02-01,Positive surprise,Guidance raised,Company reports demand surge,https://ex/1",
                "AAPL,2021-02-02,Penalty announced,Regulatory issue,Fine expected,https://ex/2",
            ]
        ),
        encoding="utf-8",
    )
    out = list(
        iter_kaggle_news(
            "AAPL",
            dataset_path=str(fp),
            dataset_key="headline_time_ticker",
            start_date=date(2021, 2, 1),
            end_date=date(2021, 2, 3),
        )
    )
    assert len(out) == 2
    assert all(o.external_id for o in out)
    assert all(o.content_sha256 for o in out)


def test_iter_kaggle_news_multi_dedupes_cross_source(tmp_path):
    sp = tmp_path / "sp500.csv"
    yg = tmp_path / "yogesh.csv"
    sp.write_text(
        "\n".join(
            [
                "Title,Date",
                "Apple launches new product,2020-01-02T10:30:00Z",
            ]
        ),
        encoding="utf-8",
    )
    yg.write_text(
        "\n".join(
            [
                "ticker,published_at,headline,summary,text,url",
                # Same headline/minute as S&P row; empty URL so dedupe key matches market row.
                "AAPL,2020-01-02T10:30:40Z,Apple launches new product,,details,",
            ]
        ),
        encoding="utf-8",
    )
    out = list(
        iter_kaggle_news_multi(
            "AAPL",
            dataset_pairs=[
                ("sp500_headlines_2008_2024", str(sp)),
                ("yogeshchary_financial_news", str(yg)),
            ],
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 5),
        )
    )
    assert len(out) == 1
