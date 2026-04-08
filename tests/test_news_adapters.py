from datetime import UTC, date

from data_pipeline.news.gdelt_adapter import iter_gdelt_news
from data_pipeline.news.sec_adapter import iter_sec_news


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


def test_gdelt_adapter_strict_filter(monkeypatch):
    def _fake_get(_url, params=None, timeout=None):
        return _Resp(
            {
                "articles": [
                    {
                        "title": "AAPL reports earnings beat",
                        "seendate": "2015-01-02T12:00:00Z",
                        "url": "https://example.com/aapl",
                    },
                    {
                        "title": "Apple recipe ideas for weekend",
                        "seendate": "2015-01-02T13:00:00Z",
                        "url": "https://example.com/noise",
                    },
                ]
            }
        )

    monkeypatch.setattr("data_pipeline.news.gdelt_adapter.requests.get", _fake_get)
    out = list(iter_gdelt_news("AAPL", date(2015, 1, 2), date(2015, 1, 2)))
    assert len(out) == 1
    assert out[0].symbol == "AAPL"
    assert out[0].url == "https://example.com/aapl"


def test_sec_adapter_yields_recent_forms(monkeypatch):
    tickers_payload = {"0": {"ticker": "AAPL", "cik_str": 320193}}
    subs_payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "S-8"],
                "filingDate": ["2015-01-02", "2015-01-03"],
                "accessionNumber": ["0000320193-15-000001", "0000320193-15-000002"],
                "primaryDocument": ["a8k.htm", "s8.htm"],
                "isInlineXBRL": [1, 0],
            }
        }
    }

    def _fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _Resp(tickers_payload)
        return _Resp(subs_payload)

    monkeypatch.setattr("data_pipeline.news.sec_adapter.requests.get", _fake_get)
    out = list(iter_sec_news("AAPL", date(2015, 1, 1), date(2015, 1, 5)))
    # S-8 is filtered out by precision policy.
    assert len(out) == 1
    assert out[0].external_id.startswith("sec:")
    assert out[0].published_at.tzinfo == UTC
