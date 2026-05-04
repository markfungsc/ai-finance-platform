from ml.sentiment.embed_sync import _point_id, embed_and_upsert_article_ids


def test_point_id_stable():
    assert _point_id(54181, 0) == 54181 * 10_000
    assert _point_id(54181, 1) == 54181 * 10_000 + 1


def test_embed_and_upsert_article_ids_empty_returns_zero():
    assert embed_and_upsert_article_ids("NVDA", []) == 0
