from src.eval.timebox import filter_items_before_as_of


def test_filter_items_before_as_of_removes_future_and_unparseable_by_default():
    items = [
        {"title": "past", "pub_date": "2024-01-01T00:00:00Z"},
        {"title": "future", "pub_date": "2025-01-01T00:00:00Z"},
        {"title": "unknown", "pub_date": "not-a-date"},
    ]

    kept, removed = filter_items_before_as_of(items, as_of_time="2024-06-01T00:00:00Z")

    assert len(kept) == 1
    assert kept[0]["title"] == "past"
    assert removed == 2


def test_filter_items_before_as_of_can_keep_unparseable():
    items = [
        {"title": "past", "publishedDate": "2024-01-01T00:00:00Z"},
        {"title": "unknown", "publishedDate": "not-a-date"},
    ]

    kept, removed = filter_items_before_as_of(
        items,
        as_of_time="2024-06-01T00:00:00Z",
        keep_unparseable=True,
    )

    assert len(kept) == 2
    assert removed == 0
